"""Belief viewport overlay for SO-101 Isaac Lab tasks (RSS-20-style).

Draws the online object-pose belief directly in the Isaac Sim viewport as
lightweight USD prims, restyled per project request:

- one small SPHERE per particle, **low alpha**, color ramped yellow -> red by
  normalized weight (probability mass reads at a glance),
- two semi-transparent ORANGE square OUTLINES for the compressed two-mode
  belief (what the controller/policy actually consumes),
- a CYAN flat PUCK + drop-line at the belief-mean grasp target ("where the
  robot needs to go").

Zero dependency on the RL stack: it reads whatever callable you bind. The
default binding expects the env-side filter port planned in
BELIEF_INTEGRATION.md §2 (state dataclass with ``xy_w``/``weights`` stored on
``env.unwrapped._so101_belief``). Until that lands, run the built-in fake-belief
demo to validate rendering::

    python -m envs.isaac.scripts.belief_markers            # animated fake belief
    python -m envs.isaac.scripts.belief_markers --collapse # ...converging to truth

Usage inside a real play loop (after the filter is ported)::

    from envs.isaac.scripts.belief_markers import BeliefViewportOverlay

    overlay = BeliefViewportOverlay(num_particles=256)
    overlay.bind_env(env)          # env = unwrapped ManagerBasedRLEnv
    while simulation_app.is_running():
        ... env.step(...) ...
        overlay.update()
        simulation_app.update()

All tensors may live on GPU; everything is copied to CPU once per frame.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field

import torch


@dataclass
class BeliefVizCfg:
    """Styling for :class:`BeliefViewportOverlay`."""

    particle_radius: float = 0.0045
    """Sphere radius per particle [m]."""

    particle_alpha_range: tuple[float, float] = (0.06, 0.75)
    """Display-opacity range mapped from lowest- to highest-weight particle."""

    particle_weight_power: float = 2.0
    """Exponent on normalized weights before mapping alpha (>1 emphasizes the mass)."""

    color_low: tuple[float, float, float] = (1.0, 0.85, 0.10)   # pale yellow
    color_high: tuple[float, float, float] = (0.95, 0.15, 0.05)  # strong red

    mode_color: tuple[float, float, float] = (1.0, 0.45, 0.0)   # orange
    mode_alpha: float = 0.55
    mode_outline_thickness: float = 0.0035
    mode_side_length: float = 0.038  # ~object footprint

    target_color: tuple[float, float, float] = (0.10, 0.90, 0.95)  # cyan
    target_alpha: float = 0.80
    target_puck_radius: float = 0.014
    target_puck_height: float = 0.004

    particle_z: float = 0.010   # hover height above the table for particles
    mode_z: float = 0.006
    target_z: float = 0.002     # puck sits on the table surface

    max_drawn_particles: int = 256  # subsample if the filter holds more


@dataclass
class _Prim:
    prim: object
    translate: object
    scale: object | None = None


class BeliefViewportOverlay:
    """Create/update the belief marker prims on the current USD stage."""

    def __init__(
        self,
        num_particles: int = 256,
        root_path: str = "/World/SO101Belief",
        cfg: BeliefVizCfg | None = None,
    ) -> None:
        import omni.usd
        from pxr import Gf, UsdGeom

        self._Gf = Gf
        self._UsdGeom = UsdGeom
        self._cfg = cfg or BeliefVizCfg()
        self._num_particles = min(num_particles, self._cfg.max_drawn_particles)
        self._stage = omni.usd.get_context().get_stage()

        UsdGeom.Xform.Define(self._stage, root_path)
        UsdGeom.Xform.Define(self._stage, f"{root_path}/Particles")
        UsdGeom.Xform.Define(self._stage, f"{root_path}/Modes")
        UsdGeom.Xform.Define(self._stage, f"{root_path}/Target")

        self._particles: list[_Prim] = []
        for i in range(self._num_particles):
            sphere = UsdGeom.Sphere.Define(self._stage, f"{root_path}/Particles/p_{i:03d}")
            sphere.CreateRadiusAttr(self._cfg.particle_radius)
            self._particles.append(_Prim(sphere.GetPrim(), *self._xformops(sphere.GetPrim())))

        self._mode_edges: list[list[_Prim]] = []
        for m in range(2):
            edges = []
            for e, name in enumerate(("front", "back", "left", "right")):
                box = UsdGeom.Cube.Define(self._stage, f"{root_path}/Modes/m{m}_{name}")
                box.CreateSizeAttr(1.0)
                edges.append(_Prim(box.GetPrim(), *self._xformops(box.GetPrim())))
                self._style(edges[-1].prim, self._cfg.mode_color, self._cfg.mode_alpha)
            self._mode_edges.append(edges)

        puck = UsdGeom.Cylinder.Define(self._stage, f"{root_path}/Target/go_puck")
        puck.CreateRadiusAttr(self._cfg.target_puck_radius)
        puck.CreateHeightAttr(self._cfg.target_puck_height)
        self._target_puck = _Prim(puck.GetPrim(), *self._xformops(puck.GetPrim()))
        line = UsdGeom.Cylinder.Define(self._stage, f"{root_path}/Target/go_line")
        line.CreateRadiusAttr(0.0015)
        self._target_line = _Prim(line.GetPrim(), *self._xformops(line.GetPrim()))
        for p in (self._target_puck, self._target_line):
            self._style(p.prim, self._cfg.target_color, self._cfg.target_alpha)

        # cache last-drawn styles to skip redundant attribute writes
        self._last_colors: list[tuple] = [(0.0, 0.0, 0.0)] * self._num_particles
        self._last_alphas: list[float] = [-1.0] * self._num_particles
        self._bound = None  # callable -> dict(xy, weights, modes, target) or None

    # ------------------------------------------------------------------
    # binding
    # ------------------------------------------------------------------

    def bind(self, get_state) -> None:
        """Bind a callable returning ``None`` or a dict with keys:

        - ``xy``      : torch.Tensor ``[N, 2]`` particle positions (world XY)
        - ``weights`` : torch.Tensor ``[N]``   normalized particle weights
        - ``modes``   : torch.Tensor ``[2, 2]`` two-component centers (world XY)
        - ``target``  : torch.Tensor ``[2]``   belief-mean go-to point (world XY)
        """
        self._bound = get_state

    def bind_env(self, env, belief_attr: str = "_so101_belief", target_from_goal: bool = True) -> None:
        """Default binding for the planned env-side filter (§2 of
        BELIEF_INTEGRATION.md): reads ``getattr(env, belief_attr)`` with fields
        ``xy_w`` ``[num_envs, N, 2]``, ``weights`` ``[num_envs, N]``, and the
        two-mode centers helper. Env 0 is visualized."""

        def get_state():
            state = getattr(env, belief_attr, None)
            if state is None:
                return None
            xy = state.xy_w[0]
            w = state.weights[0]
            w = w / w.sum().clamp_min(1e-8)
            spread = torch.sqrt((w * (xy - (w.unsqueeze(-1) * xy).sum(0)).square()).sum(0).clamp_min(1e-6))
            centers = torch.stack(
                (
                    (w.unsqueeze(-1) * xy).sum(0) + torch.tensor((0.0, -spread[1]), device=xy.device),
                    (w.unsqueeze(-1) * xy).sum(0) + torch.tensor((0.0, spread[1]), device=xy.device),
                ),
                dim=0,
            )
            target = (w.unsqueeze(-1) * xy).sum(0)  # belief mean = go-to estimate
            return {"xy": xy, "weights": w, "modes": centers, "target": target}

        self.bind(get_state)

    # ------------------------------------------------------------------
    # per-frame update
    # ------------------------------------------------------------------

    def update(self) -> bool:
        """Redraw markers from the bound callable. Returns False when unbound
        or when the producer returned None (e.g., filter not initialized yet).
        Safe to call every frame."""
        if self._bound is None:
            return False
        s = self._bound()
        if s is None or s.get("xy") is None:
            return False

        xy = s["xy"].detach().cpu()
        w = s["weights"].detach().cpu().clamp_min(1e-12)
        n = min(xy.shape[0], self._num_particles)
        if xy.shape[0] > n:  # posterior-sample the drawn subset
            idx = torch.multinomial(w / w.sum(), n, replacement=True)
            xy, w = xy[idx], w[idx]

        wn = (w / w.sum()).pow(self._cfg.particle_weight_power)
        lo_a, hi_a = self._cfg.particle_alpha_range
        alphas = lo_a + (hi_a - lo_a) * (wn / wn.max().clamp_min(1e-12))
        t = (wn / wn.max().clamp_min(1e-12)).unsqueeze(-1)
        lo_c = torch.tensor(self._cfg.color_low)
        hi_c = torch.tensor(self._cfg.color_high)
        colors = lo_c + (hi_c - lo_c) * t
        z = self._cfg.particle_z

        for i, p in enumerate(self._particles):
            p.translate.Set(self._Gf.Vec3d(float(xy[i, 0]), float(xy[i, 1]), z))
            c = tuple(colors[i].tolist())
            a = float(alphas[i])
            if c != self._last_colors[i]:
                self._set_display_color(p.prim, c)
                self._last_colors[i] = c
            if abs(a - self._last_alphas[i]) > 0.01:
                self._set_opacity(p.prim, a)
                self._last_alphas[i] = a

        for m, center in enumerate(s["modes"].detach().cpu()):
            self._draw_mode_outline(m, float(center[0]), float(center[1]))

        tgt = s["target"].detach().cpu()
        self._draw_target(float(tgt[0]), float(tgt[1]))
        return True

    # ------------------------------------------------------------------
    # USD helpers
    # ------------------------------------------------------------------

    def _xformops(self, prim):
        xf = self._UsdGeom.Xformable(prim)
        return xf.AddTranslateOp(precision=self._UsdGeom.XformOp.PrecisionDouble), \
            xf.AddScaleOp(precision=self._UsdGeom.XformOp.PrecisionDouble)

    def _style(self, prim, color, alpha):
        img = self._UsdGeom.Imageable(prim)
        img.CreateDisplayColorAttr([self._Gf.Vec3f(*color)])
        img.CreateDisplayOpacityAttr([alpha])

    def _set_display_color(self, prim, color):
        self._UsdGeom.Imageable(prim).CreateDisplayColorAttr([self._Gf.Vec3f(*color)])

    def _set_opacity(self, prim, alpha):
        self._UsdGeom.Imageable(prim).CreateDisplayOpacityAttr([alpha])

    def _draw_mode_outline(self, mode_index: int, cx: float, cy: float):
        side = self._cfg.mode_side_length
        half = 0.5 * side
        th = self._cfg.mode_outline_thickness
        zs = self._cfg.mode_z
        specs = (
            ((cx, cy + half, zs), (side + th, th, th)),
            ((cx, cy - half, zs), (side + th, th, th)),
            ((cx - half, cy, zs), (th, side + th, th)),
            ((cx + half, cy, zs), (th, side + th, th)),
        )
        for edge, (pos, scale) in zip(self._mode_edges[mode_index], specs):
            edge.translate.Set(self._Gf.Vec3d(*pos))
            edge.scale.Set(self._Gf.Vec3d(*scale))

    def _draw_target(self, tx: float, ty: float):
        zc = self._cfg.target_z + 0.5 * self._cfg.target_puck_height
        self._target_puck.translate.Set(self._Gf.Vec3d(tx, ty, zc))
        hz = self._cfg.target_puck_height + 0.05
        self._target_line.translate.Set(self._Gf.Vec3d(tx, ty, 0.5 * hz))
        self._target_line.scale.Set(self._Gf.Vec3d(1.0, 1.0, hz))


# ----------------------------------------------------------------------
# Fake-belief demo (works before the real filter is ported)
# ----------------------------------------------------------------------


@dataclass
class FakeBimodalBelief:
    """Synthetic bimodal belief wandering toward convergence (viz smoke test)."""

    num_particles: int = 256
    true_xy: tuple[float, float] = (0.25, 0.0)
    mode_offset_y: float = 0.075
    initial_std: tuple[float, float] = (0.055, 0.028)
    final_std: tuple[float, float] = (0.006, 0.004)
    seed: int = 7
    _t: float = field(default=0.0, repr=False)

    def __post_init__(self):
        g = torch.Generator().manual_seed(self.seed)
        self._ids = torch.arange(self.num_particles) % 2
        self._noise0 = torch.randn(self.num_particles, 2, generator=g)
        self._noise1 = torch.randn(self.num_particles, 2, generator=g)

    def step(self, dt: float = 1 / 30):
        self._t += dt

    def state(self) -> dict:
        wander_x = self.true_xy[0] + 0.01 * math.sin(0.7 * self._t)
        wander_y = self.true_xy[1] + 0.02 * math.sin(0.4 * self._t)
        collapse = 0.5 * (1.0 + math.sin(0.35 * self._t))  # 0..1..0 breathing
        true = torch.tensor([wander_x, wander_y])
        offsets = torch.where((self._ids == 0).unsqueeze(-1),
                              torch.tensor([0.0, -self.mode_offset_y]),
                              torch.tensor([0.0, self.mode_offset_y]))
        broad = true + offsets + self._noise0 * torch.tensor(self.initial_std)
        tight = true + self._noise1 * torch.tensor(self.final_std)
        xy = (1 - collapse) * broad + collapse * tight

        raw = torch.rand(self.num_particles) + 0.05
        weights = raw / raw.sum()
        top = xy[self._ids == 0].mean(0) if (self._ids == 0).any() else true
        bot = xy[self._ids == 1].mean(0) if (self._ids == 1).any() else true
        modes = torch.stack((top, bot), dim=0)
        mean = (weights.unsqueeze(-1) * xy).sum(0)
        return {"xy": xy, "weights": weights, "modes": modes, "target": mean}


def main():
    parser = argparse.ArgumentParser(description="SO-101 belief overlay smoke test (fake belief)")
    parser.add_argument("--collapse", action="store_true", help="animate convergence to the true pose")
    parser.add_argument("--particles", type=int, default=256)
    parser.add_argument("--steps", type=int, default=600)
    args, _ = parser.parse_known_args()

    from isaaclab.app import AppLauncher

    launcher = AppLauncher(headless=False)
    simulation_app = launcher.app

    import omni.usd
    from pxr import UsdGeom

    try:
        stage = omni.usd.get_context().get_stage()
        UsdGeom.Xform.Define(stage, "/World")
        # ground reference so motion is visible
        ground = UsdGeom.Cube.Define(stage, "/World/Ground")
        ground.CreateSizeAttr(1.0)
        xf = UsdGeom.Xformable(ground.GetPrim())
        tr = xf.AddTranslateOp()
        sc = xf.AddScaleOp()
        tr.Set((0.25, 0.0, -0.005))
        sc.Set((0.5, 0.6, 0.01))

        fake = FakeBimodalBelief(num_particles=args.particles)
        if args.collapse:
            fake.final_std = (0.002, 0.002)  # snap tighter when collapsed

        overlay = BeliefViewportOverlay(num_particles=args.particles)
        overlay.bind(fake.state)

        for _ in range(args.steps):
            fake.step()
            overlay.update()
            simulation_app.update()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
