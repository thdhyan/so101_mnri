"""Backend-agnostic RL training entry point for the SO-101 tasks.

The RL algorithm layer (skrl / rsl_rl) is independent of the simulation
backend (Isaac Lab / mjlab / MuJoCo) — pick any combination:

    # Isaac Lab (Isaac Sim 6.0.1) tasks — single-arm and dual-arm
    python -m rl.train --backend isaaclab --task SO101-PickLift-Single-v0  --algo skrl   --num-envs 4096
    python -m rl.train --backend isaaclab --task SO101-CylGrasp-Dual-v0    --algo rsl_rl --num-envs 2048

    # mjlab (MuJoCo Warp GPU batching) task
    python -m rl.train --backend mjlab --task so101_pick_lift --algo skrl --num-envs 256 --no-wandb

    # plain MuJoCo gymnasium envs (single instance)
    python -m rl.train --backend mujoco --task pick_lift --algo skrl --max-iterations 200

Notes:
    - rsl_rl requires a vectorized env (Isaac Lab / mjlab backends); it is
      not available for the single-instance MuJoCo gymnasium envs.
    - Isaac Lab and mjlab need a GPU. The MuJoCo backend runs anywhere.
    - Logging: TensorBoard always (rl/runs/<backend>/...) plus WandB by
      default on ALL paths (--no-wandb to disable). Credentials come from
      ~/.netrc (machine api.wandb.ai); set --wandb-entity or $WANDB_ENTITY
      to override the target team/entity.
"""

from __future__ import annotations

import argparse
import importlib
import os
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model

REPO_ROOT = Path(__file__).resolve().parents[1]

ISAACLAB_TASKS = [
    "SO101-PickLift-Single-v0",
    "SO101-PickPlace-Single-v0",
    "SO101-CylReach-Single-v0",
    "SO101-CylGrasp-Dual-v0",
    "SO101-CylReach-Dual-v0",
]
MJLAB_TASKS = {
    "so101_pick_lift": "rl.tasks.so101_pick_lift.env_cfg:make_pick_lift_env_cfg",
}
MUJOCO_TASKS = {
    "single": "envs.mujoco.so101_single_arm.env:make_env",
    "dual": "envs.mujoco.so101_dual_arm.env:make_env",
    "pick_lift": "envs.mujoco.so101_single_arm_pick_lift.env:make_env",
    "pick_place": "envs.mujoco.so101_single_arm_pick_place.env:make_env",
    "cyl_grasp": "envs.mujoco.so101_dual_arm_cylinder_grasp.env:make_env",
    "cyl_reach": "envs.mujoco.so101_dual_arm_cylinder_reach.env:make_env",
    "cyl_reach_single": "envs.mujoco.so101_single_arm_cylinder_reach.env:make_env",
    "push_t": "envs.mujoco.so101_single_arm_push_t.env:make_env",
    "cube_push_ramp": "envs.mujoco.so101_single_arm_cube_push_ramp.env:make_env",
    "cube_push_bridge": "envs.mujoco.so101_single_arm_cube_push_bridge.env:make_env",
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="SO-101 backend-agnostic RL trainer")
    p.add_argument("--backend", choices=["isaaclab", "mjlab", "mujoco"], required=True)
    p.add_argument("--task", required=True, help="task id / name (backend-specific)")
    p.add_argument("--algo", choices=["skrl", "rsl_rl"], default="skrl")
    p.add_argument("--num-envs", type=int, default=None)
    p.add_argument("--max-iterations", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--headless", action="store_true", default=True)
    p.add_argument("--device", default=None, help="torch device (default cuda:0 if available)")
    p.add_argument("--wandb", dest="wandb", action="store_true", default=True)
    p.add_argument("--no-wandb", dest="wandb", action="store_false")
    p.add_argument("--wandb-project", default="so101-rl")
    p.add_argument(
        "--wandb-entity",
        default=os.environ.get("WANDB_ENTITY"),
        help="WandB entity/team (default: your account default; falls back to $WANDB_ENTITY)",
    )
    p.add_argument("--log-dir", default="rl/runs")
    args = p.parse_args(argv)

    if args.backend == "isaaclab" and args.task not in ISAACLAB_TASKS:
        p.error(f"--task must be one of {ISAACLAB_TASKS} for --backend isaaclab")
    if args.backend == "mjlab" and args.task not in MJLAB_TASKS:
        p.error(f"--task must be one of {list(MJLAB_TASKS)} for --backend mjlab")
    if args.backend == "mujoco" and args.task not in MUJOCO_TASKS:
        p.error(f"--task must be one of {list(MUJOCO_TASKS)} for --backend mujoco")
    if args.backend == "mujoco" and args.algo == "rsl_rl":
        p.error("rsl_rl needs a vectorized env — use --backend isaaclab or mjlab")
    return args


def make_log_dir(args) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    d = REPO_ROOT / args.log_dir / args.backend / f"{args.task.replace('/', '_')}_{args.algo}" / stamp
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# skrl PPO building blocks (shared by the isaaclab and mujoco paths)
# ---------------------------------------------------------------------------


class SkrlPolicy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=observation_space,
            action_space=action_space,
            device=device,
        )
        GaussianMixin.__init__(self, clip_actions=False, role="policy")
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 64), nn.ELU(),
            nn.Linear(64, self.num_actions),
        )
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions, device=self.device))

    def compute(self, states, taken_actions=None, role=None):
        # skrl fork passes the full inputs dict to compute; observations carry
        # the flat obs vector (states is None when the env has no state()).
        if isinstance(states, dict):
            states = states["states"] if states.get("states") is not None else states["observations"]
        return self.net(states), {"log_std": self.log_std_parameter}


class SkrlValue(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=observation_space,
            action_space=action_space,
            device=device,
        )
        DeterministicMixin.__init__(self, clip_actions=False, role="value")
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 64), nn.ELU(),
            nn.Linear(64, 1),
        )

    def compute(self, states, taken_actions=None, role=None):
        if isinstance(states, dict):
            states = states["states"] if states.get("states") is not None else states["observations"]
        return self.net(states), {}


class DropImagesObs(gym.ObservationWrapper):
    """Strip the `images` entry from a dict observation space and flatten the
    rest to a single Box vector (RL uses state obs; cameras stay available on
    the raw env for recording/eval)."""

    def __init__(self, env):
        super().__init__(env)
        space = env.observation_space
        if hasattr(space, "spaces") and "images" in space.spaces:
            space = gym.spaces.Dict({k: v for k, v in space.spaces.items() if k != "images"})
        if hasattr(space, "spaces"):  # Dict -> flat Box (fixed key order)
            self._keys = list(space.spaces.keys())
            low = np.concatenate([np.broadcast_to(s.low, s.shape).ravel() for s in space.spaces.values()])
            high = np.concatenate([np.broadcast_to(s.high, s.shape).ravel() for s in space.spaces.values()])
            self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)
        else:
            self._keys = None
            self.observation_space = space

    def observation(self, observation):
        if self._keys is not None:
            return np.concatenate([np.asarray(observation[k], dtype=np.float32).ravel() for k in self._keys])
        if isinstance(observation, dict) and "images" in observation:
            return {k: v for k, v in observation.items() if k != "images"}
        return observation


def flat_obs_dim(space) -> int:
    """Flattened observation dim for Box or (image-free) Dict spaces."""
    import numpy as np

    if hasattr(space, "spaces"):  # gymnasium Dict
        return sum(int(np.prod(s.shape)) for s in space.spaces.values())
    return int(np.prod(space.shape))


def build_skrl_ppo(wrapped_env, args, log_dir: Path, device: str):
    """PPO agent via the skrl fork's dataclass config API (PPO_CFG)."""
    from skrl.agents.torch.base import ExperimentCfg
    from skrl.agents.torch.ppo import PPO, PPO_CFG
    from skrl.memories.torch import RandomMemory
    from skrl.resources.preprocessors.torch import RunningStandardScaler
    from skrl.resources.schedulers.torch import KLAdaptiveLR
    from skrl.utils import set_seed

    set_seed(args.seed)
    # skrl's global default device follows torch.cuda; pin it to the requested device
    from skrl import config as skrl_config

    skrl_config.torch.device = device
    obs_dim = flat_obs_dim(wrapped_env.observation_space)
    act_shape = tuple(wrapped_env.action_space.shape)

    models = {
        "policy": SkrlPolicy(wrapped_env.observation_space, wrapped_env.action_space, device).to(device),
        "value": SkrlValue(wrapped_env.observation_space, wrapped_env.action_space, device).to(device),
    }
    memory = RandomMemory(memory_size=24, num_envs=wrapped_env.num_envs, device=device)

    experiment = ExperimentCfg(
        directory=str(log_dir),
        experiment_name=f"{args.backend}/{args.task}",
        write_interval=50,
        checkpoint_interval=100,
    )
    if args.wandb:
        experiment.wandb = True
        wandb_kwargs = {
            "project": args.wandb_project,
            "name": f"{args.task}_{args.algo}",
            "dir": str(log_dir),
            "tags": [args.backend, args.algo, args.task],
            # metrics are mirrored to wandb directly (below); the TB-sync path
            # misses short runs because torch's SummaryWriter buffers ~120 s
            "sync_tensorboard": False,
        }
        if args.wandb_entity:
            wandb_kwargs["entity"] = args.wandb_entity
        experiment.wandb_kwargs = wandb_kwargs

    cfg = PPO_CFG(
        rollouts=24,
        learning_epochs=5,
        mini_batches=4,
        discount_factor=0.99,
        gae_lambda=0.95,
        learning_rate=1e-3,
        learning_rate_scheduler=KLAdaptiveLR,
        learning_rate_scheduler_kwargs={"kl_threshold": 0.01},
        observation_preprocessor=RunningStandardScaler,
        observation_preprocessor_kwargs={"size": (obs_dim,), "device": device},
        value_preprocessor=RunningStandardScaler,
        value_preprocessor_kwargs={"size": 1, "device": device},
        grad_norm_clip=1.0,
        ratio_clip=0.2,
        value_clip=0.2,
        entropy_loss_scale=0.0,
        random_timesteps=0,
        learning_starts=0,
        experiment=experiment,
    )

    agent = PPO(
        models=models,
        memory=memory,
        observation_space=wrapped_env.observation_space,
        action_space=wrapped_env.action_space,
        device=device,
        cfg=cfg,
    )

    if args.wandb:
        # Mirror skrl's aggregated tracking data to wandb at each write
        # interval (the fork only syncs TensorBoard, which is unreliable for
        # short runs — see comment on sync_tensorboard above).
        import wandb

        original_write = agent.write_tracking_data

        def write_tracking_data(*, timestep, timesteps):
            if wandb.run is not None and agent.tracking_data:
                metrics = {}
                for k, v in agent.tracking_data.items():
                    if k.endswith("(min)"):
                        metrics[k] = float(np.min(v))
                    elif k.endswith("(max)"):
                        metrics[k] = float(np.max(v))
                    else:
                        metrics[k] = float(np.mean(v))
                wandb.log(metrics, step=timestep)
            original_write(timestep=timestep, timesteps=timesteps)

        agent.write_tracking_data = write_tracking_data

    return agent


# ---------------------------------------------------------------------------
# Isaac Lab backend
# ---------------------------------------------------------------------------


def finish_wandb():
    """Finalize any active WandB run explicitly — Isaac's
    simulation_app.close() bypasses atexit, leaving runs in 'running' forever
    if we rely on process exit."""
    import wandb

    if wandb.run is not None:
        wandb.finish()


def train_isaaclab(args, log_dir: Path):
    # AppLauncher first — it configures the Kit/PhysX extensions.
    # enable_cameras must be True: the task scenes spawn camera sensors and
    # Kit crashes at sim start when they exist without camera support.
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(headless=args.headless, enable_cameras=True)
    simulation_app = launcher.app

    try:
        import gymnasium as gym

        import envs.isaac  # noqa: F401  (registers SO101-* tasks)
        from isaaclab.envs import ManagerBasedRLEnv

        # Instantiate the env cfg straight from the gym registry kwargs.
        spec = gym.spec(args.task)
        cfg_path = spec.kwargs["env_cfg_entry_point"]  # "pkg.mod:CfgName"
        module, cls_name = cfg_path.split(":")
        env_cfg = getattr(importlib.import_module(module), cls_name)()
        if args.num_envs is not None:
            env_cfg.scene.num_envs = args.num_envs
        # State-based RL: drop camera sensors (RTX headless rendering is slow
        # and crash-prone; cameras remain available in the env cfg for
        # image-obs work and via scripts/render_cameras.py for screenshots).
        for attr in [a for a in dir(env_cfg.scene) if a.endswith("_cam")]:
            setattr(env_cfg.scene, attr, None)
        env = ManagerBasedRLEnv(cfg=env_cfg)

        device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        if args.algo == "skrl":
            from isaaclab_rl.skrl import SkrlVecEnvWrapper
            from skrl.trainers.torch import SequentialTrainer

            wrapped = SkrlVecEnvWrapper(env, ml_framework="torch")
            agent = build_skrl_ppo(wrapped, args, log_dir, device)
            trainer = SequentialTrainer(
                cfg={"timesteps": args.max_iterations or 1500, "headless": True},
                env=wrapped,
                agents=[agent],
            )
            trainer.train()
            agent.save(str(log_dir / "agent.pt"))
            finish_wandb()
        else:
            from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
            from rsl_rl.runners import OnPolicyRunner

            from rl.agents.rsl_rl_cfg import SO101PPORunnerCfg

            cfg = SO101PPORunnerCfg()
            if args.max_iterations is not None:
                cfg.max_iterations = args.max_iterations
            runner_cfg = cfg.to_dict()
            if args.wandb:
                # rsl-rl-lib 5.4 dict form (the plain "wandb" string is
                # deprecated); entity flows through $WANDB_USERNAME.
                if args.wandb_entity:
                    os.environ.setdefault("WANDB_USERNAME", args.wandb_entity)
                runner_cfg["logger"] = {
                    "class_name": "WandbLogWriter",
                    "project_name": args.wandb_project,
                }
            else:
                runner_cfg["logger"] = "tensorboard"

            wrapped = RslRlVecEnvWrapper(env)
            obs_dim = wrapped.observation_space.shape[0]
            act_dim = wrapped.action_space.shape[0]
            runner_cfg["actor"]["input_dim"] = obs_dim
            runner_cfg["actor"]["output_dim"] = act_dim
            runner_cfg["critic"]["input_dim"] = obs_dim
            runner_cfg["critic"]["output_dim"] = 1

            runner = OnPolicyRunner(wrapped, runner_cfg, log_dir=str(log_dir), device=device)
            runner.learn(num_learning_iterations=runner_cfg["max_iterations"])
            runner.save(str(log_dir / "model_final.pt"))
            finish_wandb()

        env.close()
    finally:
        simulation_app.close()


# ---------------------------------------------------------------------------
# mjlab backend (MuJoCo Warp) — existing pick_lift task
# ---------------------------------------------------------------------------


def train_mjlab(args, log_dir: Path):
    module_path, func_name = MJLAB_TASKS[args.task].split(":")
    make_cfg = getattr(importlib.import_module(module_path), func_name)

    from mjlab.envs import ManagerBasedRlEnv
    from skrl.trainers.torch import SequentialTrainer

    from rl.skrl_wrapper import SkrlVecEnvWrapper

    cfg = make_cfg()
    if args.num_envs is not None:
        cfg.scene.num_envs = args.num_envs
    device = args.device or "cuda:0"
    env = ManagerBasedRlEnv(cfg, device=device)
    wrapped = SkrlVecEnvWrapper(env)

    # Same direct PPO construction as the isaaclab/mujoco paths — the fork's
    # Runner no longer accepts the old YAML "tracking" block, so WandB logging
    # goes through ExperimentCfg inside build_skrl_ppo.
    agent = build_skrl_ppo(wrapped, args, log_dir, device)
    trainer = SequentialTrainer(
        cfg={"timesteps": args.max_iterations or 1500, "headless": True},
        env=wrapped,
        agents=[agent],
    )
    trainer.train()
    agent.save(str(log_dir / "agent.pt"))
    finish_wandb()


# ---------------------------------------------------------------------------
# plain MuJoCo gymnasium backend
# ---------------------------------------------------------------------------


def train_mujoco(args, log_dir: Path):
    from skrl.envs.wrappers.torch import wrap_env
    from skrl.trainers.torch import SequentialTrainer

    module_path, func_name = MUJOCO_TASKS[args.task].split(":")
    make_env = getattr(importlib.import_module(module_path), func_name)

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    vec_env = make_env(n_envs=1)
    single_env = vec_env.envs[0]  # skrl manages batching itself; train on one env
    wrapped = wrap_env(DropImagesObs(single_env), wrapper="gymnasium")
    wrapped._device = torch.device(device)  # wrapper defaults to cuda when available

    agent = build_skrl_ppo(wrapped, args, log_dir, device)
    agent.cfg.mini_batches = 1  # single env: small batches
    trainer = SequentialTrainer(
        cfg={"timesteps": args.max_iterations or 200, "headless": True},
        env=wrapped,
        agents=[agent],
    )
    trainer.train()
    agent.save(str(log_dir / "agent.pt"))
    finish_wandb()
    vec_env.close()


def main(argv=None):
    args = parse_args(argv)
    log_dir = make_log_dir(args)
    # Pin skrl's global device BEFORE any wrapper/agent construction
    # (parse_device(None) otherwise defaults to cuda when available).
    from skrl import config as skrl_config

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    skrl_config.torch.device = device
    print(f"[train] backend={args.backend} task={args.task} algo={args.algo} device={device} log_dir={log_dir}")

    if args.backend == "isaaclab":
        train_isaaclab(args, log_dir)
    elif args.backend == "mjlab":
        train_mjlab(args, log_dir)
    else:
        train_mujoco(args, log_dir)


if __name__ == "__main__":
    main()
