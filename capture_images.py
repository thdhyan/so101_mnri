import sys
sys.path.insert(0, '/home/thakk100/Projects/so101_mnri')
import mujoco
from pathlib import Path
from PIL import Image

out = Path('/home/thakk100/Projects/so101_mnri/images')
out.mkdir(exist_ok=True)

from so101_single_arm_env.env import SO101SingleArmEnv, SingleArmEnvConfig
env = SO101SingleArmEnv(SingleArmEnvConfig())
env.reset()
import mujoco as mj
mj.mj_forward(env.model, env.data)
r = mj.Renderer(env.model, 480, 640)
for cam in ["wrist", "outside_left", "outside_right"]:
    r.update_scene(env.data, camera=cam)
    img = r.render().copy()
    Image.fromarray(img).save(out / f"single_{cam}.png")
    print(f"saved single_{cam}.png")
env.close()

from so101_dual_arm_env.env import SO101DualArmEnv, DualArmEnvConfig
env2 = SO101DualArmEnv(DualArmEnvConfig())
env2.reset()
mj.mj_forward(env2.model, env2.data)
r2 = mj.Renderer(env2.model, 480, 640)
for cam in ["left_wrist", "right_wrist", "overhead_left", "overhead_right"]:
    r2.update_scene(env2.data, camera=cam)
    img = r2.render().copy()
    Image.fromarray(img).save(out / f"dual_{cam}.png")
    print(f"saved dual_{cam}.png")
env2.close()
print("done")
