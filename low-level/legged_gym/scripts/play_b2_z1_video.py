"""Record B2-Z1 low-level policy rollouts with Isaac Gym EGL rendering.

This entry point follows ``play.py`` and supports every task registered for
B2-Z1.  It keeps Isaac Gym graphics enabled for camera sensors without
creating an X11 viewer, which allows recording on headless GPU servers.
"""

from pathlib import Path
import os
import sys


# Prefer the packages from this checkout, before importing Isaac Gym.
SCRIPT_DIR = Path(__file__).resolve().parent
LOW_LEVEL_DIR = SCRIPT_DIR.parents[1]
REPO_DIR = LOW_LEVEL_DIR.parent
for import_path in (
    REPO_DIR / "third_party" / "isaacgym" / "python",
    REPO_DIR / "third_party" / "rsl_rl",
    LOW_LEVEL_DIR,
    SCRIPT_DIR,
):
    import_path = str(import_path)
    if import_path not in sys.path:
        sys.path.insert(0, import_path)
os.chdir(SCRIPT_DIR)


# Isaac Gym's default Vulkan ICD may require Xorg.  Use NVIDIA EGL only for
# this process and leave the system Vulkan configuration untouched.
egl_icd = Path(os.environ.get("B2Z1_EGL_ICD", "/tmp/nvidia_icd_egl.json"))
if not egl_icd.exists():
    egl_icd.write_text(
        '{\n'
        '  "file_format_version": "1.0.1",\n'
        '  "ICD": {\n'
        '    "library_path": "libEGL_nvidia.so.0",\n'
        '    "api_version": "1.4.312"\n'
        '  }\n'
        '}\n'
    )
os.environ.setdefault("VK_ICD_FILENAMES", str(egl_icd))
os.environ.setdefault("XDG_RUNTIME_DIR", "/tmp")
Path(os.environ["XDG_RUNTIME_DIR"]).mkdir(parents=True, exist_ok=True)

import time
import numpy as np
import isaacgym
import torch
from isaacgym import gymapi

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.envs.manip_loco.manip_loco import ManipLoco
from legged_gym.utils import Logger, get_args, task_registry


np.set_printoptions(precision=3, suppress=True)


def render_record(env, mode="rgb_array"):
    """Render a B2-Z1 camera after moving it to the current robot pose."""
    if env.global_steps % 2 != 0:
        return None

    # GPU PhysX state must be complete before graphics consumes it.
    env.gym.fetch_results(env.sim, True)
    cameras = []
    for env_index, camera in enumerate(env._rendering_camera_handles):
        root = env.root_states[env_index, :3].detach().cpu().numpy()
        camera_position = root + np.array([0.0, 2.0, 1.0])
        camera_target = root + np.array([0.0, 0.0, 0.35])
        cameras.append((camera_position, camera_target))
        env.gym.set_camera_location(
            camera,
            env.envs[env_index],
            gymapi.Vec3(*camera_position),
            gymapi.Vec3(*camera_target),
        )

    env.gym.step_graphics(env.sim)
    env.gym.render_all_camera_sensors(env.sim)
    images = []
    for env_index, camera in enumerate(env._rendering_camera_handles):
        image = env.gym.get_camera_image(
            env.sim, env.envs[env_index], camera, gymapi.IMAGE_COLOR
        )
        height, packed_width = image.shape
        images.append(image.reshape(height, packed_width // 4, 4))
    return images


# The environment implementation is shared by B1-Z1 and B2-Z1.  Bind the
# corrected camera order only for this B2-Z1 recording process.
ManipLoco.render_record = render_record


def play(args):
    log_pth = os.path.join(
        LEGGED_GYM_ROOT_DIR, "logs", args.proj_name, args.exptid
    )
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # Keep the same playback overrides as play.py.  Camera sensors are
    # created by ManipLoco from this flag, so it must be set before make_env.
    env_cfg.env.num_envs = 1
    env_cfg.env.record_video = bool(args.record_video)
    env_cfg.terrain.num_rows = 6
    env_cfg.terrain.num_cols = 3
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = True
    env_cfg.domain_rand.randomize_base_com = False
    if args.flat_terrain:
        env_cfg.terrain.height = [0.0, 0.0]

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    train_cfg.runner.resume = True
    ppo_runner, _, checkpoint, log_pth = task_registry.make_alg_runner(
        log_root=log_pth,
        env=env,
        name=args.task,
        args=args,
        train_cfg=train_cfg,
        return_log_dir=True,
    )
    policy = ppo_runner.get_inference_policy(
        device=env.device, stochastic=args.stochastic
    )

    mp4_writers = []
    if args.record_video:
        import imageio

        env.enable_viewer_sync = False
        run_name = os.path.basename(os.path.normpath(log_pth))
        video_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "videos", run_name)
        os.makedirs(video_dir, exist_ok=True)
        for env_index in range(env.num_envs):
            video_name = f"{args.exptid}-{env_index}-{checkpoint}.mp4"
            mp4_writers.append(
                imageio.get_writer(
                    os.path.join(video_dir, video_name), fps=25, format="FFMPEG"
                )
            )
        print("Recording videos to:", video_dir)

    traj_length = int(env.max_episode_length) if args.record_video else 1000 * int(env.max_episode_length)
    env.reset()
    for step in range(traj_length):
        start_time = time.time()
        if args.use_jit:
            policy_obs = torch.cat(
                (
                    obs[:, :env.cfg.env.num_proprio],
                    obs[:, env.cfg.env.num_proprio + env.cfg.env.num_priv :],
                ),
                dim=1,
            )
            actions = policy(policy_obs)
        else:
            actions = policy(obs.detach(), hist_encoding=True)
        obs, _, _, _, _, _ = env.step(actions.detach())

        if args.record_video:
            images = env.render_record(mode="rgb_array")
            if images is not None:
                for env_index, image in enumerate(images):
                    mp4_writers[env_index].append_data(image)

        elapsed = time.time() - start_time
        time.sleep(max(0.02 - elapsed, 0.0))
        if step % 50 == 0:
            print(
                "step", step,
                "cmd", env.commands[0, :3].detach().cpu().numpy(),
                "lin", env.base_lin_vel[0, :3].detach().cpu().numpy(),
                "act_leg_abs", actions[0, :12].abs().mean().item(),
            )

    for writer in mp4_writers:
        writer.close()
    if args.record_video:
        print("Finished recording", traj_length, "simulation steps")


if __name__ == "__main__":
    args = get_args()
    # ``BaseTask`` compares this value with True/False.  The sentinel keeps
    # the graphics device for camera sensors while preventing viewer creation.
    args.headless = object()
    play(args)
