# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
print(args_cli)
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch

from rsl_rl.runners import OnPolicyRunner

from omni.isaac.lab.utils.dict import print_dict

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    LOG_ACTOBS = False
    ZEROACT = False
    if LOG_ACTOBS: # Daniel: simplified scene for logging
        env_cfg.scene.num_envs = 1
        env_cfg.curriculum = {}
        env_cfg.scene.terrain.terrain_type = "plane"
        env_cfg.scene.terrain.terrain_generator = None
        env_cfg.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        env_cfg.commands.base_velocity.ranges.heading = (0.0, 0.0)
        env_cfg.observations.policy.enable_corruption = False
        env_cfg.events.base_external_force_torque = None
        env_cfg.events.push_robot = None

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    if args_cli.video: # center camera on robot and zoom
        env_cfg.viewer.origin_type = "asset_root"
        env_cfg.viewer.asset_name = "robot"
        env_cfg.viewer.eye = (3., 3., 3.)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(
        ppo_runner.alg.actor_critic, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
    )
    export_policy_as_onnx(
        ppo_runner.alg.actor_critic, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )


    actobs_log = {}
    def NP(tensor):
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        else:
            return tensor
    def log_actobs(actobs_log, actions, obs, rew, dones, log=True, N=100):
        import numpy as np
        import json
        if 'actobs_npy' not in actobs_log:
            actobs_log['actobs_npy'] = []
            actobs_log['fields'] = [
                ['actions', tuple(actions.shape)],
                ['obs', tuple(obs.shape)],
                ['rew', tuple(rew.shape)],
                ['dones', tuple(dones.shape)],
            ]
        if len(actobs_log['actobs_npy']) < N:
            actions = NP(actions)
            obs = NP(obs)
            rew = NP(rew)
            dones = NP(dones)
            actobs_log['actobs_npy'].append(np.concatenate([actions.flatten(), obs.flatten(), rew.flatten(), dones.flatten()]).tolist())
            if len(actobs_log['actobs_npy']) == N:
                path = "/tmp/actobs_log.json"
                with open(path, 'w') as f:
                    json.dump(actobs_log, f)
                    print(path, "written")


    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            if ZEROACT:
                actions = actions * 0.0
            # env stepping
            obs, rew, dones, _ = env.step(actions)
            log_actobs(actobs_log, actions[0], obs[0], rew[0], dones[0], log=LOG_ACTOBS) # log 1 robot
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    # close the simulator
    env.close()
    if args_cli.video:
        print("Video recorded to:", env.env.video_folder)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

    daniel_args = """
--headless --task Isaac-Velocity-Rough-H1-v0 --device cpu --disable_fabric

--headless --task Isaac-Velocity-Rough-H1-v0 --load_run 2024-08-26_16-31-15 

--headless --num_envs 1 --task Isaac-Velocity-Flat-H1-v0 --load_run 2024-09-10_14-07-02 --video --video_length 500

    """