from omni.isaac.lab.utils import configclass
from omni.isaac.lab.envs.mdp.actions.delayed_joint_actions_cfg import DelayedJointPositionActionCfg, random_1step_delay

from .flat_env_cfg import H1FlatEnvCfg
from .rough_env_cfg import H1RoughEnvCfg

# no imu ang/lin vel, no previous joint pos, no linear target
@configclass
class H1StandFlatEnvCfg(H1RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.base_ang_vel = None
        self.observations.policy.base_lin_vel = None
        self.observations.policy.actions = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None
        self.rewards.feet_air_time.weight = 1.0
        self.rewards.feet_air_time.params["threshold"] = 0.6
        # target
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

@configclass
class H1FlatNoLinVelCfg(H1FlatEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.observations.policy.base_lin_vel = None    
        self.actions.joint_pos = DelayedJointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)
        self.actions.joint_pos.variable_delay_term = random_1step_delay
        self.actions.joint_pos.max_delay = 2

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    joint_pos = DelayedJointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)

# delay
@configclass
class H1FlatDelayEnvCfg(H1FlatEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.actions.joint_pos = DelayedJointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)

    
@configclass
class H1FlatFreezeArmsEnvCfg(H1FlatEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.actions.joint_pos = DelayedJointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True, FREEZE_ARMS=True)
    
@configclass
class H1FlatNoLinVelFreezeArmsEnvCfg(H1FlatEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.observations.policy.base_lin_vel = None    
        self.actions.joint_pos = DelayedJointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True, FREEZE_ARMS=True)
        self.actions.joint_pos.variable_delay_term = random_1step_delay
        self.actions.joint_pos.max_delay = 2