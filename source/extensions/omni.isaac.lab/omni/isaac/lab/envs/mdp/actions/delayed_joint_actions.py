from __future__ import annotations
import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.utils.torch_utils import pick_from_dim
from omni.isaac.lab.envs.mdp.actions import JointPositionAction

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv

    from . import delayed_joint_actions_cfg

class DelayedJointPositionAction(JointPositionAction):
    """Joint action term that applies the processed actions to the articulation's joints as
    position commands with a configuration-specified delay."""

    cfg: delayed_joint_actions_cfg.DelayedJointPositionActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: delayed_joint_actions_cfg.DelayedJointPositionActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)
        
        # create buffer
        self._processed_actions_buffer = torch.zeros(self.num_envs, cfg.max_delay+1, self.action_dim, device=self.device)
        self._constant_delay = cfg.const_delay_term(env)
        self._variable_delay_func = cfg.variable_delay_term

    def process_actions(self, actions: torch.Tensor):
        # Delay is applied in process_actions because apply_actions is called at every decimation step
        # (see manager_based_env step() method.)
        # this action will be applied to the environment N steps in the future (N = Delay)
        super().process_actions(actions)
        # TODO roll delay buffer: if inefficient, could use something like
        # "from omni.isaac.lab.utils.buffer import CircularBuffer"
        self._processed_actions_buffer = torch.roll(self._processed_actions_buffer, 1, 1) # shift by one
        self._processed_actions_buffer[:, 0] = self.processed_actions
        # sample delay, apply to action.  0 delay means we pass the latest, 1 the prev action, etc
        delay = self._constant_delay # (N_envs,)
        if self._variable_delay_func is not None:
            delay = delay + self._variable_delay_func(self._env)
        # clamp delay to 0 - cfg.max_delay
        delay = torch.clamp(delay, 0, self.cfg.max_delay).to(torch.int64)
        # for each env, pick actions according to delay
        # TODO: should we store delayed actions directly in processed_actions?
        # depending on what other components expect processed_actions to be
        self._delayed_processed_actions = pick_from_dim(self._processed_actions_buffer, 0, 1, delay) # (n_envs, n_act)

    def apply_actions(self):
        # set position targets
        self._asset.set_joint_position_target(self._delayed_processed_actions, joint_ids=self._joint_ids)


def test_delayed_joint_position_action(human=False):
    from omni.isaac.lab.managers.action_manager import ActionManager
    from .delayed_joint_actions_cfg import DelayedJointPositionActionCfg, constant_1step_delay
    from omni.isaac.lab.utils import configclass

    @configclass
    class ActionsCfg:
        """Action specifications for the MDP."""
        joint_pos = DelayedJointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)
    actions_config = ActionsCfg()
    actions_config.joint_pos.const_delay_term = constant_1step_delay # for testing, constant delay of 1
    actions_config.joint_pos.scale = 1 # less confusing than default 0.5 scale
    # spoofing classes, with just enough to test this 
    N_ENV = 2
    N_JNT = 4
    class SpoofAsset:
        def __init__(self):
            self.num_joints = N_JNT
            self.data = type("SpoofData", (), {"default_joint_pos": torch.zeros(N_ENV, N_JNT)})
            self.applied_actions = []
        def set_joint_position_target(self, processed_actions, **kwargs):
            self.applied_actions.append(processed_actions)
        def find_joints(self, *args, **kwargs):
            return list(range(N_JNT)), ["spoofjoint" + str(i) for i in range(N_JNT)]
    class SpoofEnv:
        def __init__(self):
            self.num_envs = N_ENV
            self.device = 'cpu'
            self.scene = {'robot': SpoofAsset()}
    env = SpoofEnv()
    # create a sequence and check that delay is 1
    N_SEQ = 20
    action_sequence = [torch.ones((N_ENV, N_JNT)) * i for i in range(N_SEQ)]
    DECIMATION = 4
    device = 'cpu'
    action_manager = ActionManager(actions_config, env) # type: ignore
    for i, action in enumerate(action_sequence):
        action_manager.process_action(action.to(device))
        # perform physics stepping
        for _ in range(DECIMATION):
            action_manager.apply_action()
    applied_actions = env.scene['robot'].applied_actions
    assert len(applied_actions) == (N_SEQ * DECIMATION)
    for i in range(N_SEQ-1):
        env_idx = 0
        if human:
            print("{}, {}".format(applied_actions[::4][i+1][env_idx], action_sequence[i][env_idx]))
        assert torch.all(applied_actions[::4][i+1] == action_sequence[i]), "actions not delayed correctly"
    print("test_delayed_joint_position_action passed")
