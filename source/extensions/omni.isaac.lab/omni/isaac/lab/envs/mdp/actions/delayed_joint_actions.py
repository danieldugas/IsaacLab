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
        self._processed_actions_buffer = torch.zeros(self.num_envs, cfg.max_delay, self.action_dim, device=self.device)
        self._constant_delay = cfg.const_delay_term(env)
        self._variable_delay_func = cfg.variable_delay_term

    def process_actions(self, actions: torch.Tensor):
        # Delay is applied in process_actions because apply_actions is called at every decimation step
        # (see manager_based_env step() method.)
        # this action will be applied to the environment N steps in the future (N = Delay)
        super().process_actions(actions)
        # TODO roll delay buffer: if inefficient, could use something like
        # "from omni.isaac.lab.utils.buffer import CircularBuffer"
        torch.roll(self._processed_actions_buffer, 1, 1) # shift by one
        self._processed_actions_buffer[:, 0] = self.processed_actions
        # sample delay, apply to action.  0 delay means we pass the latest, 1 the prev action, etc
        delay = self._constant_delay # (N_envs,)
        if self._variable_delay_func is not None:
            delay = delay + self._variable_delay_func(self._env)
        # clamp delay to 0 - cfg.max_delay
        delay = torch.clamp(delay, 0, self.cfg.max_delay)
        # for each env, pick actions according to delay
        # TODO: should we store delayed actions directly in processed_actions?
        # depending on what other components expect processed_actions to be
        self._delayed_processed_actions = pick_from_dim(self._processed_actions_buffer, 0, 1, delay) # (n_envs, n_act)

    def apply_actions(self):
        # set position targets
        self._asset.set_joint_position_target(self._delayed_processed_actions, joint_ids=self._joint_ids)


def test_delayed_joint_position_action():
    from omni.isaac.lab.managers.action_manager import ActionManager
    from .delayed_joint_actions_cfg import DelayedJointPositionActionCfg, constant_delay
    config = DelayedJointPositionActionCfg
    config.const_delay_term = constant_delay
    class SpoofEnv:
        def __init__(self):
            self.num_envs = 2
    env = SpoofEnv()
    N_ACT = 10
    action_sequence = [torch.ones(N_ACT) * i for i in range(10)]
    applied_actions = []
    DECIMATION = 4
    device = 'cpu'
    action_manager = ActionManager(config, env) # type: ignore
    for action in action_sequence:
        action_manager.process_action(action.to(device))
        # perform physics stepping
        for _ in range(DECIMATION):
            action_manager.apply_action()
