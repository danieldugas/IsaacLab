import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.envs.mdp.actions import JointPositionActionCfg
from omni.isaac.lab.utils import configclass

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv
    from omni.isaac.lab.managers.action_manager import ActionTerm
    from collections.abc import Callable
    from . import delayed_joint_actions

def constant_delay(env: ManagerBasedEnv) -> torch.Tensor:
    return torch.ones(env.num_envs)

def random_delay(env: ManagerBasedEnv) -> torch.Tensor:
    return torch.randint(0, 2, (env.num_envs,))

@configclass
class DelayedJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for the delayed joint position action term.

    delay is measured in integers, with a delay of 1 meaning that action is delayed by one simulation step.
    delay applied to actions at each step is delay = constant delay + variable delay

    See :class:`DelayedJointPositionAction` for more details.
    """

    class_type: type[ActionTerm] = delayed_joint_actions.DelayedJointPositionAction

    max_delay: int = 1
    """The maximum number of steps the action may be delayed. Defaults to 1. """
    const_delay_term: Callable[..., torch.Tensor] = random_delay
    """The function to determine the constant delay for the action. Defaults to a constant delay of 1 step for all envs."""
    variable_delay_term: None | Callable[..., torch.Tensor] = None
    """The function to determine the variable delay for the action. Defaults to 0 for all envs."""