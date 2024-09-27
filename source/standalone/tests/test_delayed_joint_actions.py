from omni.isaac.lab.app import AppLauncher # otherwise omni.kit is missing
app_launcher = AppLauncher()
simulation_app = app_launcher.app

from omni.isaac.lab.envs.mdp.actions.delayed_joint_actions import test_delayed_joint_position_action

if __name__ == "__main__":
    test_delayed_joint_position_action()