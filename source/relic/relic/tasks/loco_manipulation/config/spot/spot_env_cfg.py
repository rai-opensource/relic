# Copyright (c) 2024 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from isaaclab.utils import configclass
from relic.tasks.loco_manipulation.interlimb_env_cfg import InterlimbEnvCfg

##
# Pre-defined configs
##
from relic.assets.spot.spot import SPOT_ARM_CFG  # isort: skip


@configclass
class SpotInterlimbEnvCfg_Phase_1(InterlimbEnvCfg):
    """Configuration for the phase 1 environment."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # switch robot to spot-arm
        self.scene.robot = SPOT_ARM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.joint_drive.gains.stiffness = None

        # Weights
        self.rewards.track_lin_vel_xy_exp.weight = 7.0
        self.rewards.track_ang_vel_z_exp.weight = 3.5
        self.rewards.track_base_orientation_l2.weight = -7.0
        self.rewards.track_base_height_l2.weight = -3.5
        self.rewards.dof_torques_l2.weight = -1.0e-05
        self.rewards.dof_acc_l2.weight = -2.5e-7
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.feet_air_time.weight = 0.01
        self.rewards.joint_energy.weight = 2.0
        self.rewards.air_time_variance.weight = -10.0
        self.rewards.flight_penalty.weight = -5.0
        self.rewards.foot_impact.weight = -1.0
        self.rewards.foot_slip.weight = -0.5
        self.rewards.foot_clearance.weight = 0.0


@configclass
class SpotInterlimbEnvCfg_Phase_2(InterlimbEnvCfg):
    """Configuration for the phase 2 environment."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # switch robot to spot-arm
        self.scene.robot = SPOT_ARM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.joint_drive.gains.stiffness = None

        # relax contact events
        self.terminations.undesired_ground_contact.params["sensor_cfg"].body_names = [
            ".*uleg"
        ]

        # Weights
        self.rewards.track_lin_vel_xy_exp.weight = 7.0
        self.rewards.track_ang_vel_z_exp.weight = 3.5
        self.rewards.track_base_orientation_l2.weight = -7.0
        self.rewards.track_base_height_l2.weight = -30.0
        self.rewards.dof_torques_l2.weight = -0.0001
        self.rewards.dof_acc_l2.weight = -1.0e-06
        self.rewards.action_rate_l2.weight = -0.1
        self.rewards.feet_air_time.weight = 0.01
        self.rewards.joint_energy.weight = 2.0
        self.rewards.air_time_variance.weight = -10.0
        self.rewards.flight_penalty.weight = -10.0
        self.rewards.foot_impact.weight = -1.0
        self.rewards.foot_slip.weight = -0.5
        self.rewards.foot_clearance.weight = 1.0


@configclass
class SpotInterlimbEnvCfg_Phase_3(InterlimbEnvCfg):
    """Configuration for the phase 3 environment."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # switch robot to spot-arm
        self.scene.robot = SPOT_ARM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.joint_drive.gains.stiffness = None

        # change event and termination terms
        self.terminations.undesired_ground_contact.params["sensor_cfg"].body_names = [
            ".*uleg"
        ]

        # Weights
        self.rewards.track_lin_vel_xy_exp.weight = 7.0
        self.rewards.track_ang_vel_z_exp.weight = 3.5
        self.rewards.track_base_orientation_l2.weight = -45.0
        self.rewards.track_base_height_l2.weight = -120.0
        self.rewards.dof_torques_l2.weight = -0.0001
        self.rewards.dof_acc_l2.weight = -1.0e-06
        self.rewards.action_rate_l2.weight = -0.1
        self.rewards.feet_air_time.weight = 0.01
        self.rewards.joint_energy.weight = 2.0
        self.rewards.air_time_variance.weight = -10.0
        self.rewards.flight_penalty.weight = -10.0
        self.rewards.foot_impact.weight = -1.0
        self.rewards.foot_slip.weight = -1.0
        self.rewards.foot_clearance.weight = 1.0


@configclass
class SpotInterlimbEnvCfg_Phase_4(InterlimbEnvCfg):
    """Configuration for the phase 4 environment."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # switch robot to spot-arm
        self.scene.robot = SPOT_ARM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.joint_drive.gains.stiffness = None

        # change event and termination terms
        self.terminations.undesired_ground_contact.params["sensor_cfg"].body_names = [
            ".*uleg"
        ]

        # Weights
        self.rewards.track_lin_vel_xy_exp.weight = 7.0
        self.rewards.track_ang_vel_z_exp.weight = 3.5
        self.rewards.track_base_orientation_l2.weight = -45.0
        self.rewards.track_base_height_l2.weight = -120.0
        self.rewards.dof_torques_l2.weight = -0.0001
        self.rewards.dof_acc_l2.weight = -1.0e-06
        self.rewards.action_rate_l2.weight = -0.1
        self.rewards.feet_air_time.weight = 0.01
        self.rewards.joint_energy.weight = 2.0
        self.rewards.air_time_variance.weight = 0.0
        self.rewards.flight_penalty.weight = -10.0
        self.rewards.foot_impact.weight = -1.0
        self.rewards.foot_slip.weight = -2.0
        self.rewards.foot_clearance.weight = 1.0
        self.rewards.dof_torque_limits_l2.weight = -1.0
        self.rewards.gait.weight = 5.0
        self.rewards.three_leg_gait.weight = 5.0
        self.rewards.feet_air_time_target.weight = 0.5


@configclass
class SpotInterlimbEnvCfg_PLAY(SpotInterlimbEnvCfg_Phase_1):
    """Configuration for the play environment."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 128
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        self.curriculum.terrain_levels = None

        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.3, 1.0)
        self.commands.base_velocity.debug_vis = False

        self.terminations.undesired_ground_contact.params["sensor_cfg"].body_names = [
            ".*uleg"
        ]
        self.terminations.base_contact = None
