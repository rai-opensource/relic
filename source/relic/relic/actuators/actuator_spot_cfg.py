# Copyright (c) 2024 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from __future__ import annotations

from isaaclab.actuators.actuator_cfg import RemotizedPDActuatorCfg
from isaaclab.utils import configclass
from relic.actuators import SpotKneeActuator
from relic.assets.spot.constants import POS_TORQUE_SPEED_LIMIT, NEG_TORQUE_SPEED_LIMIT


@configclass
class SpotKneeActuatorCfg(RemotizedPDActuatorCfg):
    """Configuration for the Spot knee actuator."""

    class_type: type = SpotKneeActuator

    enable_torque_speed_limit: bool = False

    pos_torque_speed_limit = POS_TORQUE_SPEED_LIMIT
    neg_torque_speed_limit = NEG_TORQUE_SPEED_LIMIT
