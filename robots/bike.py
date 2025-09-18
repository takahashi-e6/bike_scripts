import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

BIKE_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/workspace/lab_share/bike.usd",
    ),
    actuators={"all_joints": ImplicitActuatorCfg(
        joint_names_expr=["back_wheel_joint", "steering_joint"],
        damping={"back_wheel_joint": 1.0,
                "steering_joint": 1.0},
        stiffness={"back_wheel_joint": 0.0,
                "steering_joint":  10.0}
    )},
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "back_wheel_joint": 0.0,
            "steering_joint": 0.0,
        },
    ),
)
