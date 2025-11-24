import gymnasium as gym

# from custom_lab.env.g1_env import G1ModularEnv
# from custom_lab.Agent.ppo.rsl_rl_ppo_cfg import PPORunnerCfg_CM
# from custom_lab.Agent.task.g1_task_cfg import RobotEnvCfg_CM, RobotPlayEnvCfg_CM

gym.register(
    id="Unitree-g1-29dof-modular-velocity",
    entry_point="custom_lab.env.g1_env:G1ModularEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "custom_lab.Agent.task.g1_task_cfg:RobotEnvCfg_CM",
        "play_env_cfg_entry_point": "custom_lab.Agent.task.g1_task_cfg:RobotPlayEnvCfg_CM",
        "rsl_rl_cfg_entry_point": "custom_lab.Agent.ppo.rsl_rl_ppo_cfg:PPORunnerCfg_CM",
    },
)
