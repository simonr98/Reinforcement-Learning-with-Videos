from gym.envs.registration import register

register(
    id='Pusher2dEnv-v1',
    entry_point='experiment_envs.envs.pusher2d:Pusher2dEnv',
)
register(
    id='ForkReacherEnv-v1',
    entry_point='experiment_envs.envs.pusher2d:ForkReacherEnv',
)
register(
    id='ImagePusher2dEnv-v1',
    entry_point='experiment_envs.envs.visual_pusher2d:ImagePusher2dEnv',
)
register(
    id='ImageForkReacher2dEnv-v1',
    entry_point='experiment_envs.envs.visual_pusher2d:ImageForkReacher2dEnv',
)