from .adapters.gym_adapter import GymAdapter

ADAPTERS = {
    'gym': GymAdapter,
}


UNIVERSES = set(ADAPTERS.keys())

def get_environment(universe, domain, task, environment_params):
    return ADAPTERS[universe](domain, task, **environment_params)

def get_environment_from_params(environment_params):
    universe = environment_params['universe']
    task = environment_params['task']
    domain = environment_params['domain']
    environment_kwargs = environment_params.get('kwargs', {}).copy()

    return get_environment(universe, domain, task, environment_params)

def get_goal_example_environment_from_variant(variant):
    import gym
    
    if variant['task'] not in [env.id for env  in gym.envs.registry.all()]:
        if 'Manip' in variant['task']:
            #import manip_envs
            pass
        else:
            from multiworld.envs.mujoco import register_goal_example_envs
            register_goal_example_envs()
            from metaworld.envs.mujoco import register_rl_with_videos_custom_envs
            register_rl_with_videos_custom_envs()
#            import mj_envs.hand_manipulation_suite

    #        from metaworld.envs.mujoco.sawyer_xyz import register_environments; register_environments()
    return GymAdapter(env=gym.make(variant['task']))
