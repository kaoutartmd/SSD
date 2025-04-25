from social_dilemmas.envs.cleanup import CleanupEnv

def get_env_creator(
    use_collective_reward=False,
    inequity_averse_reward=False,
    alpha=0.0,
    beta=0.0,
):
    def env_creator(_):
        return CleanupEnv(
            num_agents=2,
            return_agent_actions=True,
            use_collective_reward=use_collective_reward,
            inequity_averse_reward=inequity_averse_reward,
            alpha=alpha,
            beta=beta,
        )
    
    return env_creator
