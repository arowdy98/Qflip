from gym.envs.registration import register

register(
    id='Flipit-v0',
    entry_point='gym_flipit.envs:FlipitEnv',  
)
register(
	id='FlipNet-v0',
    entry_point='gym_flipit.envs:FlipNetEnv',    
)
