from gym.envs.registration import register

register(
	id='KheperaPosition-v0',
	entry_point='gym_kheperaposition.envs:KheperaPositionEnv'
	)