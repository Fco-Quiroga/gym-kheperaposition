from gym.envs.registration import register

register(
	id='KheperaPositionPixel-v0',
	entry_point='gym_kheperaposition.envs:KheperaPositionPixel'
	)

register(
	id='KheperaPositionControl-v0',
	entry_point='gym_kheperaposition.envs:KheperaPositionControl'
	)

register(
	id='KheperaPositionDiscrete-v0',
	entry_point='gym_kheperaposition.envs:KheperaPositionDiscrete'
	)