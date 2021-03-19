from gym.envs.registration import register


register(
	id='KheperaPositionObstacle-v0',
	entry_point='gym_kheperaposition.envs:KheperaPositionObstacle'
	)

register(
	id='KheperaPositionObstacleD-v0',
	entry_point='gym_kheperaposition.envs:KheperaPositionObstacleD'
	)

register(
	id='KheperaPositionControl-v0',
	entry_point='gym_kheperaposition.envs:KheperaPositionControl'
	)