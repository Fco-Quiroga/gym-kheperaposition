import gym
import gym_kheperaposition
import numpy as np

def random_agent(episodes=300):
	episode_reward = 0
	env = gym.make('KheperaPosition-v0')
	d, alpha, Oc = env.reset()
	env.render()
	for e in range(episodes):
		action = env.Wmax*np.sin(Oc)
		state, reward, done, info = env.step(action)
		d, alpha, Oc = state
		env.render()
		print(reward)
		episode_reward += reward
		if done:
			print('Reward of the episode is: ',episode_reward)
			break

random_agent()