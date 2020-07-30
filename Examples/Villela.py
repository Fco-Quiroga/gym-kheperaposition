import gym
import gym_kheperaposition
import numpy as np
import math

Vmax=0.08
Wmax=math.pi/4
L=0.1
Kr_V_RL=0.05   # radio de seguridad

def random_agent(steps=300):
	episode_reward = 0
	env = gym.make('KheperaPositionControl-v0')
	d, alpha, Oc = env.reset()
	#env.render()
	for e in range(steps):
		w = Wmax*math.sin(Oc)
		if d > Kr_V_RL:
			v = Vmax
		else:
			v = d*(Vmax/Kr_V_RL)
		if d < 0.02:
			v = 0
			w = 0
		Vl = (v-(w*0.1)/2)*48
		Vr = (v+(w*0.1)/2)*48
		action = [Vl, Vr]
		state, reward, done, info = env.step(action)
		d, alpha, Oc = state
		#env.render()
		print(reward)
		episode_reward += reward
		if done:
			print('Reward of the episode is: ',episode_reward)
			break
	env.close()
random_agent()