import gym
import gym_kheperaposition
import numpy as np
import math

Vmax=0.08
Wmax=math.pi/4

Kr_Prop=0.02   
K1=0.15
Kp=0.75
Ki=0.0000001 
PI=3.1415 

def random_agent(steps=300):
	episode_reward = 0
	ErrorAcumulado=0
	env = gym.make('KheperaPositionControl-v0')
	d, alpha, Oc = env.reset()
	#env.render()
	for e in range(steps):
		error = math.atan2(math.sin(Oc), math.cos(Oc))
		p = ((math.pi-math.fabs(error))/math.pi)
		v = min(K1*d*p, Vmax)
		if p > 0.9 and d > Kr_Prop:
			v = Vmax
		if d < Kr_Prop:
			v = 0
			w = 0
		ErrorAcumulado = error + ErrorAcumulado
		w = Kp*math.sin(Oc)+Ki*ErrorAcumulado
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