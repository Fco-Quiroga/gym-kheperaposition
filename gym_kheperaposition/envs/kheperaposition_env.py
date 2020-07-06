#  24 de junio de 2020

import gym
from gym import error, spaces, utils
from gym.utils import seeding
try:
    import sim
except:
    print ('--------------------------------------------------------------')
    print ('"sim.py" could not be imported. This means very probably that')
    print ('either "sim.py" or the remoteApi library could not be found.')
    print ('Make sure both are in the same folder as this file,')
    print ('or appropriately adjust the file "sim.py"')
    print ('--------------------------------------------------------------')
    print ('')
import sys
import time
import numpy as np
import matplotlib.pyplot as mlp


"""
    Description:
    	The objetive of this environment is to drive the wheeled mobile robot from its
    	current position to a predefined target point.    

    Source:
        This environment corresponds to the implementation of the Khepera IV library
        for robotic control education using V-REP for the position control experiment.

    Observation:
        Type: Box(3)
        Num	Observation               Min             Max
        0	Distance to the target     0               5
        1	Target Angle              -pi              pi
        2	Error Angle               -pi              pi

    Actions:
        Type: Box (1)
		Num	Action                    Min             Max
        0	Angular Velocity         -pi/2            pi/2 

    Reward:
    	reward = -(d**2)

    Starting State:
        
    Episode Termination:
        When the robot reaches the target point or the boundaries of the platform.

"""  

class KheperaPositionEnv(gym.Env):
	metadata = {'render.modes':['human']}

	def __init__(self):

		sim.simxFinish(-1) # just in case, close all opened connections
		self.clientID = sim.simxStart('127.0.0.1',19997,True,True,5000,5)
		if self.clientID!=-1:  #check if client connection successful	
			print('Connected to remote API server')
			# enable the synchronous mode on the client:
			sim.simxSynchronous(self.clientID,True)
		else:
			print('Connection not successful')
			sys.exit('Could not connect')

		# Set Robot

		errorCode, self.right_motor = sim.simxGetObjectHandle(self.clientID, 'K4_Right_Motor',sim.simx_opmode_oneshot_wait)
		errorCode, self.left_motor = sim.simxGetObjectHandle(self.clientID, 'K4_Left_Motor',sim.simx_opmode_oneshot_wait)
		errorCode, self.khepera = sim.simxGetObjectHandle(self.clientID, 'Khepera_IV',sim.simx_opmode_oneshot_wait)
		errorCode, self.target = sim.simxGetObjectHandle(self.clientID, 'Target',sim.simx_opmode_oneshot_wait)  
		errorCode, self.camera = sim.simxGetObjectHandle(self.clientID, 'Vision_sensor',sim.simx_opmode_oneshot_wait)
		#sim.simxSetObjectParent(self.clientID, self.target, -1, True, sim.simx_opmode_oneshot)
		errorCode, resolution, image = sim.simxGetVisionSensorImage(self.clientID,self.camera, 0, sim.simx_opmode_streaming)
		pos = sim.simxGetObjectPosition(self.clientID,self.khepera,-1,sim.simx_opmode_streaming)
		ori_body = sim.simxGetObjectOrientation(self.clientID,self.khepera,-1,sim.simx_opmode_streaming)
		tar = sim.simxGetObjectPosition(self.clientID,self.target,-1,sim.simx_opmode_streaming)

		self.Vmax = 0.08     #Maximum Linear Velocity [m/s]
		self.Wmax = (np.pi)/2   #Maximum Angular Velocity [m/s]
		high = np.array([5,np.pi,np.pi], dtype=np.float32)
		low = np.array([0.,-np.pi,-np.pi], dtype=np.float32)

		# Action Space
		self.action_space = spaces.Box(
			low=-self.Wmax,
			high=self.Wmax,
			shape=(1,),
			dtype=np.float32
		)
		# Observation Space
		self.observation_space = spaces.Box(
			low=low,
			high=high,
			dtype=np.float32
		)

		self.viewer = None
		self.seed()
		self.steps = 0
		self.MaxSteps = 300

		self.xp, self.yp = self.getPositionTarget()

	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def getPositionRobot(self):
		pos = sim.simxGetObjectPosition(self.clientID,self.khepera,-1,sim.simx_opmode_buffer)
		ori_body = sim.simxGetObjectOrientation(self.clientID,self.khepera,-1,sim.simx_opmode_buffer)
		theta = ori_body[1][2]
		xc = pos[1][0]   
		yc = pos[1][1]
		return xc,yc,theta

	def getPositionTarget(self):
		tar = sim.simxGetObjectPosition(self.clientID,self.target,-1,sim.simx_opmode_buffer)
		xp = tar[1][0]
		yp = tar[1][1]
		return xp,yp

	def updateVelocities(self,v,w):
		Vr = (2*v+w*0.1)/2
		Vl = (2*v-w*0.1)/2

		sim.simxSetJointTargetVelocity(self.clientID,self.left_motor,47.5960*Vl, sim.simx_opmode_oneshot)
		sim.simxSetJointTargetVelocity(self.clientID,self.right_motor,47.5960*Vr, sim.simx_opmode_oneshot)

	def get_obs(self):
		xc,yc,theta = self.getPositionRobot()
		xp,yp = self.getPositionTarget()

		d = np.sqrt(((xp-xc)**2)+((yp-yc)**2))      
		alpha = np.arctan2(yp-yc,xp-xc)
		Oc = alpha-theta
		return d,alpha,Oc

	def step(self,action, Kp=0.2):
		d,alpha,Oc = self.get_obs()
		observation = np.array([d,alpha,Oc])
		reward = -(d**2)
		done = False

		if d>=0.05:
			done = False
			w = action
			v = (self.Vmax/Kp)*d
			if v>self.Vmax:
				v=self.Vmax
		else:
			v = 0
			w = 0
			done = True
			reward = self.MaxSteps - self.steps
			
		self.updateVelocities(v,w)
		
		if d>=2 or self.steps > (self.MaxSteps-1):
			done=True
			#reward = -10
		if not done:
			sim.simxSynchronousTrigger(self.clientID)
			self.steps+=1
		info = {'Distance':d,
				'Oc':Oc}

		return observation, reward, done, info

	def reset(self):
		# stop the simulation:
		sim.simxStopSimulation(self.clientID,sim.simx_opmode_blocking)
		time.sleep(0.1)
		sim.simxSynchronous(self.clientID,True)
		# start the simulation:
		sim.simxStartSimulation(self.clientID,sim.simx_opmode_blocking)
		d,alpha,Oc = self.get_obs()
		observation = np.array([d,alpha,Oc])
		self.steps = 0
		return observation

	def getCameraImage(self):
		errorCode, resolution, image = sim.simxGetVisionSensorImage(self.clientID, self.camera, 0, sim.simx_opmode_buffer)
		im = np.array(image,dtype=np.uint8)
		im.resize([resolution[0],resolution[1],3])
		return im

	def render(self, mode='human'):
		img = self.getCameraImage()
		if mode == 'human':
			from gym.envs.classic_control import rendering
			if self.viewer is None:
				self.viewer = rendering.SimpleImageViewer()
			self.viewer.imshow(img)
			return self.viewer.isopen

	def close(self):
		if self.viewer:
			self.viewer.close()
			self.viewer = None
		sim.simxStopSimulation(self.clientID,sim.simx_opmode_blocking)
		# Now close the connection to CoppeliaSim:
		sim.simxFinish(self.clientID)
