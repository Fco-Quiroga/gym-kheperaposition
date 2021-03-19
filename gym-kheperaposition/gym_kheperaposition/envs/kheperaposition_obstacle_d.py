#  11 de septiembre de 2020

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
import random
import math
from PIL import Image

"""
    Description:
    	The objetive of this environment is to drive the wheeled mobile robot from its
    	current position to a predefined target point.    

    Source:
        This environment corresponds to the implementation of the Khepera IV library
        for robotic control education using V-REP for the position control experiment.

    Observation:

        Type: Box(12)
        Num	Observation               Min             Max
        0	Distance to the target     0               2
        1	Error Angle               -pi              pi
        2	Vel lineal t-1             0              0.08
        3   Vel angular t-1          -pi/4            pi/4   
        4   arreglo de sensores (8)    0               1
        5
        .
        .
        .
        12       

    Actions:
        Type: Discrete(3)
        Num	 Action
        0	 Turn left
        1	 Go straight
        2	 Turn right 

    Reward:
    	reward = -(d**2)

    Starting State:
        
    Episode Termination:
        When the robot reaches the target point or the boundaries of the platform.

"""  

class KheperaPositionObstacleD(gym.Env):
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

		self.Vmax=0.05
		self.Wmax=np.pi/4

		# Action Space
		self.action_space = spaces.Discrete(3)
		# Observation Space
		self.observation_space = spaces.Box(
			low=np.array([0,0,0.,-np.pi/4,0,0,0,0,0,0,0,0], dtype=np.float32),
			high=np.array([3,2*np.pi,0.08,np.pi/4,1,1,1,1,1,1,1,1], dtype=np.float32),
			dtype=np.float32
		)

		# Objetcs in the Simulation Scene

		errorCode, self.right_motor = sim.simxGetObjectHandle(self.clientID, 'K4_Right_Motor',sim.simx_opmode_oneshot_wait)
		errorCode, self.left_motor = sim.simxGetObjectHandle(self.clientID, 'K4_Left_Motor',sim.simx_opmode_oneshot_wait)
		errorCode, self.khepera = sim.simxGetObjectHandle(self.clientID, 'Khepera_IV',sim.simx_opmode_oneshot_wait)
		errorCode, self.target = sim.simxGetObjectHandle(self.clientID, 'Target',sim.simx_opmode_oneshot_wait)  
		errorCode, self.camera = sim.simxGetObjectHandle(self.clientID, 'Vision_sensor',sim.simx_opmode_oneshot_wait)
		self.sensor = {}
		for i in range(8):
			handle = 'K4_Infrared_{}'.format(i+1)
			errorCode, self.sensor[i] = sim.simxGetObjectHandle(self.clientID, handle, sim.simx_opmode_oneshot_wait)

		self.viewer = None
		self.seed()
		self.steps = 0
		self.radius = 0.8
		self.MaxSteps = 800
		self.problem = True
		self.Velocities =[0,0]
		self.Movements = [[4.285,0.515],[4.285,4.285],[0.515,4.285]]

		self.xp, self.yp = self.getPositionTarget()
		self.ResetSimulationScene()

		self.Randomize = True
		self.RobotOrientationRand = True

	def ResetSimulationScene(self):
		errorCode, resolution, image = sim.simxGetVisionSensorImage(self.clientID,self.camera, 0, sim.simx_opmode_streaming)
		pos = sim.simxGetObjectPosition(self.clientID,self.khepera,-1,sim.simx_opmode_streaming)
		ori_body = sim.simxGetObjectOrientation(self.clientID,self.khepera,-1,sim.simx_opmode_streaming)
		tar = sim.simxGetObjectPosition(self.clientID,self.target,-1,sim.simx_opmode_streaming)
		for i in range(8):
			handle = self.sensor[i]
			errorCode,detectionState,detectedPoint,detectedObjectHandle,detectedSurfaceNormalVector=sim.simxReadProximitySensor(self.clientID,handle,sim.simx_opmode_streaming)

	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def getPositionRobot(self):
		pos = sim.simxGetObjectPosition(self.clientID,self.khepera,-1,sim.simx_opmode_buffer)
		ori_body = sim.simxGetObjectOrientation(self.clientID,self.khepera,-1,sim.simx_opmode_buffer)
		theta = ori_body[1][2]
		if theta < 0:
			theta = theta+2.0*np.pi
		xc = pos[1][0]   
		yc = pos[1][1]
		zc = pos[1][2]

		gamma, omega = ori_body[1][0], ori_body[1][1]
		if abs(gamma)>0.2 or abs(omega)>0.2 or zc<-1:
			self.problem = True

		return xc,yc,theta

	def getPositionTarget(self):
		tar = sim.simxGetObjectPosition(self.clientID,self.target,-1,sim.simx_opmode_buffer)
		xp = tar[1][0]
		yp = tar[1][1]
		return xp,yp

	def getSensorsData(self):
		sensor_val = []
		for i in range(8):
			handle = self.sensor[i]
			errorCode,det_State,det_Point,det_Object,NormalVector=sim.simxReadProximitySensor(self.clientID,handle,sim.simx_opmode_buffer)
			if det_State:
				sensor_val.append((0.2-np.linalg.norm(det_Point))/0.2)
			else:
				sensor_val.append(0) 

		return sensor_val

	def updateVelocities(self,Vl,Vr):
		sim.simxSetJointTargetVelocity(self.clientID,self.left_motor,Vl, sim.simx_opmode_oneshot)
		sim.simxSetJointTargetVelocity(self.clientID,self.right_motor,Vr, sim.simx_opmode_oneshot)

	def get_obs(self):
		xc,yc,theta = self.getPositionRobot()
		Sensors_val = self.getSensorsData()
		xp = self.xp
		yp = self.yp
		flag = False
		if abs(xc)>1.1 or abs(yc)>1.1:
			self.problem = True

		d = np.sqrt(((xp-xc)**2)+((yp-yc)**2))      
		alpha = np.arctan2(yp-yc,xp-xc)
		if alpha < 0:
			alpha = alpha+2.0*np.pi
		Oc = np.arctan2(np.sin(alpha-theta),np.cos(alpha-theta))
		v,w = self.Velocities
		self.steps += 1
		return [d,Oc,alpha,v,w],Sensors_val 

	def step(self,action): 
		[d,Oc,alpha,v,w],Sensors= self.get_obs()
		observation = np.append([d,Oc,v,w],Sensors)
		done = False

		if d>=0.05:
			Vl,Vr = self.action_translation(action)
			reward = -(d**2)

		else:
			Vl,Vr = [0,0]
			done = True
			reward = 100

		if max(Sensors)>0.97:
			self.problem = True
		
		if self.problem or self.steps == (self.MaxSteps-1):
			done=True
			reward = -100
			Vl,Vr = [0,0]
		if not done:
			sim.simxSynchronousTrigger(self.clientID)
			self.steps+=1

		self.updateVelocities(Vl,Vr)
		v = (Vr+Vl)/96
		w = 5*(Vr-Vl)/24
		xc,yc,theta = self.getPositionRobot()
		#self.save_images()

		info = {'Distance':d,
				'Oc':Oc,
				'Lineal': v, 
				'Angular' : w,
				'xc':xc,
				'yc':yc}

		return observation, reward, done, info

		if not done:
			sim.simxSynchronousTrigger(self.clientID)

		info = {'Distance':d,
				'Oc':Oc,
				'Lineal': V, 
				'Angular' : W}

		return observation, reward, done, info

	def reset(self):
		# stop the simulation:
		self.problem = False
		sim.simxStopSimulation(self.clientID,sim.simx_opmode_blocking)
		sim.simxSynchronous(self.clientID,True)
		# start the simulation:
		sim.simxStartSimulation(self.clientID,sim.simx_opmode_blocking)
		self.ResetSimulationScene()

		if self.Randomize:
			self.xp, self.yp = self.change_target_position(radius=self.radius)
		else:
			self.xp, self.yp = self.getPositionTarget()
		if self.RobotOrientationRand:
			self.change_robot_orientation()
		self.steps = 0
		[d,Oc,alpha,v,w],Sensors= self.get_obs()
		observation = np.append([d,Oc,v,w],Sensors)
		return observation

	def getCameraImage(self):
		errorCode, resolution, image = sim.simxGetVisionSensorImage(self.clientID, self.camera, 0, sim.simx_opmode_buffer)
		im = np.array(image,dtype=np.uint8)
		im.resize([resolution[1],resolution[0],3])
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

	def change_target_position(self, radius):
		minimum = -radius
		maximum = radius
		X = minimum + (maximum - minimum) * random.random()
		Y = math.sqrt(self.radius**2 - X**2)*(-1)**(random.sample([0,1],1)[0])
		Z = -0.005
		new_position = [X,Y,Z]
		errorCode = sim.simxSetObjectPosition(self.clientID,self.target,-1,new_position,sim.simx_opmode_oneshot)
		return X,Y

	def action_translation(self, action):

		set_of_actions = self.Movements
		Vl,Vr = set_of_actions[action]
		return Vl,Vr

	def change_target_angle(self, angle):
		X = math.cos(math.radians(angle))*self.radius
		Y = math.sin(math.radians(angle))*self.radius
		Z = -0.005
		new_position = [X,Y,Z]
		errorCode = sim.simxSetObjectPosition(self.clientID,self.target,-1,new_position,sim.simx_opmode_oneshot)
		return X,Y

	def change_robot_orientation(self):
		ori_body = sim.simxGetObjectOrientation(self.clientID,self.khepera,-1,sim.simx_opmode_buffer)
		angles = ori_body[1][:]
		angles[2] = 2*np.pi*random.random()
		errorCode = sim.simxSetObjectOrientation(self.clientID,self.khepera, -1, angles,sim.simx_opmode_oneshot) 
