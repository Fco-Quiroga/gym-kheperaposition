# Gym_Env_Khepera_IV

An [OpenAI Gym](https://gym.openai.com/) Environment for the positional control of the mobile robot Khepera IV in [CoppeliaSim](https://www.coppeliarobotics.com/).

![imagen de CoppeliaSim y Khepera](/Img/environment.png)

### Description:

The objective of this environment is to drive the wheeled mobile robot from its current position to a predefined target point.

![gif de entorno funcionando](/Img/position.gif)
      
### Source:

This environment corresponds to the implementation of the [Khepera IV library for robotic control education using V-REP](https://www.sciencedirect.com/science/article/pii/S2405896317323303) for the position control experiment (G. Farías et al.).
 
### Reward:

\begin{equation}
reward = \begin{cases}
r_{arrival} & \text{if the robot reaches the TP}\\
r_{collision}-(d)^2 & \text{if the robot collides}\\
-(d)^2 & \text{in another case}\\
\end{cases}\label{eq5}\end{equation}
        
### Episode Termination:

When the robot reaches the target point or the number of steps of the episode are greater than 300.


#### There are three environments, where the difference between them is the action and observation space.

## KheperaPositionControl-v0

###  Observation:
        Type: Box(3)
	Num	Observation               Min             Max
        0	Distance to the target     0               5
        1	Target Angle              -pi              pi
        2	Error Angle               -pi              pi

###  Actions:
        Type: Box (2)
	Num	Action                    Min             Max
        0       Velocity L Wheel         -7.55            7.55
        1       Velocuty R Wheel         -7.55            7.55        



## KheperaPositionDiscrete-v0

###  Observation:
        Type: Box(3)
	Num	Observation               Min             Max
        0	Distance to the target     0               5
        1	Target Angle              -pi              pi
        2	Error Angle               -pi              pi

###  Actions:
        Type: Discrete(3)
	Num	 Action
        0	 Turn left
        1	 Go straight
        2	 Turn right                



## KheperaPositionPixel-v0

###  Observation:
        RGB Image of 640x480

###  Actions:
        Type: Discrete(3)
	Num	 Action
        0	 Turn left
        1	 Go straight
        2	 Turn right  

