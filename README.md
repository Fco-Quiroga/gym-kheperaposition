# Gym_Env_Khepera_IV

An [OpenAI Gym](https://gym.openai.com/) Environment for the positional control of the mobile robot Khepera IV in [CoppeliaSim](https://www.coppeliarobotics.com/).

![imagen de CoppeliaSim y Khepera](/Img/CoppeliaSim_Khepera.png)

### Description:

The objective of this environment is to drive the wheeled mobile robot from its current position to a predefined target point.

![gif de entorno funcionando](/Img/Control-Khepera.gif)
      
### Source:

This environment corresponds to the implementation of the [Khepera IV library for robotic control education using V-REP](https://www.sciencedirect.com/science/article/pii/S2405896317323303) for the position control experiment (G. Farías et la.).

### Observation Space:

        Type: Box(3)
        Num	Observation               Min             Max
        0	Distance to the target     0               5
        1	Target Angle              -pi              pi
        2	Error Angle               -pi              pi

### Action Space:

        Type: Box (1)
        Num	Action                    Min             Max
        0	Angular Velocity         -pi/2            pi/2 
 
### Reward:

    	reward = -(d**2)
If the khepera reaches the target point, the reward = 300-steps      (the number of steps of the current episode).
        
### Episode Termination:

When the robot reaches the target point or the number of steps of the episode are greater than 300.
