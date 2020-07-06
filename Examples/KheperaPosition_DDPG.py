"""
This Agent is from the Keras-RL2 library

!pip install keras-rl2

"""


import numpy as np
import gym
import gym_kheperaposition
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Flatten, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from rl.callbacks import FileLogger, ModelIntervalCheckpoint
import sim

# Se prepara el entorno
ENV_NAME = 'KheperaPosition-v0'
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
assert len(env.action_space.shape) == 1
nb_actions = env.action_space.shape[0]

# Se crean los modelos de Actor y Critic
actor = Sequential()
actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
actor.add(Dense(16))
actor.add(Activation('relu'))
actor.add(Dense(16))
actor.add(Activation('relu'))
actor.add(Dense(16))
actor.add(Activation('relu'))
actor.add(Dense(nb_actions))
actor.add(Activation('linear'))
print(actor.summary())

action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
flattened_observation = Flatten()(observation_input)
x = Concatenate()([action_input, flattened_observation])
x = Dense(32)(x)
x = Activation('relu')(x)
x = Dense(32)(x)
x = Activation('relu')(x)
x = Dense(32)(x)
x = Activation('relu')(x)
x = Dense(1)(x)
x = Activation('linear')(x)
critic = Model(inputs=[action_input, observation_input], outputs=x)
print(critic.summary())

# Se crea la memoria y el agente, para ser compilado
memory = SequentialMemory(limit=100000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.3)

log_filename = 'ddpg_{}_Khepera_log.json'
callbacks = [FileLogger(log_filename, interval=300)]

agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=900, nb_steps_warmup_actor=900,
                  random_process=random_process, gamma=.99, target_model_update=1e-3)

agent.compile(Adam(lr=.0005, clipnorm=1.), metrics=['mae'])

weights_name = ('ddpg_{}_Khepera_weights.h5')

# Se entrena al Agente y se guardan los pesos
checkpoint_weights_filename = "ddpg_weights-{episode:02d}-{reward:.2f}.h5"
log_filename = 'ddpg_Khepera_log.json'
callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=5000)]
callbacks += [FileLogger(log_filename, interval=1000)]
agent.fit(env, nb_steps=500000, visualize=False, verbose=1, nb_max_episode_steps=300)
agent.save_weights(weights_name, overwrite=True)

# Finalmente, se prueba al agente.
agent.test(env, nb_episodes=1, visualize=True, nb_max_episode_steps=300)