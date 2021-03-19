from __future__ import division
from __future__ import print_function
import warnings
import timeit
import json
from tempfile import mkdtemp
from PIL import Image

import numpy as np
import tensorflow as tf
from tensorflow.keras import __version__ as KERAS_VERSION
from tensorflow.python.keras.callbacks import Callback as KerasCallback, CallbackList as KerasCallbackList
from tensorflow.python.keras.utils.generic_utils import Progbar

class Callback(KerasCallback):
    def _set_env(self, env):
        self.env = env

    def on_episode_begin(self, episode, logs={}):
        """Called at beginning of each episode"""
        pass

    def on_episode_end(self, episode, logs={}):
        """Called at end of each episode"""
        pass

    def on_step_begin(self, step, logs={}):
        """Called at beginning of each step"""
        pass

    def on_step_end(self, step, logs={}):
        """Called at end of each step"""
        pass

    def on_action_begin(self, action, logs={}):
        """Called at beginning of each action"""
        pass

    def on_action_end(self, action, logs={}):
        """Called at end of each action"""
        pass

class TestLogger(Callback):
    def __init__(self):
        self.filepath = 'test.pckl'
        self.data = {}
        self.paso = []
        self.observation = []
        self.reward = []
        self.infoOc = []
        self.infoD = []
        self.Lineal = []
        self.Angular = []
        self.xc = []
        self.yc = []


    def on_step_end(self, step, logs):

        #print(logs)

        self.paso.append(step)
        self.observation.append(logs['observation'])
        self.reward.append(logs['reward'])
        #self.info.append(logs['info'])
        self.infoOc.append(float(logs['info']['Oc']))
        self.infoD.append(float(logs['info']['Distance']))
        self.Lineal.append(float(logs['info']['Lineal']))
        self.Angular.append(float(logs['info']['Angular']))
        self.xc.append(float(logs['info']['xc']))
        self.yc.append(float(logs['info']['yc']))

    def on_episode_end(self, episode, logs={}):
        self.data={
            'paso' : self.paso,
            'observation' : self.observation,
            'reward' : self.reward,
            'Oc' : self.infoOc,
            'Distance' :self.infoD,
            'Lineal' : self.Lineal,
            'Angular' : self.Angular,
            'xc' : self.xc,
            'yc' : self.yc
        }

        self.save_data()

    def save_data(self):

        import pickle
        f = open(self.filepath, 'wb')
        pickle.dump(self.data, f)
        f.close()

