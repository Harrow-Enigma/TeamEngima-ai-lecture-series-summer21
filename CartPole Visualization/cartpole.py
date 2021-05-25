import tensorflow as tf
from tensorflow import keras
import gym
import numpy as np


class Agent(keras.Model):
  def __init__(self, action_space):
    super(Agent, self).__init__()
    self.output_dim = action_space
    self.dense0 = keras.layers.Dense(100, activation='relu')
    self.dense1 = keras.layers.Dense(100, activation='relu')
    self.dense2 = keras.layers.Dense(self.output_dim)

  def call(self, inputs):
    x = self.dense0(inputs)
    x = self.dense1(x)
    return self.dense2(x)
  
  def act(self, observations):
    self.observations = np.expand_dims(observations, axis=0)
    self.raw = self.predict(self.observations)
    self.raw = tf.nn.softmax(self.raw)
    self.out = np.squeeze(self.raw,0)
    self.action = np.random.choice(self.output_dim,1,p=self.out)
    return self.action[0]


env = gym.make('CartPole-v0')
ACTION_SPACE = env.action_space
OBSERVATION_SPACE = env.observation_space

agent = Agent(ACTION_SPACE.n)

# Pass sample information
obs = env.reset()
agent.act(obs)

agent.load_weights('weights.h5')

obs = env.reset()

for _ in range(1000):
  env.render()
  obs, _, done, _ = env.step(agent.act(obs))

env.close()
