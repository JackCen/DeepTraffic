import time
import sys
import numpy as np
import sys

import tensorflow as tf
import tensorflow.contrib.layers as layers
import os

from config import Config
from replay_buffer import ReplayBuffer
from schedule import LinearSchedule
from dqn_model import DQN
from TrafficSimulator import TrafficSimulator

def evaluate_policy(model, T = 100, N = 100):
	rewards = []
	for k in range(N):
		print ('I am at the %d episode'%(k))
		states, reward, actions = model.simulate_an_episode(T, model.get_best_action_fn())
		print (reward)
		rewards.append(np.mean(reward))
	return rewards

def main():
	config = Config()
	config.mode = 'test'
	config.dropout = 1.0
	model = DQN(config)
	model.initialize()
	rewards = evaluate_policy(model, config.T, config.N)
	print(np.mean(rewards))

if __name__ == '__main__':
	main()