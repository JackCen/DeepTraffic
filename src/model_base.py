import time
import sys
import numpy as np

from config import Config
from replay_buffer import ReplayBuffer
from schedule import LinearSchedule
from TrafficSimulator import TrafficSimulator

class model(object):
	def __init__(self, config):
		self._config = config
		self._eps_schedule = LinearSchedule(
			self._config.eps_begin,
			self._config.eps_end,
			self._config.nsteps)
		self._lr_schedule = LinearSchedule(
			self._config.lr_begin,
			self._config.lr_end,
			self._config.lr_nsteps)
		self._sim = TrafficSimulator(config)
		self._bf = ReplayBuffer(10000, config)

		self._action_fn = self.get_action_fn()

		self.build()

	def build(self):
		pass

	def initialize(self):
		pass

	def get_random_action(self, state):
		pass

	def get_best_action(self, state):
		### return action, q value
		pass

	def get_action(self, state):
		if np.random.random() < self._eps_schedule.get_epsilon():
			return self.get_random_action(state)[0]
		else:
			return self.get_best_action(state)[0]

	def get_random_action_fn(self):
		def random_action_fn(state):
			action = np.random.randint(5)
			return action
		return random_action_fn

	def get_action_fn(self):
		return self.get_action

	def pad_state(self, states, state_history):
		tmp_states = states
		tmp_state = np.concatenate([np.expand_dims(state, -1) for state in tmp_states], axis=-1)
		tmp_state = np.pad(tmp_state, ((0,0),(state_history-tmp_state.shape[-1],0)), 'constant', constant_values=0)
		return [tmp_state]

	def simulate_an_episode(self, T, action_fn):
		rewards = []
		states = []
		actions = []

		self._sim.reset()
		cum_reward = 0


		for t in range(T):
			state = self._sim.state()
			states.append(state)

			state_input = self.pad_state(states[-self._config.state_history:], self._config.state_history)
			action = action_fn(state_input)
			actions.append(action)

			reward = self._sim.progress(action)
			rewards.append(reward)

		return (states, rewards, actions)


	def sampling_buffer(self):
		for s in range(self._config.nBufferSample):
			if s % 20 == 0:
				print("Sample buffer: ", s)
			states, rewards, actions = self.simulate_an_episode(self._config.T, self._action_fn)
			self._bf.store(states, actions, rewards)
