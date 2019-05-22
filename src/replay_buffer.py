import numpy as np

class ReplayBuffer(object):
	def __init__(self, size, config):
		self.config = config
		self.size = size
		self.last_idx = -1
		self.history_size = 0

		self.states_stack = np.empty([self.size]+[self.config.state_length]+[self.config.state_history+1], dtype=np.float32)
		self.actions = np.empty([self.size], dtype=np.int32)
		self.rewards = np.empty([self.size], dtype=np.float32)

	def store(self, states, actions, rewards):
		for idx in range(len(actions)):
			self.last_idx += 1
			if self.last_idx == self.size:
				self.last_idx = 0
			self.actions[self.last_idx] = actions[idx]
			self.rewards[self.last_idx] = rewards[idx]
			
			tmp = states[max(idx - self.config.state_history + 1, 0) : (idx + 2)]
			tmp_state = np.concatenate([np.expand_dims(state, -1) for state in tmp_states], axis=-1)
			self.states_stack[self.last_idx] = np.pad(tmp_state, ((0,0),(self.config.state_history+1-tmp_state.shape[-1],0)), 'constant', constant_values=0)
			self.history_size += 1

	def sample(self, batch_size):
		idx = np.arange(min(self.size, self.history_size))
		np.random.shuffle(idx)
		idx_choice = idx[:batch_size]

		states = self.states_stack[idx_choice][:,:,:-1]
		states_p = self.states_stack[idx_choice][:,:,1:]
		actions = self.actions[idx_choice]
		rewards = self.rewards[idx_choice]

		return (states, states_p, actions, rewards)