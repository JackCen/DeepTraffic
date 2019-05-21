class Config:
	def __init__(self):
		self.mode = 'train'

		self.nsteps_train = 1000
		self.print_freq = 50
		self.target_update_freq = 1000
		self.saving_freq = 250
		self.simulation_freq = 1000
		self.model_output = './output'

		self.eps_begin = 1.0
		self.eps_end = 0.1
		self.nsteps = 1000
		self.dropout= 0.9

		self.lr_begin = 0.00025
		self.lr_end = 0.00005
		self.lr_nsteps = self.nsteps_train / 2

		self.gamma = 0.99
		self.grad_clip = True
		self.clip_val = 10
		self.batch_size = 32

		self.T = 100

		self.hidden_size= 10

		self.state_shape = [100, 1]
		self.state_history = 1