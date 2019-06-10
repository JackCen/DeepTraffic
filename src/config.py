class Config:
	def __init__(self):
		self.mode = 'train'

		self.nsteps_train = 100000
		self.print_freq = 50
		self.target_update_freq = 1000
		self.saving_freq = 2500
		self.simulation_freq = 2000
		self.model_output = '../outputs'

		self.eps_begin = 1.0
		self.eps_end = 0.1
		self.nsteps = 100000
		self.dropout = 0.6

		self.lr_begin = 0.001
		self.lr_end = 0.0001
		self.lr_nsteps = self.nsteps_train / 2

		self.gamma = 0.99
		self.grad_clip = True
		self.clip_val = 10
		self.batch_size = 64

		self.T = 100
		self.N = 100
		# self.nBufferSample = 1000
		self.nBufferSample = 100

		self.hidden_size = [128, 64, 32]
		self.numActions = 5

		self.numCars = 20
		self.numLanes = 7
		self.canvasHeight = 700
		# self.canvasHeight = 300
		self.canvasWidth = 140
		self.gridHeight = 10
		self.carHeightGrid = 4
		# self.decisionFreq = 5
		self.decisionFreq = 1
		self.speedScaling = 5
		self.actionSpeedHistory = 5
		# self.actionSpeedHistory = 1
		self.egoTopSpeed = 80.0
		self.carTopSpeed = 65.0
		self.acc = 1
		self.minSpeedFrac = 0.5
		self.egoCarPos = 18.0
		# self.egoCarPos = 10.0
		self.state_length = 1 + 2 + 2 * self.actionSpeedHistory + 3 * self.numCars
								# + self.numLanes * self.canvasHeight // self.gridHeight
								
		self.state_history = 1