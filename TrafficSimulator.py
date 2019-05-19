import numpy as np

class TrafficSimulator(object):
	def __init__(self, numCars = 20, numLanes = 7,
				canvasHeight = 700, canvasWidth = 140, 
				gridHeight = 10, carHeightGird = 4,
				stepTime = 1, decisionFreq = 5):
		self.canvasSize = [canvasHeight, canvasWidth]
		self.numLanes = numLanes
		self.stepTime = stepTime
		self.carHeightGrid = carHeightGird
		self.grid = np.zeros((canvasHeight // gridHeight, numLanes))
		self.decisionFreq = decisionFreq
		self.numCars = numCars
		
		self.initEgoCar()
		self.initCars()


	def initEgoCar(self):
		# Ego car initialized to have speed 80mph
		self.EgoCarTopSpeed = 80.0
		self.EgoCarSpeedFrac = 1.0
		self.EgoCarPos = [18, 3]
		self.grid[self.EgoCarPos[0] : (self.EgoCarPos[0] + self.carHeightGrid), self.EgoCarPos[1]] = self.EgoCarTopSpeed * self.EgoCarSpeedFrac
		self.actionHistory = []
		self.speedHistory = []


	def initCars(self):
		# All other cars initialized to have speed 65mph
		self.carsTopSpeed = np.full((self.numCars), 65.0)
		self.carsSpeedFrac = np.full((self.numCars), 1.0)
		
		# Initialize all car positions
		self.carsPos = np.zeros((self.numCars, 2), dtype = int)
		for i in range(self.numCars):
			gridHeight = np.random.randint(self.grid.shape[0] - self.carHeightGrid)
			lane = np.random.randint(self.grid.shape[1])
			# Need to ensure that cars on the same lane are apart by at least 4 grids to avoid collision
			while (np.sum(self.grid[max(0, gridHeight - self.carHeightGrid):min(gridHeight + 2 * self.carHeightGrid, self.grid.shape[0]), lane]) != 0):
				gridHeight = np.random.randint(self.grid.shape[0] - self.carHeightGrid)
				lane = np.random.randint(self.grid.shape[1])
			self.carsPos[i, 0] = gridHeight
			self.carsPos[i, 1] = lane
			self.grid[self.carsPos[i, 0] : (self.carsPos[i, 0] + self.carHeightGrid), self.carsPos[i, 1]] = self.carsTopSpeed[i] * self.carsSpeedFrac[i]



	def progress(self, action):
		# Valid actions: [0: stay the same; 1: left; 2: right; 3: accelerate; 4: decelerate]
		self.actionHistory.append(action)
		if len(self.actionHistory) >= 5: self.actionHistory.pop(0)

		carAction = np.random.randint(0, 5, self.numCars)
		egoTurned = False
		carTurned = [False] * self.numCars

		for t in range(decisionFreq):
			if action == 3:
				self.EgoCarSpeedFrac = max(1.00, self.EgoCarSpeedFrac + 0.01)
			elif action == 4:
				self.EgoCarSpeedFrac = min(0.01, self.EgoCarSpeedFrac - 0.01)


			order = np.flip(np.argsort(self.carsPos[:,0], axis = 0), axis = 0)
			egoToMove = True
			for i in range(self.numCars):
				carID = order[i]

				if self.carsPos[carID, 0] < self.EgoCarPos[0] and egoToMove:
					if action == 1 and !egoTurned:
						egoTurned = self.egoTurn(-1)
					elif action == 2 and !egoTurned:
						egoTurned = self.egoTurn(1)
					egoToMove = False

				if carAction[carID] == 1 and !carTurned[carID]:
					carTurned[carID] = self.carTurn(-1, carID)
				elif carAction[carID] == 2 and !carTurned[carID]:
					carTurned[carID] = self.carTurn(1, carID)
				elif action == 3:
					self.carsSpeedFrac[carID] = max(1.00, self.carsSpeedFrac[carID] + 0.01)
				elif action == 4:
					self.carsSpeedFrac[carID] = min(0.01, self.carsSpeedFrac[carID] - 0.01)
				self.moveCar(carID)
			self.speedHistory.append(self.EgoCarTopSpeed * self.EgoCarSpeedFrac)
				if len(self.speedHistory) >= 5: self.speedHistory.pop(0)

		return self.reward()

	def moveCar(self, carID):
		return True



	def egoTurn(self, direction):
		# direction: [-1: left; +1: right]
		# Check if already at the leftmost/rightmost lane and if it is safe to change lane 
		if self.EgoCarPos[1] + direction < 0 or \
					self.EgoCarPos[1] + direction >= self.numLanes or \
					np.sum(self.grid[(self.EgoCarPos[0] - self.carHeightGrid):(self.EgoCarPos[0] + 2 * self.carHeightGrid), self.EgoCarPos[1] + direction]) != 0:
			return False
		else:
			self.grid[self.EgoCarPos[0] : (self.EgoCarPos[0] + self.carHeightGrid), self.EgoCarPos[1] + direction] = self.EgoCarTopSpeed * self.EgoCarSpeedFrac
			self.grid[self.EgoCarPos[0] : (self.EgoCarPos[0] + self.carHeightGrid), self.EgoCarPos[1]] = 0.0
			self.EgoCarPos[1] += direction
			return True

	def carTurn(self, direction, carID):
		# direction: [-1: left; +1: right]
		# Check if already at the leftmost/rightmost lane and if it is safe to change lane 
		if self.carsPos[carID, 1] + direction < 0 or \
					self.carsPos[carID, 1] + direction >= self.numLanes or \
					np.sum(self.grid[max(0, self.carsPos[carID, 0] - self.carHeightGrid):min(self.carsPos[carID, 0] + 2 * self.carHeightGrid, self.grid.shape[0]), self.carsPos[carID, 1] + direction]) != 0:
			return False
		else:
			self.grid[max(0, self.carsPos[carID, 0]): min(self.carsPos[carID, 0] + self.carHeightGrid, self.grid.shape[0]), self.carsPos[carID, 1] + direction] = self.carsTopSpeed[carID] * self.carsSpeedFrac[carID]
			self.grid[max(0, self.carsPos[carID, 0]): min(self.carsPos[carID, 0] + self.carHeightGrid, self.grid.shape[0]), self.carsPos[carID, 1]] = 0.0
			self.carsPos[carID, 1] += direction
			return True

	def self.reward(self):
		return 1.0
		
	def print_grid(self):
		print(np.flip(self.grid, axis=0))