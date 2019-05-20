import numpy as np

class TrafficSimulator(object):
	def __init__(self, numCars = 20, numLanes = 7,
				canvasHeight = 700, canvasWidth = 140, 
				gridHeight = 10, carHeightGird = 4,
				stepTime = 1, decisionFreq = 5,
				speedScaling = 15):
		self.canvasSize = [canvasHeight, canvasWidth]
		self.numLanes = numLanes
		self.stepTime = stepTime
		self.carHeightGrid = carHeightGird
		self.grid = np.zeros((canvasHeight // gridHeight, numLanes))
		self.decisionFreq = decisionFreq
		self.numCars = numCars
		self.speedScaling = speedScaling

		self.initEgoCar()
		self.initCars()


	def initEgoCar(self):
		# Ego car initialized to have speed 80mph
		self.EgoCarTopSpeed = 80.0
		self.EgoCarSpeedFrac = 1.0
		self.EgoCarPos = [18.0, 3]
		self.grid[int(np.around(self.EgoCarPos[0])) : (int(np.around(self.EgoCarPos[0])) + self.carHeightGrid), \
				int(np.around(self.EgoCarPos[1]))] = self.EgoCarTopSpeed * self.EgoCarSpeedFrac
		self.actionHistory = []
		self.speedHistory = []


	def initCars(self):
		# All other cars initialized to have speed 65mph
		self.carsTopSpeed = np.full((self.numCars), 65.0)
		self.carsSpeedFrac = np.full((self.numCars), 1.0)
		
		# Initialize all car positions
		self.carsPos = np.zeros((self.numCars, 2))
		for i in range(self.numCars):
			gridHeight = np.random.randint(self.grid.shape[0] - self.carHeightGrid)
			lane = np.random.randint(self.grid.shape[1])
			# Need to ensure that cars on the same lane are apart by at least 4 grids to avoid collision
			while (np.sum(self.grid[max(0, gridHeight - self.carHeightGrid):min(gridHeight + 2 * self.carHeightGrid, self.grid.shape[0]), lane]) != 0):
				gridHeight = np.random.randint(self.grid.shape[0] - self.carHeightGrid)
				lane = np.random.randint(self.grid.shape[1])
			self.carsPos[i, 0] = gridHeight
			self.carsPos[i, 1] = lane
			self.grid[int(np.around(self.carsPos[i, 0])) : (int(np.around(self.carsPos[i, 0])) + self.carHeightGrid), 
						int(np.around(self.carsPos[i, 1]))] = self.carsTopSpeed[i] * self.carsSpeedFrac[i]



	def progress(self, action):
		# Valid actions: [0: stay the same; 1: left; 2: right; 3: accelerate; 4: decelerate]
		self.actionHistory.append(action)
		if len(self.actionHistory) >= 5: self.actionHistory.pop(0)

		carAction = np.random.randint(0, 5, self.numCars)
		egoTurned = False
		carTurned = [False] * self.numCars

		for t in range(decisionFreq):
			order = np.flip(np.argsort(self.carsPos[:,0], axis = 0), axis = 0)
			egoToMove = True
			for i in range(self.numCars):
				carID = order[i]

				if self.carsPos[carID, 0] < self.EgoCarPos[0] and egoToMove:
					if action == 1 and not egoTurned:
						egoTurned = self.egoTurn(-1)
					elif action == 2 and not egoTurned:
						egoTurned = self.egoTurn(1)
					if action == 3:
						self.EgoCarSpeedFrac = max(1.00, self.EgoCarSpeedFrac + 0.01)
					elif action == 4:
						self.EgoCarSpeedFrac = min(0.01, self.EgoCarSpeedFrac - 0.01)
					egoToMove = False

					self.checkCollisionEgo()

				if carAction[carID] == 1 and not carTurned[carID]:
					carTurned[carID] = self.carTurn(-1, carID)
				elif carAction[carID] == 2 and not carTurned[carID]:
					carTurned[carID] = self.carTurn(1, carID)
				elif action == 3:
					self.carsSpeedFrac[carID] = max(1.00, self.carsSpeedFrac[carID] + 0.01)
				elif action == 4:
					self.carsSpeedFrac[carID] = min(0.01, self.carsSpeedFrac[carID] - 0.01)
				self.checkCollisionCar(carID)

			for i in range(self.numCars):
				carID = order[i]
				self.moveCar(carID)

			self.speedHistory.append(self.EgoCarTopSpeed * self.EgoCarSpeedFrac)
			if len(self.speedHistory) >= 5: self.speedHistory.pop(0)

		return self.reward()


	def checkCollisionEgo(self):
		frontCarSpeed = np.max(self.grid[int(np.around(self.EgoCarPos[0])) + self.carHeightGrid : (int(np.around(self.EgoCarPos[0])) + 2 * self.carHeightGrid), 
										int(np.around(self.EgoCarPos[1]))])
		if frontCarSpeed > 0:
			self.EgoCarSpeedFrac = frontCarSpeed / 2.0 / self.EgoCarTopSpeed
		elif self.grid[int(np.around(self.EgoCarPos[0])) + 2 * self.carHeightGrid] > 0:
			self.EgoCarSpeedFrac = self.grid[int(np.around(self.EgoCarPos[0])) + 2 * self.carHeightGrid] / self.EgoCarTopSpeed

	def checkCollisionCar(self, carID):
		frontCarSpeed = np.max(self.grid[int(np.around(self.carsPos[carID, 0])) + self.carHeightGrid : (int(np.around(self.carsPos[carID, 0])) + 2 * self.carHeightGrid), 
										int(np.around(self.carsPos[carID, 1]))])
		if frontCarSpeed > 0:
			self.carsSpeedFrac[carID] = frontCarSpeed / 2.0 / self.carsTopSpeed[carID]
		elif self.grid[int(np.around(self.carsPos[carID, 0])) + 2 * self.carHeightGrid] > 0:
			self.carsSpeedFrac[carID] = self.grid[int(np.around(self.EgoCarPos[0])) + 2 * self.carHeightGrid] / self.carsTopSpeed[carID]

	def moveCar(self, carID):
		self.grid[max(0, int(np.around(self.carsPos[carID, 0]))): min(int(np.around(self.carsPos[carID, 0])) + self.carHeightGrid, self.grid.shape[0]), int(np.around(self.carsPos[carID, 1]))] = 0.0

		diff = [(self.carsTopSpeed[carID] * self.carsSpeedFrac[carID]) - 
										(self.EgoCarTopSpeed * self.EgoCarSpeedFrac)] / self.speedScaling
		self.carsPos[carID,0] += diff

		# Move out of bounds
		if int(np.around(self.carsPos[carID, 0])) >= self.grid.shape[0]:
			lane = np.random.randint(self.grid.shape[1])
			gridHeight = -3
			# Need to ensure that cars on the same lane are apart by at least 4 grids to avoid collision
			while (np.sum(self.grid[0:5, lane]) != 0):
				lane = np.random.randint(self.grid.shape[1])
			self.carsPos[carID, 0] = gridHeight
			self.carsPos[carID, 1] = lane
		elif int(np.around(self.carsPos[carID, 0])) < -3.0:
			lane = np.random.randint(self.grid.shape[1])
			gridHeight = self.grid.shape[0] - 1
			# Need to ensure that cars on the same lane are apart by at least 4 grids to avoid collision
			while (np.sum(self.grid[(self.grid.shape[0] - 4):self.grid.shape[0], lane]) != 0):
				lane = np.random.randint(self.grid.shape[1])
			self.carsPos[carID, 0] = gridHeight
			self.carsPos[carID, 1] = lane

		self.grid[max(0, int(np.around(self.carsPos[carID, 0]))): min(int(np.around(self.carsPos[carID, 0])) + self.carHeightGrid, self.grid.shape[0]), int(np.around(self.carsPos[carID, 1]))] = self.carsTopSpeed[carID] * self.carsSpeedFrac[carID]

		return True



	def egoTurn(self, direction):
		# direction: [-1: left; +1: right]
		# Check if already at the leftmost/rightmost lane and if it is safe to change lane 
		if self.EgoCarPos[1] + direction < 0 or \
					self.EgoCarPos[1] + direction >= self.numLanes or \
					np.sum(self.grid[(int(np.around(self.EgoCarPos[0])) - self.carHeightGrid):(int(np.around(self.EgoCarPos[0])) + 2 * self.carHeightGrid), int(np.around(self.EgoCarPos[1])) + direction]) != 0:
			return False
		else:
			self.grid[int(np.around(self.EgoCarPos[0])) : (int(np.around(self.EgoCarPos[0])) + self.carHeightGrid), int(np.around(self.EgoCarPos[1])) + direction] = self.EgoCarTopSpeed * self.EgoCarSpeedFrac
			self.grid[int(np.around(self.EgoCarPos[0])) : (int(np.around(self.EgoCarPos[0])) + self.carHeightGrid), int(np.around(self.EgoCarPos[1]))] = 0.0
			self.EgoCarPos[1] += direction
			return True

	def carTurn(self, direction, carID):
		# direction: [-1: left; +1: right]
		# Check if already at the leftmost/rightmost lane and if it is safe to change lane 
		if self.carsPos[carID, 1] + direction < 0 or \
					self.carsPos[carID, 1] + direction >= self.numLanes or \
					np.sum(self.grid[max(0, int(np.around(self.carsPos[carID, 0])) - self.carHeightGrid):min(int(np.around(self.carsPos[carID, 0])) + 2 * self.carHeightGrid, self.grid.shape[0]), int(np.around(self.carsPos[carID, 1])) + direction]) != 0:
			return False
		else:
			self.grid[max(0, int(np.around(self.carsPos[carID, 0]))): min(int(np.around(self.carsPos[carID, 0])) + self.carHeightGrid, self.grid.shape[0]), int(np.around(self.carsPos[carID, 1])) + direction] = self.carsTopSpeed[carID] * self.carsSpeedFrac[carID]
			self.grid[max(0, int(np.around(self.carsPos[carID, 0]))): min(int(np.around(self.carsPos[carID, 0])) + self.carHeightGrid, self.grid.shape[0]), int(np.around(self.carsPos[carID, 1]))] = 0.0
			self.carsPos[carID, 1] += direction
			return True

	def reward(self):
		return 1.0

	def print_grid(self):
		print(np.flip(self.grid, axis=0))