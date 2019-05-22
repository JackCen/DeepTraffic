import numpy as np

class TrafficSimulator(object):
	def __init__(self, numCars = 20, numLanes = 7,
				canvasHeight = 700, canvasWidth = 140, 
				gridHeight = 10, carHeightGird = 4,
				decisionFreq = 5, speedScaling = 15):
		self.canvasSize = [canvasHeight, canvasWidth]
		self.gridHeight = gridHeight
		self.numLanes = numLanes
		self.carHeightGrid = carHeightGird #num of verticaal grids a car occupies
		self.decisionFreq = decisionFreq #how many steps to simulate between two actions
		self.numCars = numCars #other cars
		self.speedScaling = speedScaling #scale mph to grids

		self.reset()

	def reset(self):
		self.grid = np.zeros((self.canvasSize[0] // self.gridHeight, self.numLanes)) # grid consists of the speed of occupied car
		self.initEgoCar()
		self.initCars()

	# Initialize Ego Car parameters
	def initEgoCar(self):
		# Ego car initialized to have top speed 80mph
		self.EgoCarTopSpeed = 80.0
		self.EgoCarSpeedFrac = 1.0
		# Position of ego car. vertical axis always fixed so that we only worry about relative movements of other cars
		self.EgoCarPos = [18.0, 3] 
		# Fill ego car speed into the grid (need to round vertical axis value)
		self.grid[int(np.around(self.EgoCarPos[0])) : (int(np.around(self.EgoCarPos[0])) + self.carHeightGrid), \
				int(np.around(self.EgoCarPos[1]))] = self.EgoCarTopSpeed * self.EgoCarSpeedFrac
		# Tracking history for potential reward function calculation
		self.actionHistory = [0] * 5
		self.speedHistory = [self.EgoCarTopSpeed] * 5

	# Initialize other 20 cars parameters
	def initCars(self):
		# All other cars initialized to have top speed 65mph
		self.carsTopSpeed = np.full((self.numCars), 65.0)
		self.carsSpeedFrac = np.full((self.numCars), 1.0)
		
		# Randomly initialize all car positions
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


	# Take an action for ego car, and simulate {{self.decisionFreq}} steps
	def progress(self, action):
		# Valid actions: [0: stay the same; 1: left; 2: right; 3: accelerate; 4: decelerate]
		self.actionHistory.append(action)
		if len(self.actionHistory) >= 5: self.actionHistory.pop(0) # Potentially only keep a short history

		# Assume all other cars follow a random action
		carAction = np.random.randint(0, 5, self.numCars)
		
		# If action is to turn, retry in the next t if failed to turn
		egoTurned = False
		carTurned = [False] * self.numCars

		#Simulate t steps
		for t in range(self.decisionFreq):
			# Simulate in the order of car positions: top cars move first. This order is easier to track
			# and avoid collisions
			order = np.flip(np.argsort(self.carsPos[:,0], axis = 0), axis = 0)
			egoToMove = True #Need to move ego car in one of the loop

			# Note that in the first loop we only change lane and set up the speeds. Once we have the speed
			# for all cars (including ego), we can then easily move according to relative speed diff
			for i in range(self.numCars):
				carID = order[i]

				# Ego car's turn to move
				if self.carsPos[carID, 0] < self.EgoCarPos[0] and egoToMove:
					if action == 1 and not egoTurned:
						egoTurned = self.egoTurn(-1)
					elif action == 2 and not egoTurned:
						egoTurned = self.egoTurn(1)
					if action == 3:
						self.EgoCarSpeedFrac = min(1.00, self.EgoCarSpeedFrac + 0.01)
					elif action == 4:
						self.EgoCarSpeedFrac = max(0.50, self.EgoCarSpeedFrac - 0.01)
					egoToMove = False
					
					self.checkCollisionEgo() #Check for collision. If dangerous, will change speed

				# other car take actions
				if carAction[carID] == 1 and not carTurned[carID]:
					carTurned[carID] = self.carTurn(-1, carID)
				elif carAction[carID] == 2 and not carTurned[carID]:
					carTurned[carID] = self.carTurn(1, carID)
				elif carAction[carID] == 3:
					self.carsSpeedFrac[carID] = min(1.00, self.carsSpeedFrac[carID] + 0.01)
				elif carAction[carID] == 4:
					self.carsSpeedFrac[carID] = max(0.50, self.carsSpeedFrac[carID] - 0.01)
				self.checkCollisionCar(carID) #Check for collision. If dangerous, will change speed

			# Now update the grid and location of all cars
			self.grid[int(np.around(self.EgoCarPos[0])) : (int(np.around(self.EgoCarPos[0])) + self.carHeightGrid), \
				int(np.around(self.EgoCarPos[1]))] = self.EgoCarTopSpeed * self.EgoCarSpeedFrac
			for i in range(self.numCars):
				carID = order[i]
				self.moveCar(carID)

			#print(self.carsTopSpeed)
			#print(self.carsSpeedFrac)

			# Append speed history for reward
			self.speedHistory.append(self.EgoCarTopSpeed * self.EgoCarSpeedFrac)
			if len(self.speedHistory) >= 5: self.speedHistory.pop(0)

		return self.reward()

	# Check collision for ego car
	def checkCollisionEgo(self):
		# Car in front of us and if with distance < 3, too dangerous, we immediately step on brake
		frontCarSpeed = np.max(self.grid[int(np.around(self.EgoCarPos[0])) + self.carHeightGrid : (int(np.around(self.EgoCarPos[0])) + 2 * self.carHeightGrid), 
										int(np.around(self.EgoCarPos[1]))])
		if frontCarSpeed > 0:
			self.EgoCarSpeedFrac = frontCarSpeed / 2.0 / self.EgoCarTopSpeed #brake reduce our speed to half of front car
		# If car is 4 grids from us, just take its speed and follow
		elif self.grid[int(np.around(self.EgoCarPos[0])) + 2 * self.carHeightGrid, int(np.around(self.EgoCarPos[1]))] > 0:
			self.EgoCarSpeedFrac = self.grid[int(np.around(self.EgoCarPos[0])) + 2 * self.carHeightGrid, int(np.around(self.EgoCarPos[1]))] / self.EgoCarTopSpeed

	# Same check collision for other cars
	def checkCollisionCar(self, carID):
		if int(np.around(self.carsPos[carID, 0])) + self.carHeightGrid >= self.grid.shape[0] or \
			int(np.around(self.carsPos[carID, 0])) + self.carHeightGrid < 0: return
		frontCarSpeed = np.max(self.grid[int(np.around(self.carsPos[carID, 0])) + self.carHeightGrid : min((int(np.around(self.carsPos[carID, 0])) + 2 * self.carHeightGrid), self.grid.shape[0]), 
										int(np.around(self.carsPos[carID, 1]))])
		if frontCarSpeed > 0:
			self.carsSpeedFrac[carID] = frontCarSpeed / 2.0 / self.carsTopSpeed[carID]
		elif (int(np.around(self.carsPos[carID, 0])) + 2 * self.carHeightGrid) < self.grid.shape[0] and self.grid[int(np.around(self.carsPos[carID, 0])) + 2 * self.carHeightGrid, int(np.around(self.carsPos[carID, 1]))] > 0:
			self.carsSpeedFrac[carID] = self.grid[int(np.around(self.carsPos[carID, 0])) + 2 * self.carHeightGrid, int(np.around(self.carsPos[carID, 1]))] / self.carsTopSpeed[carID]

	# Update the position and grid for all other cars according to relative speed diff
	def moveCar(self, carID):
		# remove the grid of our old positions 
		self.grid[max(0, int(np.around(self.carsPos[carID, 0]))): min(int(np.around(self.carsPos[carID, 0])) + self.carHeightGrid, self.grid.shape[0]), int(np.around(self.carsPos[carID, 1]))] = 0.0

		diff = ((self.carsTopSpeed[carID] * self.carsSpeedFrac[carID]) - 
										(self.EgoCarTopSpeed * self.EgoCarSpeedFrac)) / self.speedScaling
		self.carsPos[carID, 0] += diff

		# Move out of bounds, top -> bottom and bottom -> top
		if int(np.around(self.carsPos[carID, 0])) >= self.grid.shape[0]:
			lane = np.random.randint(self.grid.shape[1])
			if self.carsPos[self.carsPos[:,1] == lane, 0].shape[0] == 0:
				gridHeight = -3
			else:
				gridHeight = min(-3, np.min(self.carsPos[self.carsPos[:,1] == lane, 0]) - 2 * self.carHeightGrid)
			self.carsPos[carID, 0] = gridHeight
			self.carsPos[carID, 1] = lane
		elif int(np.around(self.carsPos[carID, 0])) < -3.0:
			lane = np.random.randint(self.grid.shape[1])
			if self.carsPos[self.carsPos[:,1] == lane, 0].shape[0] == 0:
				gridHeight = self.grid.shape[0] - 1
			else:
				gridHeight = max(self.grid.shape[0] - 1, np.max(self.carsPos[self.carsPos[:,1] == lane, 0]) + 2 * self.carHeightGrid)
			self.carsPos[carID, 0] = gridHeight
			self.carsPos[carID, 1] = lane

		# Update grid
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

	# Placeholder reward function
	def reward(self):
		return np.mean(self.speedHistory)

	# Print the grid as temp graphic outputs
	def print_grid(self):
		print(np.flip(self.grid, axis=0))

	# Return state of the simulator
	def state(self):
		#self.EgoCarTopSpeed * self.EgoCarSpeedFrac, self.EgoCarPos, self.actionHistory, 
		#self.speedHistory, self.grid, self.carsTopSpeed * self.carsSpeedFrac, self.carsPos
		toReturn = np.zeros(1 + 2 + 5 + 5 + 7 * 70 + 
							self.numCars + 2 * self.numCars)
		toReturn[0] = self.EgoCarTopSpeed * self.EgoCarSpeedFrac
		toReturn[1] = self.EgoCarPos[0]
		toReturn[2] = self.EgoCarPos[1]
		toReturn[3:8] = self.actionHistory
		toReturn[9:14] = self.speedHistory
		toReturn[15: (15 + 490)] = np.ndarray.flatten(self.grid)
		toReturn[505: 535] = self.carsTopSpeed * self.carsSpeedFrac
		toReturn[535: 575] = np.ndarray.flatten(self.carsPos)
		return toReturn