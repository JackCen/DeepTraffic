import numpy as np

class TrafficSimulator(object):
	def __init__(self, numCars = 20, numLanes = 7,
				canvasHeight = 700, canvasWidth = 140, 
				gridHeight = 10, carHeightGird = 4,
				stepTime = 1):
		self.canvasSize = [canvasHeight, canvasWidth]
		self.numLanes = numLanes
		self.stepTime = stepTime
		self.carHeightGrid = carHeightGird
		self.grid = np.zeros((canvasHeight // gridHeight, numLanes))

		self.numCars = numCars
		
		self.initEgoCar()
		self.initCars()


	def initEgoCar(self):
		self.EgoCarTopSpeed = 80.0
		self.EgoCarSpeedFrac = 1.0
		self.EgoCarPos = (18, 3)
		self.grid[self.EgoCarPos[0] : (self.EgoCarPos[0] + 4), self.EgoCarPos[1]] = self.EgoCarTopSpeed * self.EgoCarSpeedFrac



	def initCars(self):
		self.carsTopSpeed = np.full((self.numCars), 65.0)
		self.carsSpeedFrac = np.full((self.numCars), 1.0)
		self.carsPos = np.zeros((self.numCars, 2), dtype=int)
		for i in range(self.numCars):
			gridHeight = np.random.randint(self.grid.shape[0] - 4)
			lane = np.random.randint(self.grid.shape[1])
			while (np.sum(self.grid[max(0, gridHeight - 4):min(gridHeight + 7, self.grid.shape[0] - 1), lane]) != 0):
				gridHeight = np.random.randint(self.grid.shape[0] - 4)
				lane = np.random.randint(self.grid.shape[1])
			self.carsPos[i, 0] = gridHeight
			self.carsPos[i, 1] = lane
			self.grid[self.carsPos[i, 0] : (self.carsPos[i, 0] + 4), self.carsPos[i, 1]] = self.carsTopSpeed[i] * self.carsSpeedFrac[i]



	def print_grid(self):
		print(np.flip(self.grid, axis=0))