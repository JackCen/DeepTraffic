import numpy as np
from TrafficSimulator import TrafficSimulator
from config import Config

def get_best_action_fn():
	return 3

def evaluate_policy(config):
	rewards = []
	for k in range(config.N):
		print(k)
		simulator = TrafficSimulator(config)
		rewards.append(simulate_reward(simulator, config.T))
	return rewards

def simulate_reward(simulator, T):
	r = np.zeros(T)
	for t in range(T):
		r[t] = simulator.progress(get_best_action_fn())
		# print(t, r[t])
	return r

def main():
	config = Config()
	rewards = evaluate_policy(config)
	print(np.mean(rewards))

if __name__ == '__main__':
	main()
