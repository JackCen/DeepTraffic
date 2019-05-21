import numpy as np
from TrafficSimulator import TrafficSimulator

#Evaluate 100 times on 100 actions
def evaluate_policy(T = 100, N = 100):
	rewards = []
	for k in range(N):
		print(k)
		simulator = TrafficSimulator()
		rewards.append(simulate_reward(simulator, T))
	return rewards

def simulate_reward(simulator, T):
	r = np.zeros(T)
	for t in range(T):
		r[t] = simulator.progress(3)
		print(t, r[t])
	return np.mean(r)


results = evaluate_policy()

print(results)
print(np.mean(results))