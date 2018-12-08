import numpy as np

counts = []
mode = 'real'

if mode is 'sim':
	simobs = np.load('obs.npy')
	for obs in simobs:
		count = np.count_nonzero(obs == -1) / 2
		counts.append(7-count)

if mode is 'real':
	f = open('note_random.txt','r')
	for line in f:
		token = line.strip().split(',')
		obs = np.array([float(i) for i in token[4:]])
		count = np.count_nonzero(obs == -1) / 2
		counts.append(7-count)

	
import matplotlib.pyplot as plt

n, bins, patches = plt.hist(x=counts, bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()


sums = [0, 0, 0, 0, 0, 0, 0, 0]

for i in counts:
	sums[int(i)]+=1

ratio = 0
for i in range(8):
	print(i, sums[i]/sum(sums))	
	ratio += i * sums[i]/sum(sums)
print(ratio)
