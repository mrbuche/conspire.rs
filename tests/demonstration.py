import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('tests/demonstration.csv', delimiter=',')
time = data[:, 0]
strain = data[:, 1]
stress = data[:, 2]

plt.plot(strain, stress)
plt.show()

for eqps in data[:, 3:].T:
    plt.plot(strain, eqps)

plt.show()
