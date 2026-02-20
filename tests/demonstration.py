import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('tests/demonstration.csv', delimiter=',')
time = data[:, 0]
strain = data[:, 1]
stress = data[:, 2]

plt.plot(time, strain, '.-')
plt.show()

plt.plot(strain, stress)
# plt.xlim([0.0, 0.6])
# plt.ylim([0.0, 30.0])
plt.show()

for eqps in data[:, 3:].T:
    plt.plot(strain, eqps)

plt.show()
