import numpy as np
from matplotlib import pyplot as plt

X = np.array([400, 450, 900, 390, 550])

# TODO: Write the code as explained in the instructions - done
N = 5
ticks = 100
T = np.linspace(0.01, N, ticks)
alpha = min(X)

s = sum([(xi / alpha) ** - (1 / T) for xi in X])

P = np.zeros((ticks, N))

for i in range(len(X)):
    P[:, i] = ((X[i] / alpha) ** - (1 / T)) / s
    plt.plot(T, P[:, i], label=str(X[i]))

plt.xlabel("T")
plt.ylabel("P")
plt.title("Probability as a function of the temperature")
plt.legend()
plt.grid()
plt.show()
exit()
