import numpy as np
from scipy.linalg import eig, inv, logm


F = np.array(
    [
        [0.63595746, 0.69157849, 0.71520784],
        [0.80589604, 0.83687323, 0.19312595],
        [0.05387420, 0.86551549, 0.41880244],
    ]
)
h = 1e-6

derivative_f = np.zeros((3, 3, 3, 3))
for i in range(3):
    for j in range(3):
        for k in range(3):
            for l in range(3):
                C_plus = F.T @ F
                C_plus[k][l] += 0.5 * h
                C_minus = F.T @ F
                C_minus[k][l] -= 0.5 * h
                derivative_f[i][j][k][l] = (
                    logm(C_plus)[i][j] - logm(C_minus)[i][j]
                ) / h

print(derivative_f)

C = F.T @ F
C_inv = inv(C)
derivative_a = np.zeros((3, 3, 3, 3))
for i in range(3):
    for j in range(3):
        for k in range(3):
            for l in range(3):
                derivative_a[i][j][k][l] = 0.25 * (
                    C_inv[i][k] * (j == l)
                    + C_inv[j][k] * (i == l)
                    + C_inv[i][l] * (j == k)
                    + C_inv[j][l] * (i == k)
                )

print("--------------------------------")
print(derivative_a)

C = F.T @ F
e, Q = eig(C)
e = np.real(e)
E = np.diag(e)

R = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        if e[i] != e[j]:
            R[i][j] = (np.log(e[i]) - np.log(e[j])) / (e[i] - e[j])
        else:
            R[i][j] = 1.0 / e[j]

grad = np.zeros((3, 3, 3, 3))
for i in range(3):
    for j in range(3):
        for k in range(3):
            for l in range(3):
                grad[i][j][k][l] = 0.5 * ((i == k) * (j == l) + (j == k) * (i == l))

derivative = np.zeros((3, 3, 3, 3))
for i in range(3):
    for j in range(3):
        for k in range(3):
            for l in range(3):
                derivative[i][j][k][l] = (Q @ (R * (Q.T @ grad[k][l] @ Q)) * Q.T)[i][j]

print("--------------------------------")
print(derivative)
