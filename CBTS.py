# import numpy as np

# def LUdecomp(B):
#     R = np.zeros((2, 2))
#     R[0, 0] = B[0, 0]
#     R[0, 1] = B[0, 1]
#     R[1, 0] = B[1, 0]/B[0, 0]
#     R[1, 1] = B[1, 1] - B[0, 1]*R[1, 0]
#     return R

# def ApplyLUbacksubToBlock(A, R):
#     for j in range(2):
#         A[:, j] = ApplyLUbacksubToVector(A[:, j], R)
#     return A

# def ApplyLUbacksubToVector(v, R):
#     x = np.array([(v[0] - R[0, 1]*(v[1] - R[1, 0]*v[0])/R[0, 0]), (v[1] - R[1, 0]*v[0])/R[1, 1]])
#     return x

# NumPontos = 5
# M = np.array([[1,0],[0,1]])
# u0 = np.zeros((2,NumPontos))
# A = np.ones(NumPontos)
# B = A.copy()
# C = A.copy()
# A = []
# B = []
# C = []
# for i in range(NumPontos):
#     A.append(-M/4)
#     B.append(np.eye(2))
#     C.append(M/4)

# A = np.array(A)
# B = np.array(B)
# C = np.array(C)

# U = C.copy() * 0
# R = np.zeros((NumPontos, NumPontos))
# Vtzi = R.copy()
# vty = np.zeros(NumPontos)

# U[0] = A[0]
# U[-1] = C[-1]
# B[0] = B[0] - A[0]
# B[-1] = B[-1] - C[-1]

# for k in range(NumPontos):
#     if k != 0:
#         B[k] = B[k] - np.dot(A[k], B[k-1])
#     R = LUdecomp(B[k])
#     u0[:, k] = u0[:, k] - np.dot(A[k], u0[:, k-1])

#     if k!=1 and k!=NumPontos-1:
#         U[k] = np.dot(-A[k], U[k-1])

#     if k==NumPontos-1:
#         U[k] = U[k] - np.dot(A[k], U[k-1])

#     U[k] = ApplyLUbacksubToBlock(U[k], R)
#     u0[:, k] = ApplyLUbacksubToVector(u0[:, k], R)

#     if k!=NumPontos- 1:
#         C[k] = ApplyLUbacksubToBlock(C[k], R)
#         B[k] = C[k]

# for k in range(NumPontos - 2, 0, -1):
#     u0[:, k] = u0[:, k] - np.dot(B[k], u0[:, k+1])
#     U[k] = U[k] - np.dot(B[k],U[k+1])

# Vtzi = U[0] + U[-1] + np.eye(2)
# R = LUdecomp(Vtzi)
# vty = u0[:, 0] + u0[:, -1]
# vty = ApplyLUbacksubToVector(vty, R)
import numpy as np
# A = np.ones(5)
# B = A.copy()
# C = A.copy()
# A = []
# B = []
# C = []
# M = np.array([[1,0],[0,-1]])
# for i in range(5):
#     A.append(-M * 1/(4 * 1))
#     B.append(np.eye(2))
#     C.append(M * 1/(4 * 1))

# print(C[1])
import matplotlib.pyplot as plt
# x = np.linspace(0, 1, 101)
# y = np.cos(2 * np.pi * x)
# y1 = np.cos(2 * np.pi *(0.5 - x))
# plt.plot(x, y)
# plt.plot(x, y1)
# plt.show()

def CBTS1(A1, B1, C1, U1, NumPontos):
    A = A1.copy()
    B = B1.copy()
    C = C1.copy()
    U = U1.copy()
    Z = np.copy(U) * 0
    for k in range(NumPontos):
        if k!=NumPontos - 1:
            C[k] = np.dot(np.linalg.inv(B[k]), C[k])
            U[k] = np.linalg.inv(B[k])@U[k]
            B[k] = np.linalg.inv(B[k])@B[k]
            B[k+1] = B[k+1] - np.dot(A[k+1], C[k])
            U[k+1] = U[k+1] - np.dot(A[k + 1], U[k])
            A[k+1] = np.zeros((2, 2))
        
        if k==NumPontos - 1:
            C[k] = np.dot(np.linalg.inv(B[k]), C[k])
            U[k] = np.linalg.inv(B[k])@U[k]
            B[k] = np.linalg.inv(B[k])@B[k]

    Z[-1] = U[-1]
    for k in range(NumPontos - 2, -1, -1):
        Z[k] = U[k] - C[k]@Z[k+1]

    return Z

B = []
B.append(np.array([[1,0], [0, -1]]))
B.append(np.array([[1,0], [0, -1]]))
B = np.array(B)
A = B.copy() * 0
C = np.copy(B) * 0
U = C.copy()
U[0] = np.array([[1,2], [3, 4]])
U[1] = np.array([[5,6], [7, 8]])

# print(A)
# print(B)
# print(C)
Z = CBTS1(A, B, C, U, 2)
print(Z)
print(B)
print(B[0]@Z[0] + C[0]@Z[1])
print(A[1]@Z[0] + B[1]@Z[1])


def CBTS1(A1, B1, C1, u01, NumPontos):
    A = np.array(A1[1:-1])
    B = np.array(B1[1:-1])
    C = np.array(C1[1:-1])
    U = np.copy(C) * 0
    Z = np.copy(C) * 0
    f = np.copy(u01[:, 1:-1]) * 0
    y = np.copy(u01[:, 1:-1]) * 0
    x = y.copy() * 0
    U[0] = A[0]
    U[-1] = C[-1]
    B[0] = B[0] - A[0]
    B[-1] = B[-1] - C[-1]

    for k in range(NumPontos):
        
        f[:, k] = u01[:, k+1] + np.dot(C1[k], u01[:, k]) + np.dot(A1[k+2], u01[:, k+2])

        if k!=NumPontos - 1:
            C[k] = np.dot(np.linalg.inv(B[k]), C[k])
            f[:, k] = np.linalg.inv(B[k])@f[:, k]
            U[k] = np.linalg.inv(B[k])@U[k]
            B[k] = np.linalg.inv(B[k])@B[k]
            B[k+1] = B[k+1] - np.dot(A[k+1], C[k])
            f[:, k+1] = f[:, k+1] - np.dot(A[k + 1], f[:, k])
            U[k+1] = U[k+1] - np.dot(A[k + 1], U[k])
            A[k+1] = np.zeros((2, 2))
        
        if k==NumPontos - 1:
            C[k] = np.dot(np.linalg.inv(B[k]), C[k])
            f[:, k] = np.linalg.inv(B[k])@f[:, k]
            U[k] = np.linalg.inv(B[k])@U[k]
            B[k] = np.linalg.inv(B[k])@B[k]

    y[:, -1] = f[:, -1]
    Z[-1] = U[-1]
    for k in range(NumPontos - 2, -1, -1):
        y[:, k] = f[:, k] - C[k]@y[:, k+1]
        Z[k] = U[k] - C[k]@Z[k+1]
    
    Vtzi = Z[0] + Z[-1] + np.eye(2)
    vty = y[:, 0] + y[:, -1]
    vty = np.linalg.inv(Vtzi)@vty

    for k in range(NumPontos):
        x[:, k] = y[:, k] - Z[k]@vty

    return x