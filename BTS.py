import numpy as np
import matplotlib.pyplot as plt


# M = np.array([[1,1],[1,-1]])
# autovalores = np.linalg.eigvals(M)
# cfl = 0.5
# NumPontos = 201
# x0 = 0 
# x1 = 1
# dx = (x1 - x0)/(NumPontos - 1)
# x = np.linspace(x0 - dx, x1 + dx, NumPontos + 2)
# t = 0
# tf = 2
# dt = dx * cfl/np.max(np.abs(autovalores))
# Nt = int(round(tf/dt))

# u0 = np.zeros((2,NumPontos))
# u0[:, :] = np.cos(2 * np.pi * x[1:-1])

# #Esquema de diferençãs finitas implicito Crank-Nicholson
# t = 0
# u = np.copy(u0)
# A = np.ones(NumPontos)
# B = A.copy()
# C = A.copy()
# A = []
# B = []
# C = []
# for i in range(NumPontos):
#     A.append(-M * dt/(4 * dx))
#     B.append(np.eye(2))
#     C.append(M * dt/(4 * dx))

# A = np.array(A)
# B = np.array(B)
# C = np.array(C)
# U = np.copy(C) * 0
# f = np.copy(u0) * 0
# y = np.copy(u0) * 0
# U[0] = A[0]
# U[-1] = C[-1]

# B[0] = B[0] - A[0]
# B[-1] = B[-1] - C[-1]

# for k in range(NumPontos):
#     if k==0:
#         f[:, k] = u0[:, k] + np.dot(C[-2],u0[:, -2]) + np.dot(A[k+1], u0[:, k+1])
    
#     if k == NumPontos - 1:
#         f[:, k] = u0[:, k] + np.dot(C[k-1],u0[:, k-1]) + np.dot(A[1], u0[:, 1])

#     if k!= 0 and k!=NumPontos - 1:
#         f[:, k] = u0[:, k] + np.dot(C[k-1], u0[:, k-1]) + np.dot(A[k+1], u0[:, k+1])

#     if k!=NumPontos - 1:
#         B[k] = np.eye(2)
#         C[k] = np.dot(np.linalg.inv(B[k]), C[k])
#         f[:, k] = np.linalg.solve(B[k], f[:, k])
#         B[k+1] = B[k+1] - np.dot(A[k+1], C[k])
#         f[:, k+1] = np.linalg.solve(A[k + 1], f[:, k-1])
#         A[k+1] = np.zeros((2, 2))
    
#     if k==NumPontos - 1:
#         B[k] = np.eye(2)
#         C[k] = np.dot(np.linalg.inv(B[k]), C[k])
#         f[:, k] = np.linalg.solve(B[k], f[:, k])

# y[:, -1] = f[:, -1]
# for k in range(NumPontos - 2, -1, -1):
#     y[:, k] = f[:, k] - np.linalg.solve(C[k], y[:, k+1])

# A = np.array([[1,2], [2, 1]])
# B = np.array([[5,3], [2,1]])
# C = np.linalg.inv(A)@B
# D = A * 0
# D[:, 0] = np.linalg.solve(A,B[:, 0])
# D[:, 1] = np.linalg.solve(A,B[:, 1])
# print(C)
# print(D)

# A = np.array([[1, 1], [1, -1]])
# b = np.array([1, 1])

# x = np.linalg.solve(A, b)
# print(x)
# print(np.dot(x,A.T))

# def ThomasBlock(A, B, C, u0, NumPontos):
#     P = A.copy() * 0
#     q = u0.copy() * 0
#     P[1] = np.eye(2)

#     for  i in range(1, NumPontos - 1):
#         InvA = np.linalg.inv(A[i] @ P[i] + B[i])
#         dd = u0[:, i] + A[i] @ u0[:, i + 1] + C[i] @ u0[:, i - 1]
#         P[i + 1] = -InvA @ C[i]
#         q[:, i + 1] = InvA @ (dd - A[i] @ q[:, i])

#     u0[:, -1] = np.linalg.inv(np.eye(2) - P[-1]) @ q[:, -1]

#     for i in range(NumPontos - 2, -1, -1):
#         u0[:, i] = P[i + 1] @ u0[:, i + 1] + q[:, i + 1]

#     return u0

def LUdecomp(B, N):
    L = np.eye(N)
    U = np.zeros((N, N))
    U[0, :] = B[0, :]
    for  i in range(1, N):
        for j in range(i):
            sum = 0
            for k in range(i):
                sum = sum + L[i, k] * U[k, j]
            L[i, j] = 1/U[j, j] * (B[i, j] - sum)
        
        for j in range(i, N):
            sum = 0
            for k in range(i):
                sum = sum + L[i, k] * U[k, j]
            U[i, j] = B[i, j] - sum
    return L, U

def ApplyLUbacksubToVector(L, U, b, N):
    y = np.zeros(N)
    x = y.copy()
    y[0] = b[0]/L[0, 0]
    for i in range(1, N):
        sum = 0
        for j in range(i):
            sum = sum + L[i, j] * y[j]
        y[i] = 1/L[i, i] * (b[i] - sum)
    
    x[-1] = y[-1]/U[-1 ,-1]
    for i in range(N-1, -1, -1):
        sum = 0
        for j in range(i + 1, N):
            sum = sum + U[i, j] * x[j]
        x[i] = 1/U[i, i] * (y[i] - sum)
    return x

def ApplyLUbacksubToBlock(L, U, B, N):
    B1 = np.zeros(B.shape)
    for j in range(N):
        B1[:, j] = ApplyLUbacksubToVector(L, U, B[:, j], N)
        # print(B1[:, j])
    return B1


A = np.array([ [1, 2 ,3], [1, 3 ,2], [0, 1 ,1] ])
L, U = LUdecomp(A, 3)
# print(L)
# print(U)
b = np.array([[1,1,1], [1,1,1], [1,1,1]])
x = ApplyLUbacksubToBlock(L, U, b, 3)
print(x)
print(np.linalg.inv(A)@b)


