import numpy as np

def LUdecomp(B):
    R = np.zeros((2, 2))
    R[0, 0] = B[0, 0]
    R[0, 1] = B[0, 1]
    R[1, 0] = B[1, 0]/B[0, 0]
    R[1, 1] = B[1, 1] - B[0, 1]*R[1, 0]
    return R

def ApplyLUbacksubToBlock(A, R):
    for j in range(2):
        A[:, j] = ApplyLUbacksubToVector(A[:, j], R)
    return A

def ApplyLUbacksubToVector(v, R):
    x = np.array([(v[0] - R[0, 1]*(v[1] - R[1, 0]*v[0])/R[0, 0]), (v[1] - R[1, 0]*v[0])/R[1, 1]])
    return x

NumPontos = 5
M = np.array([[1,0],[0,1]])
u0 = np.zeros((2,NumPontos))
A = np.ones(NumPontos)
B = A.copy()
C = A.copy()
A = []
B = []
C = []
for i in range(NumPontos):
    A.append(-M/4)
    B.append(np.eye(2))
    C.append(M/4)

A = np.array(A)
B = np.array(B)
C = np.array(C)

U = C.copy() * 0
R = np.zeros((NumPontos, NumPontos))
Vtzi = R.copy()
vty = np.zeros(NumPontos)

U[0] = A[0]
U[-1] = C[-1]
B[0] = B[0] - A[0]
B[-1] = B[-1] - C[-1]

for k in range(NumPontos):
    if k != 0:
        B[k] = B[k] - np.dot(A[k], B[k-1])
    R = LUdecomp(B[k])
    u0[:, k] = u0[:, k] - np.dot(A[k], u0[:, k-1])

    if k!=1 and k!=NumPontos-1:
        U[k] = np.dot(-A[k], U[k-1])

    if k==NumPontos-1:
        U[k] = U[k] - np.dot(A[k], U[k-1])

    U[k] = ApplyLUbacksubToBlock(U[k], R)
    u0[:, k] = ApplyLUbacksubToVector(u0[:, k], R)

    if k!=NumPontos- 1:
        C[k] = ApplyLUbacksubToBlock(C[k], R)
        B[k] = C[k]

for k in range(NumPontos - 2, 0, -1):
    u0[:, k] = u0[:, k] - np.dot(B[k], u0[:, k+1])
    U[k] = U[k] - np.dot(B[k],U[k+1])

Vtzi = U[0] + U[-1] + np.eye(2)
R = LUdecomp(Vtzi)
vty = u0[:, 0] + u0[:, -1]
vty = ApplyLUbacksubToVector(vty, R)
