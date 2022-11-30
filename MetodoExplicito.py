#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 14:44:11 2022

@author: cbarrios
"""

import numpy as np
import sys
import matplotlib.pyplot as plt

def f(x, t):
    """
    Função que avalia o termo fonte  na equção u_t + M(x)u_x = f(x, t)
    
    Parameters
    ----------
    x : np.array([])
        Pontos em que se avalia a função.
    t : float
        Tempo em que se avalia a função.

    Returns
    -------
    Array de tamano 2x1
    Valores da fonte na equção u_t + M(x)u_x = f(x, t)
    
    """
    return np.array([[0],[0]])

def Fronteira(u, tipo='periodicas', ezquerda = np.array([[0],[0]]), direita=np.array([[0],[0]])):
    """
    Parameters
    ----------
    u : np.array(shape=(2,:))
        Variable na qual se aplica a codição de fronteira
    tipo : String, optional 
        Pode ser 'periodicas' ou 'fixa'
        Tipo de fornteira. The default is 'periodicas'.
    ezquerda : np.array(shape=(2,1)), optional
        se tipo é fixa então debe dar o valor a impor como fronteira fixa do lado esquerdo. 
        The default is np.array([[0],[0]]).
    direita : TYPE, optional
        se tipo é fixa então debe dar o valor a impor como fronteira fixa do lado direito. 
        The default is np.array([[0],[0]]).
    """
    if tipo == 'periodicas':
        u[:, 0] = u[:, -2]
        u[:, -1] = u[:, 1]
    else:
        u[:, 0] = ezquerda
        u[:, -1] = direita
    
    return u
        
def CondicaoInicial(u, x):
    """
    Parameters
    ----------
    u : Variable na qual se aplica a codição inicial
    
    x : Variavel espacial

    Returns
    -------
    A condição inicial.
    """
    u[0, :] = np.exp(-(x - 0.5)**2/0.005)
    u[1, :] = np.exp(-(x - 0.5)**2/0.005)
    return u

def ApplyLUbacksubToBlock(A, R):
    A1 = A.copy()
    for j in range(2):
        A1[:, j] = ApplyLUbacksubToVector(A[:, j], R)
    return A1

def ApplyLUbacksubToVector(v, R):
    x = np.array([(v[0] - R[0, 1]*(v[1] - R[1, 0]*v[0]))/R[0, 0], (v[1] - R[1, 0]*v[0])/R[1, 1]])
    return x

def LUdecomp(B):
    R = np.zeros((2, 2))
    R[0, 0] = B[0, 0]
    R[0, 1] = B[0, 1]
    R[1, 0] = B[1, 0]/B[0, 0]
    R[1, 1] = B[1, 1] - B[0, 1]*R[1, 0]
    return R

def CBTS(A1, B1, C1, u01, NumPontos):
    A = np.copy(A1)
    B = np.copy(B1)
    C = np.copy(C1)
    u0 = np.copy(u01)

    U = C.copy() * 0

    U[0] = A[0]
    U[-1] = C[-1]
    B[0] = B[0] - A[0]
    B[-1] = B[-1] - C[-1]

    for k in range(NumPontos):
        if k != 0:
            B[k] = B[k] - np.dot(A[k], B[k-1])
            u0[:, k] = u0[:, k] - np.dot(A[k], u0[:, k-1]) 

        R = LUdecomp(B[k])

        if k!=0 and k!=NumPontos-1:
            U[k] = np.dot(-A[k], U[k-1])

        if k==NumPontos-1:
            U[k] = U[k] - np.dot(A[k], U[k-1])

        U[k] = ApplyLUbacksubToBlock(U[k], R)
        u0[:, k] = ApplyLUbacksubToVector(u0[:, k], R)

        if k!=NumPontos- 1:
            C[k] = ApplyLUbacksubToBlock(C[k], R)
            B[k] = C[k]

    for k in range(NumPontos - 2, -1, -1):
        u0[:, k] = u0[:, k] - np.dot(B[k], u0[:, k+1])
        U[k] = U[k] - np.dot(B[k],U[k+1])

    Vtzi = U[0] + U[-1] + np.eye(2)
    R = LUdecomp(Vtzi)
    vty = u0[:, 0] + u0[:, -1]
    vty = ApplyLUbacksubToVector(vty, R)

    for k in range(NumPontos):
        u0[:, k] = u0[:, k] - np.dot(U[k], vty)

    return u0

M = np.array([[1,0],[0,-1]])

autovalores = np.linalg.eigvals(M)
print(autovalores)

for autovalor in autovalores:
   if 'complex' in str(type(autovalor)):
       sys.exit("La matriz tiene autovalores imaginarios")
       
       
cfl = 1.5
NumPontos = 201
x0 = 0 
x1 = 1
dx = (x1 - x0)/(NumPontos - 1)
x = np.linspace(x0 - dx, x1 + dx, NumPontos + 2)
t = 0
tf = 1
dt = dx * cfl/np.max(np.abs(autovalores))
Nt = int(round(tf/dt))

u0 = np.zeros((2,NumPontos + 2))
u = u0.copy()
u0[:, 1:-1] = CondicaoInicial(u0[:, 1:-1], x[1:-1])

# fig = plt.figure(figsize= (8,8))
# ax = fig.add_subplot(2,1,1)
# ax.plot(x[1:-1], u0[0,1:-1])
# ax.set_title("u_1, tempo t = %.3f" %t)
# ax.set_xlim(x0, x1)
# ax1 = fig.add_subplot(2,1,2)
# ax1.plot(x[1:-1], u0[1,1:-1])
# ax1.set_title("u_2, tempo t = %.3f" %t)
# ax1.set_xlim(x0, x1)

#Esquema de diferençãs finitas explicito Lax-Wendroff
for i in range(Nt):
    u[:, 1:-1] = (u0[:, 1:-1] - np.dot(M, dt/(2 * dx) * (u0[:, 2:] - u0[:, :-2])) + 
                  np.dot(np.dot(M, M), dt**2/(2 * dx**2) * (u0[:, 2:] - 2 * u0[:, 1:-1] + u0[:, :-2])) +
                  dt/2 * (f(x[1:-1], t+dt) + f(x[1:-1], t)) -
                  np.dot(M, dt**2/(4 * dx) * (f(x[2:], t) - f(x[:-2], t))))
    
    
    u = Fronteira(u)
    u0 = np.copy(u)
    t += dt
    
u_LW = u.copy()


u0 = np.zeros((2,NumPontos))
u0[:, :] = CondicaoInicial(u0[:, :], x[1:-1])

#Esquema de diferençãs finitas implicito Crank-Nicholson
t = 0
u = np.copy(u0)
A = np.ones(NumPontos)
B = A.copy()
C = A.copy()
A = []
B = []
C = []
for i in range(NumPontos):
    A.append(-M * dt/(2 * dx))
    B.append(np.eye(2))
    C.append(M * dt/(2 * dx))

A = np.array(A)
B = np.array(B)
C = np.array(C)

for i in range(Nt):
    u = CBTS(A, B, C, u0, NumPontos)
    u0 = u.copy()
    t += dt
    
u_CN = u.copy()
fig = plt.figure(figsize= (8,8))
ax = fig.add_subplot(2,1,1)
ax.plot(x[1:-1], u_LW[0,1:-1], label='LW')
ax.plot(x[1:-1], u_CN[0, :], label='CN')
ax.set_title("u_1, tempo t = %.3f" %t)
ax.set_xlim(x0, x1)
ax.legend()
ax1 = fig.add_subplot(2,1,2)
ax1.plot(x[1:-1], u_LW[1,1:-1], label='LW')
ax1.plot(x[1:-1], u_CN[1, :], label='CN')
ax1.set_title("u_2, tempo t = %.3f" %t)
ax1.set_xlim(x0, x1)
ax1.legend()
plt.show()

fig = plt.figure(figsize= (8,8))
ax = fig.add_subplot(2,1,1)
ax.plot(x[1:-1], u_LW[0,1:-1], label='LW')
ax.set_title("u_1, tempo t = %.3f" %t)
ax.set_xlim(x0, x1)
ax.legend()
ax1 = fig.add_subplot(2,1,2)
ax1.plot(x[1:-1], u_LW[1,1:-1], label='LW')
ax1.set_title("u_2, tempo t = %.3f" %t)
ax1.set_xlim(x0, x1)
ax1.legend()
plt.show()

fig = plt.figure(figsize= (8,8))
ax = fig.add_subplot(2,1,1)
ax.plot(x[1:-1], u_CN[0, :], label='CN')
ax.set_title("u_1, tempo t = %.3f" %t)
ax.set_xlim(x0, x1)
ax.legend()
ax1 = fig.add_subplot(2,1,2)
ax1.plot(x[1:-1], u_CN[1, :], label='CN')
ax1.set_title("u_2, tempo t = %.3f" %t)
ax1.set_xlim(x0, x1)
ax1.legend()
plt.show()