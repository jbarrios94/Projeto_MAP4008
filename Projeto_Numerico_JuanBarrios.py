#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 14:44:11 2022

@author: cbarrios
"""

import numpy as np
import sys
import matplotlib.pyplot as plt
import time
        
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
    return B1

def CBTS(A1, B1, C1, u01, NumPontos):
    A = np.copy(A1)
    B = np.copy(B1)
    C = np.copy(C1)
    U1 = np.zeros(C.shape)
    Z = np.zeros(C.shape)
    f = np.copy(u01[:, 1:-1]) * 0
    y = np.copy(u01[:, 1:-1]) * 0
    x = y.copy() * 0
    U1[0] = A[0].copy()
    U1[-1] = C[-1].copy()
    B[0] = B[0] - A[0]
    B[-1] = B[-1] - C[-1]
    for k in range(NumPontos):
        
        f[:, k] = u01[:, k+1] + np.dot(C1[k], u01[:, k]) + np.dot(A1[k], u01[:, k+2])

        if k!= 0:
            B[k] = B[k] - np.dot(A[k], C[k-1])
            f[:, k] = f[:, k] - np.dot(A[k], f[:, k-1])
            U1[k] = U1[k] - np.dot(A[k], U1[k-1])

        L, U = LUdecomp(B[k], 2)
        C[k] = ApplyLUbacksubToBlock(L, U, C[k], 2)
        f[:, k] = ApplyLUbacksubToVector(L, U, f[:, k], 2)
        U1[k] = ApplyLUbacksubToBlock(L, U, U1[k], 2)
        B[k] = np.eye(2)

    y[:, -1] = f[:, -1]
    Z[-1] = U1[-1]
    for k in range(NumPontos - 2, -1, -1):
        y[:, k] = f[:, k] - C[k]@y[:, k+1]
        Z[k] = U1[k] - C[k]@Z[k+1]
    
    Vtzi = Z[0] + Z[-1] + np.eye(2)
    vty = y[:, 0] + y[:, -1]
    L, U = LUdecomp(Vtzi, 2)
    vty = ApplyLUbacksubToVector(L, U, vty, 2)

    for k in range(NumPontos):
        x[:, k] = y[:, k] - Z[k]@vty

    return x


## -------PARAMETROS INICIAIS ------------------------------------

cfl = 1.5
NumPontos = 101
x0 = 0
x1 = 1
dx = (x1 - x0)/(NumPontos - 1)
x = np.linspace(x0, x1, NumPontos)
t = 0
tf = 1.

#-----------------------------------------------------------------

#Definição das entradas da matriz M
teste = 1

if teste == 1:
    m11 = np.ones(x.shape)
    m12 = np.zeros(x.shape)
    m21 = np.zeros(x.shape)
    m22 = np.ones(x.shape) * -1
elif teste == 2:
    m11 = np.ones(x.shape)
    m12 = np.ones(x.shape)
    m21 = np.ones(x.shape)
    m22 = np.ones(x.shape) * -1
elif teste == 3:
    m11 = np.where(x<=1/3, 2, 0) + np.where(x1>=1/3 and x<=2/3, 1, 0) + np.where(x>2/3, 2, 0)
    m12 = np.where(x<=1/3, 2, 0) + np.where(x1>=1/3 and x<=2/3, 1, 0) + np.where(x>2/3, 2, 0)
    m21 = np.where(x<=1/3, 2, 0) + np.where(x1>=1/3 and x<=2/3, 1, 0) + np.where(x>2/3, 2, 0)
    m22 = np.where(x<=1/3, -2, 0) + np.where(x1>=1/3 and x<=2/3, -1, 0) + np.where(x>2/3, -2, 0)
elif teste == 4:
    m11 = np.cos(x)
    m12 = np.cos(x)
    m21 = np.sin(x)
    m22 = np.sin(x)

MaximoAutovalor = 0

for i in range(NumPontos):
    M = np.array([ [m11[i], m12[i]], [m21[i], m22[i]] ])
    autovalores = np.linalg.eigvals(M)
    for autovalor in autovalores:
        if np.abs(autovalor) > MaximoAutovalor:
            MaximoAutovalor = np.abs(autovalor)
        if 'complex' in str(type(autovalor)):
            sys.exit("La matriz tiene autovalores imaginarios")



dt = dx * cfl/MaximoAutovalor
Nt = int(round(tf/dt))

#Esquema de diferençãs finitas implicito Crank-Nicholson

u0 = np.zeros((2,NumPontos+2))
u0[:, 1:-1] = CondicaoInicial(u0[:, 1:-1], x)
u0[:, 0] = u0[:, -2]
u0[:, -1] = u0[:, 1]

t = 0
u = np.copy(u0)
AA = []
BB = []
CC = []
for i in range(NumPontos):
    M = np.array([ [m11[i], m12[i]], [m21[i], m22[i]] ])
    AA.append(-M * dt/(4 * dx))
    BB.append(np.eye(2))
    CC.append(M * dt/(4 * dx))

A1 = np.array(AA)
B1 = np.array(BB)
C1 = np.array(CC)

for i in range(Nt):
    u[:, 1:-1] = CBTS(A1, B1, C1, u0[:, :], NumPontos)
    u[:, 0] = u[:, -2]
    u[:, -1] = u[:, 1]
    u0 = u.copy()
    t += dt


u_CN = u.copy()

t = 0
u0 = np.zeros((2,NumPontos+2))
u = u0.copy()
u0[:, 1:-1] = CondicaoInicial(u0[:, 1:-1], x)

#Esquema de diferençãs finitas explicito Lax-Wendroff
u0[:, 0] = u0[:, -2]
u0[:, -1] = u0[:, 1]

for i in range(Nt):
    u1 = u0.copy()
    u[0, 1:-1] = (u0[0, 1:-1] - m11 * dt/(2 * dx) * (u0[0, 2:] - u0[0, :-2]) - m12 * dt/(2 * dx) * (u0[1, 2:] - u0[1, :-2]) + 
                  (m11 * m11 + m12 * m21) * dt**2/(2 * dx**2) * (u0[0, 2:] - 2 * u0[0, 1:-1] + u0[0, :-2]) + (m11 * m12 + m12 * m22) * dt**2/(2 * dx**2) * (u0[1, 2:] - 2 * u0[1, 1:-1] + u0[1, :-2]))

    u[1, 1:-1] = (u1[1, 1:-1] - m21 * dt/(2 * dx) * (u1[0, 2:] - u1[0, :-2]) - m22 * dt/(2 * dx) * (u1[1, 2:] - u1[1, :-2]) + 
                  (m21 * m11 + m22 * m21)  * dt**2/(2 * dx**2) * (u1[0, 2:] - 2 * u1[0, 1:-1] + u1[0, :-2]) + (m21 * m12 + m22 * m22)* dt**2/(2 * dx**2) * (u1[1, 2:] - 2 * u1[1, 1:-1] + u1[1, :-2]))
        
    u[:, 0] = u[:, -2]
    u[:, -1] = u[:, 1]
    # u = Fronteira(u0)
    u0 = np.copy(u)
    t += dt

u_LW = u0.copy()


fig = plt.figure(figsize= (8,8))
ax = fig.add_subplot(2,1,1)
ax.plot(x[:], u_LW[0, 1:-1], label='LW')
ax.plot(x[:], np.exp(-((x ) - 0.5)**2/0.005), label='Exata')
ax.set_title("u_1, tempo t = %.3f" %t)
ax.set_xlim(x0, x1)
ax.legend()
ax1 = fig.add_subplot(2,1,2)
ax1.plot(x[:], u_LW[1, 1:-1], label='LW')
ax1.set_title("u_2, tempo t = %.3f" %t)
ax1.set_xlim(x0, x1)
ax1.legend()

fig = plt.figure(figsize= (8,8))
ax = fig.add_subplot(2,1,1)
ax.plot(x[:], u_CN[0, 1:-1], label='CN')
ax.set_title("u_1, tempo t = %.3f" %t)
ax.set_xlim(x0, x1)
ax.legend()
ax1 = fig.add_subplot(2,1,2)
ax1.plot(x[:], u_CN[1, 1:-1], label='CN')
ax1.set_title("u_2, tempo t = %.3f" %t)
ax1.set_xlim(x0, x1)
ax1.legend()
plt.show()