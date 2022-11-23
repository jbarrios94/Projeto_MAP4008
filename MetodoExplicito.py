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

M = np.array([[1,0],[0,-1]])
autovalores = np.linalg.eigvals(M)

for autovalor in autovalores:
   if 'complex' in str(type(autovalor)):
       sys.exit("La matriz tiene autovalores imaginarios")
       
       
cfl = 1
NumPontos = 101
x0 = 0 
x1 = 1
dx = (x1 - x0)/(NumPontos - 1)
x = np.linspace(x0 - dx, x1 + dx, NumPontos + 2)
t = 0
tf = 0.5
dt = dx * cfl/np.max(autovalores)
Nt = int(round(tf/dt))

u0 = np.zeros((2,NumPontos + 2))
u = u0.copy()
u0[:, 1:-1] = CondicaoInicial(u0[:, 1:-1], x[1:-1])

fig = plt.figure(figsize= (8,8))
ax = fig.add_subplot(2,1,1)
ax.plot(x[1:-1], u0[0,1:-1])
ax.set_title("u_1, tempo t = %.3f" %t)
ax.set_xlim(x0, x1)
ax1 = fig.add_subplot(2,1,2)
ax1.plot(x[1:-1], u0[1,1:-1])
ax1.set_title("u_2, tempo t = %.3f" %t)
ax1.set_xlim(x0, x1)

#Esquema de diferençãs finitas explicito Lax-Wendroff
for i in range(Nt):
    u[:, 1:-1] = (u0[:, 1:-1] - np.dot(M, dt/(2 * dx) * (u0[:, 2:] - u0[:, :-2])) + 
                  np.dot(M**2, dt**2/(2 * dx**2) * (u0[:, 2:] - 2 * u0[:, 1:-1] + u0[:, :-2])) +
                  dt/2 * (f(x[1:-1], t+dt) + f(x[1:-1], t)) -
                  np.dot(M, dt**2/(4 * dx) * (f(x[2:], t) - f(x[:-2], t))))
    
    
    u = Fronteira(u)
    u0 = np.copy(u)
    t += dt
    
u_LW = u.copy()


u0 = np.zeros((2,NumPontos + 2))
u0[:, 1:-1] = CondicaoInicial(u0[:, 1:-1], x[1:-1])

#Esquema de diferençãs finitas implicito Crank-Nicholson
t = 0
u = np.copy(u0)
u1 = u.copy()
u2 = u.copy()
u3 = u.copy()
p1 = u.copy()
p21 = u.copy()
p22 = u.copy()
p23 = u.copy()
mm = -M * dt/(4*dx)
cc = - mm

for i in range(Nt):
    
    p1[0] = 0
    p21[0] = 0
    p22[0] = 1
    p23[0] = 0
    
    for j in range(1, NumPontos + 1):
        denom = np.dot(mm, p1[:, j]) + 1.
        p1[:, j+1] = - np.dot(cc, 1/denom)
        dd = u1[:, j] + np.dot(mm , (u1[:, j+1] - u1[:, j-1]))
        p21[:, j+1] = (dd - np.dot(mm, p21[:, j]))/denom
        p22[:, j+1] = - np.dot(mm, p22[:, j])/denom 
        p23[:, j+1] = - np.dot(mm, p23[:, j])/denom
    
    u1[:, -1] = 0
    u2[:, -1] = 0
    u3[:, -1] = 1
    
    for j in range(NumPontos, -1, -1):
        u1[:, j] = p1[:, j+1]*u1[:, j+1] + p21[:, j+1]
        u2[:, j] = p1[:, j+1]*u2[:, j+1] + p22[:, j+1]
        u3[:, j] = p1[:, j+1]*u3[:, j+1] + p23[:, j+1]
        
    D = (1 - u2[:, -2])*(1 - u3[:, 1]) - u2[:, 1]*u3[:, -1]
    r = u1[:, -2]*(1- u3[:, 1]) + u1[:, 1]*u3[:, -2]
    r = r / D
    s = u1[:, -2]*u2[:, 1] + u1[:, 1]*(1 - u2[:, -2])
    s = s / D
    u = u1 + np.dot(r, u2) + np.dot(s, u3)
    
    t += dt
    u1 = u.copy()
    u2 = u.copy()
    u3 = u.copy()
    
u_CN = u.copy()
fig = plt.figure(figsize= (8,8))
ax = fig.add_subplot(2,1,1)
ax.plot(x[1:-1], u_LW[0,1:-1], label='LW')
ax.plot(x[1:-1], u_CN[0,1:-1], label='CN')
ax.set_title("u_1, tempo t = %.3f" %t)
ax.set_xlim(x0, x1)
ax.legend()
ax1 = fig.add_subplot(2,1,2)
ax1.plot(x[1:-1], u_LW[1,1:-1], label='LW')
ax1.plot(x[1:-1], u_CN[1,1:-1], label='CN')
ax1.set_title("u_2, tempo t = %.3f" %t)
ax1.set_xlim(x0, x1)
ax1.legend()