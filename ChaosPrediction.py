# -*- coding: utf-8 -*-

import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import sparse

## Our goal in generating matrices is making element which is needed in training.
def GeneratingMatrices(Parameter):
  # Setting variables
  N = Parameter['N']
  M = Parameter['M']
  density = Parameter['density']
  sigma = Parameter['sigma']
  degree = Parameter['degree']

  # Making weight of reservoir
  sparsity = degree/float(N)
  WeightReservoir = sparse.rand(N, N, density = density).toarray()
  EigenValues = np.linalg.eigvals(WeightReservoir)
  WeightReservoir = (WeightReservoir / np.max(np.abs(EigenValues))) * density

  # Making weight of input
  q = int(N / M)
  WeightInput = np.zeros((N, M))
  for i in range(M):
    np.random.seed(seed = i)
    WeightInput[i * q:(i + 1) * q, i] = sigma * (-1 + 2 * np.random.rand(1, q)[0])

  # Making weight of output arbitrarily
  WeightOutput = np.array([])

  return (WeightInput, WeightReservoir, WeightOutput)

## Our goal in training is making actual weight of output.
def Training(Parameter, Input, Matrices):
  # Setting variables
  N = Parameter['N']
  T = Parameter['T']
  regulation = Parameter['regulation']
  WeightInput, WeightReservoir, WeightOutput = Matrices

  # Making matrix of state (R/C)
  ReservoirStates = np.zeros((N, T))
  for i in range(T - 1):
    ReservoirStates[:,i+1] = np.tanh(np.dot(WeightReservoir, ReservoirStates[:,i]) + np.dot(WeightInput, Input[:,i]))
      
  Temp = ReservoirStates.copy()

  # Non-linear transformation in RC-ESN
  for j in range(2, np.shape(Temp)[0] - 2):
    if j % 2 == 0:
      Temp[j, :] = (Temp[j - 1, :] * Temp[j - 2, :]).copy()

  # Making actual weight of output
  U = np.dot(Temp, Temp.T) + regulation * np.identity(N)
  WeightOutput = np.dot(np.linalg.inv(U), np.dot(Temp, Input.T))
  
  return ReservoirStates, WeightOutput.T

## Our goal in predicting is predicting (x,y,z) at next time interval.
def Predicting(Parameter, ReservoirState, Matrices):
  # Setting variables
  L = Parameter['L']
  P = Parameter['P']
  WeightInput, WeightReservoir, WeightOutput = Matrices
  Output = np.zeros((L, P))

  # Non-linear transformation in RC-ESN
  for i in range(P):
    Temp = ReservoirState.copy()
    for j in range(2, np.shape(Temp)[0] - 2):
      if j % 2 == 0:
        Temp[j] = (Temp[j - 1] * Temp[j - 2]).copy()

    # Making expected output
    out = np.dot(WeightOutput, Temp)
    Output[:,i] = out
    ReservoirState = np.tanh(np.dot(WeightReservoir, ReservoirState) + np.dot(WeightInput, out))
  return Output

## Main part of program
Parameter = {'N': 5000,
             'L': 3,
             'M': 3,
             'density': 0.1,
             'sigma': 0.5,
             'degree': 5,
             'regulation': 0.0001,
             'T': 5000,
             'P': 5000
}

Data = open('/content/lorenz_train5000.txt', 'r').readlines()
Test = open('/content/lorenz_test5000.txt', 'r').readlines()

Input = [line.split() for line in Data]
Input = np.array([[float(i) for i in j] for j in Input]).T

TestInput = [line.split() for line in Test]
TestInput = np.array([[float(i) for i in j] for j in TestInput]).T

figure = plt.figure()
ax1 = figure.gca(projection='3d')
ax2 = figure.gca(projection='3d')

WeightInput, WeightReservoir, WeightOutput = GeneratingMatrices(Parameter)
ReservoirStates, WeightOutput = Training(Parameter, Input, (WeightInput, WeightReservoir, WeightOutput))
ReservoirState = ReservoirStates[:, -1]

Output = Predicting(Parameter, ReservoirState, (WeightInput, WeightReservoir, WeightOutput))

ax1.plot(Output[0, :], Output[1, :], Output[2, :], 'b', linewidth = 1)
ax2.plot(TestInput[0, :], TestInput[1, :], TestInput[2, :], 'g--', linewidth = 0.5)

plt.show()

figure = plt.figure()
ax1 = figure.gca()
ax2 = figure.gca()

ax1.plot(Output[0, :], 'b', linewidth = 1, label='Predicted')
ax2.plot(TestInput[0, :], 'g--', linewidth = 0.5, label='Observed')

plt.legend(loc = 1)
plt.xlabel('$t$ (frame)')
plt.ylabel('$x$')
plt.show()

figure = plt.figure()
ax1 = figure.gca()
ax2 = figure.gca()

ax1.plot(Output[1, :], 'b', linewidth = 1, label='Predicted')
ax2.plot(TestInput[1, :], 'g--', linewidth = 0.5, label='Observed')

plt.legend(loc = 1)
plt.xlabel('$t$ (frame)')
plt.ylabel('$y$')
plt.show()

figure = plt.figure()
ax1 = figure.gca()
ax2 = figure.gca()

ax1.plot(Output[2, :], 'b', linewidth = 1, label='Predicted')
ax2.plot(TestInput[2, :], 'g--', linewidth = 0.5, label='Observed')

plt.legend(loc = 1)
plt.xlabel('$t$ (frame)')
plt.ylabel('$z$')
plt.show()