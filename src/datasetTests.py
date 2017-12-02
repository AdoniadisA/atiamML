# @pierrotechnique
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

fs = 44100. # Sampling frequency
f1 = 55. # Fundamental frequency (n = 1)
n_max = np.floor((0.25*fs)/f1) # Max harmonic below fs/4
N = 4096 # Target number of points
T = float(N)/fs # Corresponding temporal period
t = np.linspace(0,T,N) # Corresponding time vector
betaVect = np.linspace(0.,55.,201,endpoint=False) # Inharmonicity factor [0-f1[
nVect = [i+2 for i in xrange(199)] # Modal indices
f = np.linspace(0,0.25*fs,N/4) # Frequency vector

X = np.zeros((200*200,1024)) # Initialize collector array (for now)
a1 = np.sin(2*np.pi*f1*t) # Calculate fundamental
A1 = abs(np.fft.fft(a1)) # Amplitude spectrum of fundamental
X[0] = A1[0:1024] # Store from 0 to 11025 Hz only
i = 1 # Initialize collector index counter

for beta in betaVect:
    a = np.sin(2*np.pi*f1*t) # Reset fundamental for each new beta value
    for n in nVect:
        a = a + np.sin(2*np.pi*n*(f1+beta)*t) # Progressively add modes
        A = abs(np.fft.fft(a)) # Amplitude spectrum
        X[i] = A[0:1024] # Store it
        if (i%4000 == 0): # Plot 10 example spectra along the way
            plt.plot(f,20*np.log(X[i]))
            plt.show()
        i += 1 # Update index counter