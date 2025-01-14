#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 15:36:42 2025

@author: trcn27
"""
import matplotlib.pyplot as plt
import numpy as np

def plot_log(ts, log, title = 0, strikes = [], mags = []):
    fig = plt.figure(figsize = (5,5))

    plt.plot(ts, log)
    if len(strikes) > 0:
        scale = 100.0/max(mags)
        for i, strike in enumerate(strikes):
            plt.scatter(ts[strike], log[strike], s = mags[i]*scale, c= 'black')
    plt.title(title)
    #plt.xlim(5.5, 20)
    plt.show()

def plotamps(ts, amps, strikes, bell, xmin = 0, xmax = 30):
    plt.plot(ts, amps[bell])
    fact = len(ts)/ts[-1]
    ampsplot = (np.array(strikes[bell])*fact).astype('int')
    plt.scatter(strikes[bell],amps[bell][ampsplot], c = 'black')
    plt.xlim(xmin, xmax)
    plt.title(bell)
    plt.show()
    
def plot_freq(toplot, fs, freqs_reference, title = 0, end = True):
    #Plots the frequency profile with the bell references
    #fig = plt.figure(figsize = (5,5))
    if len(freqs_reference) > 0:
        freq_ints = np.array((len(toplot))*2.0*freqs_reference/fs).astype('int')   #Integer values for the correct frequencies
        for freq in freq_ints:
            plt.plot([freq,freq], [0,max(toplot)])
        plt.xlim(0,1.2*freq_ints[0])

    xs = np.linspace(0,1,len(toplot))*fs
    plt.plot(toplot, c= 'black')
    if end:
        plt.title(title)
        plt.show()
