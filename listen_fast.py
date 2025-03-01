# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 19:18:11 2025

@author: eleph
"""

import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.io import wavfile
import numpy as np
from scipy.signal import find_peaks, peak_prominences, peak_widths
import statistics
from scipy.signal import hilbert, chirp
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from scipy.stats import linregress

from frequency_functions import normalise, find_strike_probabilities, find_first_strikes, do_frequency_analysis, find_strike_times_rounds

import matplotlib
from plot_tools import plotamps, plot_log, plot_freq
import pandas as pd

plt.style.use('default')
cmap = plt.cm.jet

    
def transform(fs, norm_cut):
    #Produce the fourier transform of the input data
    trans1 = abs(fft(norm_cut)[:len(norm_cut)//2])
    return 0.5*trans1*fs/len(norm_cut)
    

def find_best_strikes(fs, dt, cut_length, strike_probs, strikesmax = 10):
    #Using the probabilities, finds the best strikes in the range -- doesn't need any reliable rhythm.
    #JUST to be used to reinforce frequencies as some will be missed out
    
    #This is used to find the strikes used to reinforce rhythm -- NOT the full thing and doesn't care about missed ones
    
    nbells = len(strike_probs)

    strike_probs = gaussian_filter1d(strike_probs, 5, axis = 1)
    
    #Obtain adjusted probs
    strike_probs_adjust = np.zeros(strike_probs.shape)
    strike_probs_adjust = strike_probs[:, :]**4/(np.sum(strike_probs[:,:], axis = 0) + 1e-6)**3

    allpeaks = np.zeros((nbells, strikesmax))
    allconfs = np.zeros((nbells, strikesmax))
    
    print('Finding best peaks...')
    for bell in range(nbells):
        
        probs = strike_probs_adjust[bell]  
        
        peaks, _ = find_peaks(probs)
                
        prominences = peak_prominences(probs, peaks)[0]
        
        peaks = peaks[prominences > 0.5*np.percentile(prominences, 90)]
        prominences = peak_prominences(probs, peaks)[0]

        peaks = np.array([val for _, val in sorted(zip(prominences, peaks), reverse = True)]).astype('int')

        npeaks = min(strikesmax, len(peaks))
        
        peaks = peaks[:npeaks]
        confs = prominences[:npeaks]/np.max(prominences)
        
        allpeaks[bell,:len(peaks)] = sorted(peaks)
        allconfs[bell,:len(peaks)] = confs

        
        print('Bell', bell+1,':',  len(peaks), 'good strikes found...')
        
        
    plot_max = 20   #Do some plotting
    if True:
        fig, axs = plt.subplots(3,4)
        tplots = np.arange(len(strike_probs[bell]))*dt
        for bell in range(nbells):
            ax = axs[bell//4, bell%4]
            ax.plot(tplots, strike_probs[bell,:], linestyle = 'dotted')
            ax.plot(tplots, strike_probs_adjust[bell,:])
            ax.set_title(bell+1)
            ax.set_xlim(0,30)
            ax.scatter(allpeaks[bell]*dt, np.zeros(len(allpeaks[bell])), c = 'black')
            ax.set_xticks([])
            ax.set_yticks([])
        plt.tight_layout()
        plt.show()
         
    return allpeaks, allconfs
    
def plot_strikes(Paras, Data):
    #Plots the things
    fig = plt.figure(figsize = (10,7))
    
    nrows = len(Data.strikes[0])
    yvalues = np.arange(Paras.nbells) + 1
    
    #for bell in range(nbells):
    #    plt.scatter(all_strikes[bell], yvalues[bell]*np.ones(len(all_strikes[bell])),s=all_confidences[bell]*100)
    
    for row in range(nrows):
        plt.plot(Data.strikes[:,row],yvalues)
        order = np.array([val for _, val in sorted(zip(Data.strikes[:,row], yvalues), reverse = False)])
        confs = np.array([val for _, val in sorted(zip(Data.strikes[:,row], Data.strike_certs[:,row]), reverse = False)])
        print('Strikes', row, order, np.array(sorted(Data.strikes[:,row]))*Paras.dt)#, confs)
        #print(all_strikes[:,row])
        #print('Louds', row, order, sorted(all_louds[:,row]))

    plt.xlim(0.0,30.0)
    plt.gca().invert_yaxis()
    plt.close(fig)

def unit_test(all_strikes,dt):
    #Checks changes are all within the right length and end in the correct order
    print('_______________')
    print('Testing...')
    nbells = len(all_strikes)
    nrows = len(all_strikes[0])
    maxlength = 0.0
    for row in range(nrows):
        maxlength = np.max(all_strikes[:,row] - np.min(all_strikes[:,row]))
    print('Max change length', maxlength*dt)
    if maxlength*dt > 3.0:
        raise Exception('Change length not correct.')
    #Check last few changes for rounds
    yvalues = np.arange(nbells) + 1
    order = np.array([val for _, val in sorted(zip(all_strikes[:,nrows-1], yvalues), reverse = False)])
    print('Final change:', order)
    print('_______________')
    return 

def save_strikes(Paras, Data, tower):
    #Saves as a pandas thingummy like the strikeometer does
    allstrikes = []
    allbells = []
    yvalues = np.arange(Paras.nbells) + 1

    if not Paras.handstroke_first:
        for row in range(len(Data.strikes[0])):
            order = np.array([val for _, val in sorted(zip(Data.strikes[:,row], yvalues), reverse = False)])
            allstrikes = allstrikes + sorted((Data.strikes[:,row]).tolist())
            allbells = allbells + order.tolist()
    else:
        for row in range(1, len(Data.strikes[0])):
            order = np.array([val for _, val in sorted(zip(Data.strikes[:,row], yvalues), reverse = False)])
            allstrikes = allstrikes + sorted((Data.strikes[:,row]).tolist())
            allbells = allbells + order.tolist()
        
    allstrikes = 1000*np.array(allstrikes)*Paras.dt
    allbells = np.array(allbells)
    
    data = pd.DataFrame({'Bell No': allbells, 'Actual Time': allstrikes})
    data.to_csv('%s.csv' % tower)  
    return
    
class audio_data():
    #Does the initial audio normalisation things
    def __init__(self, audio_filename):
        self.fs, self.data = wavfile.read(audio_filename)
        if len(self.data.shape) > 1:  #Is stereo
            import_wave = np.array(self.data)[:,0]
        else:  #Isn't
            import_wave = np.array(self.data)[:]
            
        self.signal = normalise(16, import_wave)

class parameters():
    #Contains information like number of bells, max times etc. 
    #Also all variables that can theoretically be easily changed
    def __init__(self, Audio, nominal_freqs, overall_tmin, overall_tmax, rounds_tmax, reinforce_tmax):
                
        self.dt = 0.01
        self.fcut_length = 0.125  #Length of each transform slice (in seconds)
        
        self.transform_smoothing = 0.05 #Transform smoothing for the initial derivatives of the transform (in seconds)
        self.frequency_range = 1    #Range over which to include frequencies in a sweep (as in, 300 will count between 300-range:frequency+range+1 etc.)
        self.derivative_smoothing = 5  #Smoothing for the derivative (in INTEGER time lumps -- could change if necessary...)
        self.smooth_time = 2.0    #Smoothing over which to apply change-long changes (in seconds)
        self.max_change_time = 3.0 #How long could a single change reasonably be
        self.nrounds_max = 8
        self.nreinforce_rows = 16
        
        self.strike_smoothing = 2 #How much to smooth the input probability function
        self.strike_tcut = 1.0 #How many times the average cadence to cut off
        self.strike_alpha = 2  #How much to care about timing
        self.strike_gamma = 2  #How much to care about prominence
        self.strike_gamma_init = 1.5  #How much to care about prominence for the initial rounds
        
        self.freq_tcut = 0.5 #How many times the average cadence to cut off for FREQUENCIES (should be identical strikes really)
        self.freq_smoothing = 5 #How much to smooth the data when looking for frequencies (as an INTEGER)
        self.beta = 2   #How much to care whether strikes are certain when looking at frequencies
        self.freq_filter = 2#How much to filter the frequency profiles (in INT)
        self.n_frequency_picks=  10  #Number of requencies to look for (per bell)
        
        self.rounds_tmax = rounds_tmax
        self.reinforce_tmax = reinforce_tmax
        if overall_tmax > 0.0:
            Audio.signal = Audio.signal[int(overall_tmin*Audio.fs):int(overall_tmax*Audio.fs)]
        else:
            Audio.signal = Audio.signal[int(overall_tmin*Audio.fs):]
        self.nbells = len(nominal_freqs)
        self.fcut_int = 2*int(self.fcut_length*Audio.fs/2)  #Length of this cut (must be even for symmetry purposes)
        self.tmax =  len(Audio.signal)/Audio.fs
        
        self.prob_tcut = 0.1   #Time cutoff for all frequency identification
        self.prob_beta = 1.0  #How much to care about prominence looking at STRIKES
        self.near_freqs = 2  #How much to care about frequency peaks being nearby
        
class data():
    def __init__(self, Paras, Audio):
        #This is called at the start -- can make some things like blank arrays for the nominals and the like. Can also do the FTs here etc (just once)
        self.nominals = np.round(nominal_freqs*Paras.fcut_length).astype('int')

        self.initial_profile = np.identity(Paras.nbells)     #Initial frequencies for the bells -- these are just the nominals
     
        self.ts, self.transform = self.do_fourier_transform(Paras, Audio)
     
        self.transform_derivative = self.find_transform_derivatives()
        
        print('Fourier transform obtained and differentiated (only done once!)')
        
        self.test_frequencies = self.nominals    #This is the case initially
        self.frequency_profile = np.identity(Paras.nbells)   #Each bell corresponds to its nominal frequency alone -- this will later be updated.
        

    def do_fourier_transform(self, Paras, Audio):
        
        full_transform = []; ts = []
        
        t = Paras.fcut_length/2   #Initial time (halfway through each transform)
        
        while t < Paras.tmax - Paras.fcut_length/2:
            cut_start  = int(t*Audio.fs - Paras.fcut_int/2)
            cut_end    = int(t*Audio.fs + Paras.fcut_int/2)
            
            signal_cut = Audio.signal[cut_start:cut_end]
            
            transform_raw = abs(fft(signal_cut)[:len(signal_cut)//2])
            transform = 0.5*transform_raw*Audio.fs/len(signal_cut)
                            
            ts.append(t)        
            full_transform.append(transform)
            
            t = t + Paras.dt
        
        ts = np.array(ts)
        full_transform = np.array(full_transform)    
                
        Paras.nt = len(ts)
        
        return ts, full_transform
    
    def find_transform_derivatives(self):
        allfreqs_smooth = gaussian_filter1d(self.transform, int(Paras.transform_smoothing/Paras.dt), axis = 0)
        diffs = np.zeros(allfreqs_smooth.shape)
        diffs[1:,:] = allfreqs_smooth[1:,:] - allfreqs_smooth[:-1,:] 
        
        diffs[diffs < 0.0] = 0.0
        return diffs
        
        
    
tower_list = ['Nics', 'Stockton', 'Brancepeth']

tower_number = 1

if tower_number == 0:
    fname = 'audio/stedman_nics.wav'
    nominal_freqs = np.array([1439.,1289.5,1148.5,1075.,962.,861.])  #ST NICS
   
if tower_number == 1 :  
    fname = 'audio/stockton_stedman.wav'
    fname = 'audio/stockton_all.wav'
    nominal_freqs = np.array([1892,1679,1582,1407,1252,1179,1046,930,828,780,693,617])

if tower_number == 2:    
    #fs, data = wavfile.read('audio/brancepeth.wav')
    fname = 'audio/Brancepeth_cambridge.wav'
    nominal_freqs = np.array([1230,1099,977,924,821.5,733])

#Input parameters which may need to be changed for given audio
overall_tmin = 0.0
overall_tmax = -1.0    #Max and min values for the audio signal (just trims overall and the data is then gone)

rounds_tmax = 60.0      #Maximum seconds of rounds
reinforce_tmax = 120.0   #Maxmum time to use reinforcement data (should never actually get this close)

n_reinforces = 5   #Number of times the frequencies should be reinforced

#Import the data
Audio = audio_data(fname)

print('Imported audio length:', len(Audio.signal)/Audio.fs, 'seconds')

#Establish parameters, some of which are hard coded into the class
Paras = parameters(Audio, nominal_freqs, overall_tmin, overall_tmax, rounds_tmax, reinforce_tmax)

print('Trimmed audio length:', len(Audio.signal)/Audio.fs, 'seconds')
print('Running assuming', Paras.nbells, 'bells')

Data = data(Paras, Audio) #This class contains all the important stuff, with outputs and things

#Find strike probabilities from the nominals
Data.strike_probabilities = find_strike_probabilities(Paras, Data, Audio, init = True, final = False)
#Find the first strikes based on these probabilities. Hopefully some kind of nice pattern to the treble at least... 

Data.strikes, Data.strike_certs = find_first_strikes(Paras, Data, Audio)

for count in range(n_reinforces):
        
    if True:
        #Find the probabilities that each frequency is useful. Also plots frequency profile of each bell, hopefully.
        print('Doing frequency analysis,  iteration number', count)
        
        Data.test_frequencies, Data.frequency_profile = do_frequency_analysis(Paras, Data, Audio)  
            
        np.save('freqs.npy', Data.test_frequencies)
        np.save('freqprobs.npy', Data.frequency_profile)

        Data.test_frequencies = np.load('freqs.npy')
        Data.frequency_profile = np.load('freqprobs.npy')
        
        print('Finding strike probabilities...')
        
        Data.strike_probabilities = find_strike_probabilities(Paras, Data, Audio, init = False, final = False)
        
        np.save('probs.npy', Data.strike_probabilities)
    
    Data.strike_probabilities = np.load('probs.npy')
    
    strikes, strike_certs = find_strike_times_rounds(Paras, Data, Audio, final = False) #Finds strike times in integer space

    #Filter these strikes for the best rows, to then be used for reinforcement
    best_strikes = []; best_certs = []; allcerts = []
    threshold = 0.2   #Need chnages to be at least this good...
    for row in range(len(strikes[0])):
        allcerts.append(np.min(strike_certs[:,row]))
    threshold = max(threshold, sorted(allcerts, reverse = True)[Paras.nreinforce_rows])
    for row in range(len(strikes[0])):
        if np.min(strike_certs[:,row]) > threshold:
            best_strikes.append(strikes[:,row])
            best_certs.append(strike_certs[:,row])
        
    Data.strikes, Data.strike_certs = np.array(best_strikes).T, np.array(best_certs).T
    
    count += 1

print('Frequency reinforcement complete, finding strike times throughout...')

Data.strike_probabilities = find_strike_probabilities(Paras, Data, Audio, init = False, final = True)

np.save('probs.npy', Data.strike_probabilities)
    
Data.strike_probabilities = np.load('probs.npy')

Data.strikes, Data.strike_certs = find_strike_times_rounds(Paras, Data, Audio, final = True) #Finds strike times in integer space
 
plot_strikes(Paras, Data)
save_strikes(Paras, Data, tower_list[tower_number])

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
