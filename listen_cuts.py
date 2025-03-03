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
    

def plot_strikes(Paras):
    #Plots the things
    fig = plt.figure(figsize = (10,7))
    
    nrows = len(Paras.allstrikes[0])
    yvalues = np.arange(Paras.nbells) + 1
    
    #for bell in range(nbells):
    #    plt.scatter(all_strikes[bell], yvalues[bell]*np.ones(len(all_strikes[bell])),s=all_confidences[bell]*100)
    
    for row in range(nrows):
        plt.plot(Paras.allstrikes[:,row],yvalues)
        order = np.array([val for _, val in sorted(zip(Paras.allstrikes[:,row], yvalues), reverse = False)])
        confs = np.array([val for _, val in sorted(zip(Paras.allstrikes[:,row], Paras.allcerts[:,row]), reverse = False)])
        print('Strikes', row, order, np.array(sorted(Paras.allstrikes[:,row]))*Paras.dt)#, confs)
        #print(all_strikes[:,row])
        #print('Louds', row, order, sorted(all_louds[:,row]))

    plt.xlim(0.0,30.0)
    plt.gca().invert_yaxis()
    plt.close(fig)

def save_strikes(Paras, tower):
    #Saves as a pandas thingummy like the strikeometer does
    allstrikes = []
    allbells = []
    yvalues = np.arange(Paras.nbells) + 1

    if not Paras.handstroke_first:
        for row in range(len(Paras.allstrikes[0])):
            order = np.array([val for _, val in sorted(zip(Paras.allstrikes[:,row], yvalues), reverse = False)])
            allstrikes = allstrikes + sorted((Paras.allstrikes[:,row]).tolist())
            allbells = allbells + order.tolist()
    else:
        for row in range(1, len(Paras.allstrikes[0])):
            order = np.array([val for _, val in sorted(zip(Paras.allstrikes[:,row], yvalues), reverse = False)])
            allstrikes = allstrikes + sorted((Paras.allstrikes[:,row]).tolist())
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
    def __init__(self, Audio, nominal_freqs, overall_tmin, overall_tmax, rounds_tmax, reinforce_tmax, overall_tcut):
                
        self.dt = 0.01
        self.fcut_length = 0.125  #Length of each transform slice (in seconds)
        
        self.transform_smoothing = 0.05 #Transform smoothing for the initial derivatives of the transform (in seconds)
        self.frequency_range = 1    #Range over which to include frequencies in a sweep (as in, 300 will count between 300-range:frequency+range+1 etc.)
        self.derivative_smoothing = 5  #Smoothing for the derivative (in INTEGER time lumps -- could change if necessary...)
        self.smooth_time = 2.0    #Smoothing over which to apply change-long changes (in seconds)
        self.max_change_time = 3.0 #How long could a single change reasonably be
        self.nrounds_max = 8
        self.nreinforce_rows = 16
        
        self.strike_smoothing = 1 #How much to smooth the input probability function
        self.strike_tcut = 1.0 #How many times the average cadence to cut off
        self.strike_alpha = 2  #How much to care about timing
        self.strike_gamma = 2  #How much to care about prominence
        self.strike_gamma_init = 1.5  #How much to care about prominence for the initial rounds
        
        self.freq_tcut = 0.5 #How many times the average cadence to cut off for FREQUENCIES (should be identical strikes really)
        self.freq_smoothing = 5 #How much to smooth the data when looking for frequencies (as an INTEGER)
        self.beta = 1   #How much to care whether strikes are certain when looking at frequencies
        self.freq_filter = 2#How much to filter the frequency profiles (in INT)
        self.n_frequency_picks=  10  #Number of requencies to look for (per bell)
        
        self.rounds_tmax = rounds_tmax
        self.reinforce_tmax = reinforce_tmax
        
        self.overall_tcut = overall_tcut  #How frequently (seconds) to do update rounds etc.
        
        if overall_tmax > 0.0:
            Audio.signal = Audio.signal[int(overall_tmin*Audio.fs):int(overall_tmax*Audio.fs)]
        else:
            Audio.signal = Audio.signal[int(overall_tmin*Audio.fs):]
            
        self.overall_tmax = overall_tmax
        self.nbells = len(nominal_freqs)
        self.fcut_int = 2*int(self.fcut_length*Audio.fs/2)  #Length of this cut (must be even for symmetry purposes)
        self.tmax =  len(Audio.signal)/Audio.fs
        
        self.prob_tcut = 0.1   #Time cutoff for all frequency identification
        self.prob_beta = 1.0  #How much to care about prominence looking at STRIKES
        self.near_freqs = 2  #How much to care about frequency peaks being nearby
        
        self.allstrikes = []
        
class data():
    def __init__(self, Paras, Audio, tmin = -1, tmax = -1):
        #This is called at the start -- can make some things like blank arrays for the nominals and the like. Can also do the FTs here etc (just once)
        
        #Chnage the length of the audio as appropriate
        
        if tmin > 0.0:
            cut_min_int = int(tmin*Audio.fs)
        else:
            cut_min_int = 0
        if tmax > 0.0:
            cut_max_int = int(tmax*Audio.fs)
        else:
            cut_max_int = -1
        
        Audio.signal_trim = Audio.signal[cut_min_int:cut_max_int]
            
        self.nominals = np.round(nominal_freqs*Paras.fcut_length).astype('int')

        self.initial_profile = np.identity(Paras.nbells)     #Initial frequencies for the bells -- these are just the nominals
     
        self.ts, self.transform = self.do_fourier_transform(Paras, Audio)
     
        self.transform_derivative = self.find_transform_derivatives()
        
        print('Fourier transformed in range', cut_min_int/Audio.fs, cut_max_int/Audio.fs)
        
        self.test_frequencies = self.nominals    #This is the case initially
        self.frequency_profile = np.identity(Paras.nbells)   #Each bell corresponds to its nominal frequency alone -- this will later be updated.
        

    def do_fourier_transform(self, Paras, Audio):
        
        full_transform = []; ts = []
        
        Paras.tmax = len(Audio.signal_trim)/Audio.fs
        
        t = Paras.fcut_length/2   #Initial time (halfway through each transform)
        
        while t < Paras.tmax - Paras.fcut_length/2:
            cut_start  = int(t*Audio.fs - Paras.fcut_int/2)
            cut_end    = int(t*Audio.fs + Paras.fcut_int/2)
            
            signal_cut = Audio.signal_trim[cut_start:cut_end]
            
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
    
def do_reinforcement(Paras, Audio):
    
    Data = data(Paras, Audio, tmin = 0.0, tmax = Paras.reinforce_tmax) #This class contains all the important stuff, with outputs and things
    
    #Find strike probabilities from the nominals
    Data.strike_probabilities = find_strike_probabilities(Paras, Data, Audio, init = True, final = False)
    #Find the first strikes based on these probabilities. Hopefully some kind of nice pattern to the treble at least... 
    
    Data.strikes, Data.strike_certs = find_first_strikes(Paras, Data, Audio)
    Paras.stop_flag = False
    
    for count in range(n_reinforces):
            
        if True:
            #Find the probabilities that each frequency is useful. Also plots frequency profile of each bell, hopefully.
            print('Doing frequency analysis,  iteration number', count + 1, 'of', n_reinforces)
            
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
        threshold = 0.1   #Need changes to be at least this good... Need to improve on this really.
        for row in range(len(strikes[0])):
            allcerts.append(np.min(strike_certs[:,row]))
        if len(allcerts) > Paras.nreinforce_rows:
            threshold = max(threshold, sorted(allcerts, reverse = True)[Paras.nreinforce_rows])    
        for row in range(len(strikes[0])):
            if np.min(strike_certs[:,row]) > threshold:
                best_strikes.append(strikes[:,row])
                best_certs.append(strike_certs[:,row])
        print('Using', len(best_strikes), 'rows, minimum confidence:', np.min(best_certs))
        Data.strikes, Data.strike_certs = np.array(best_strikes).T, np.array(best_certs).T
        
        count += 1
        
    return
    
def find_final_strikes(Paras, Audio):
    
    
     #Create new data files in turn -- will be more effeicient ways but meh...
     tmin = 0; tmax = Paras.overall_tcut
     allstrikes = []; allcerts = []
     Paras.allcadences = []
     Paras.stop_flag = False
     while not Paras.stop_flag:
         
         if tmax >= overall_tmax - 1.0:  #Last one
             Paras.stop_flag = True
             
         Data = data(Paras, Audio, tmin = tmin, tmax = tmax) #This class contains all the important stuff, with outputs and things
         
         Data.test_frequencies = np.load('freqs.npy')
         Data.frequency_profile = np.load('freqprobs.npy')
         
         Data.strike_probabilities = find_strike_probabilities(Paras, Data, Audio, init = False, final = True)
         
         np.save('probs.npy', Data.strike_probabilities)
             
         Data.strike_probabilities = np.load('probs.npy')
                  
         if len(allstrikes) == 0:  #Look for changes after this time
             Data.first_change_limit = Paras.first_change_limit 
             Data.handstroke_first = Paras.handstroke_first
         else:
             if len(allstrikes)%2 == 0:
                 Data.handstroke_first = Paras.handstroke_first
             else:
                 Data.handstroke_first = not(Paras.handstroke_first)
             Data.first_change_limit = np.array(allstrikes[-1][:]) - int(tmin/Paras.dt) - 50   
             Data.last_change = np.array(allstrikes[-1]) - int(tmin/Paras.dt)
             Data.cadence_ref = Paras.cadence_ref

         Data.strikes, Data.strike_certs = find_strike_times_rounds(Paras, Data, Audio, final = True) #Finds strike times in integer space
                           
         if len(allstrikes) == 0:
             for row in range(0,len(Data.strikes[0])):
                 allstrikes.append((Data.strikes[:,row] + int(tmin/Paras.dt)).tolist())
                 allcerts.append(Data.strike_certs[:,row].tolist())
                 Paras.allcadences.append((np.max(allstrikes[-1]) - np.min(allstrikes[-1]))/(Paras.nbells-1))
         else:
             for row in range(0,len(Data.strikes[0])):
                 allstrikes.append((Data.strikes[:,row] + int(tmin/Paras.dt)).tolist())
                 allcerts.append(Data.strike_certs[:,row].tolist())
                 Paras.allcadences.append((np.max(allstrikes[-1]) - np.min(allstrikes[-1]))/(Paras.nbells-1))
         tmin = min(allstrikes[-1])*Paras.dt - 5.0
         tmax = min(tmin + Paras.overall_tcut, Paras.overall_tmax)
             
         #Update global class things
         Paras.first_change_limit = np.array(allstrikes[-1]) - int(tmin/Paras.dt) + 20
         nrows_count = int(min(len(Paras.allcadences), 20))
         Paras.cadence_ref = np.mean(Paras.allcadences[-nrows_count:])
         Paras.allstrikes = np.array(allstrikes)
         
     return np.array(allstrikes).T, np.array(allcerts).T
     
       

tower_list = ['Nics', 'Stockton', 'Brancepeth', 'Leeds']

tower_number = 3

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

if tower_number == 3:
    fname = 'audio/leeds2.wav'
    nominal_freqs = np.array([1554,1387,1307,1163,1037,976,872,776,692.5,653,581.5,518])

#Input parameters which may need to be changed for given audio
overall_tmin = 0.0
overall_tmax = 800.0    #Max and min values for the audio signal (just trims overall and the data is then gone)

rounds_tmax = 90.0      #Maximum seconds of rounds
reinforce_tmax = 90.0   #Maxmum time to use reinforcement data (should never actually get this close)

overall_tcut = 60.0

n_reinforces = 10   #Number of times the frequencies should be reinforced

#Import the data
Audio = audio_data(fname)

print('Imported audio length:', len(Audio.signal)/Audio.fs, 'seconds')

overall_tmax = min(overall_tmax, len(Audio.signal)/Audio.fs)
#Establish parameters, some of which are hard coded into the class
Paras = parameters(Audio, nominal_freqs, overall_tmin, overall_tmax, rounds_tmax, reinforce_tmax, overall_tcut)

print('Trimmed audio length:', len(Audio.signal)/Audio.fs, 'seconds')
print('Running assuming', Paras.nbells, 'bells')

do_reinforcement(Paras, Audio)


print('Frequency reinforcement complete, finding strike times throughout...')

Paras.allstrikes, Paras.allcerts = find_final_strikes(Paras, Audio)
    
plot_strikes(Paras)
save_strikes(Paras, tower_list[tower_number])

 
    
    
    
    
    
    
    
    
