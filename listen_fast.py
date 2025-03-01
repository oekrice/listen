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

from frequency_functions import normalise, find_strike_probabilities, find_first_strikes, do_frequency_analysis

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
    
def find_strike_times_rounds(fs, dt, cut_length, strike_probs, first_strike_time):
    #Go through the rounds in turn instead of doing it bellwise
    #Allows for nicer plotting and stops mistakely hearing louder bells. Hopefully.
    
    nbells = len(strike_probs)
    
    bell = nbells - 1 #Assume the tenor is easiest to spot
    
    probs = strike_probs[bell]  
    probs = gaussian_filter1d(probs, 10)
    
    peaks, _ = find_peaks(probs)
    prominences = peak_prominences(probs, peaks)[0]

    peaks = np.array([val for _, val in sorted(zip(prominences, peaks), reverse = True)]).astype('int')
    prominences = sorted(prominences, reverse = True)
    
    #Take those above a certain threshold... 
    probs_smooth = gaussian_filter1d(probs, int(1.0/dt))
    
    peaks = np.array(sorted(peaks[prominences > 2.0*probs_smooth[peaks]]))
    
    peakdiffs = peaks[2::2] - peaks[:-2:2]
        
    print('Median time between same stroke:', np.percentile(peakdiffs, 50))
    
    avg_cadence = np.percentile(peakdiffs, 50)/(2*nbells + 1) #Avg distance in between bells
        
    #Determine if first change is backstroke or handstroke (bit of a guess, but is usually fine)
    #Check first few peakdiffs are fine
    
    ndiffs = min(len(peakdiffs), 4)

    diff1s = peaks[1:2*ndiffs-1:2] - peaks[0:2*ndiffs-2:2]
    diff2s = peaks[2:2*ndiffs:2] - peaks[1:2*ndiffs-1:2]

    print('strikes', peaks)
    print('diffs', diff1s, diff2s)

    error = (np.max(peakdiffs[:ndiffs]) - np.min(peakdiffs[:ndiffs]))/np.mean(peakdiffs[:ndiffs])
    print('Rhythm', error)
    if error > 0.1:
        raise Exception('Not sure which stroke this starts on... Change some things.')
    
    diff1s = peaks[1:2*ndiffs-1:2] - peaks[0:2*ndiffs-2:2]
    diff2s = peaks[2:2*ndiffs:2] - peaks[1:2*ndiffs-1:2]
    start_handstroke = False

    if np.mean(diff1s) < np.mean(diff2s):
        start_handstroke = True
        
    start_ref = peaks[0]
    allstrikes = []; allconfs = []
    minlength = 1e6    
    
    bellstrikes = []; bellconfs = []; allcadences = []
            
    start = 0; end = 0; taim = 0
    alpha =  2.0  #Sharp cutoff?
    nextend = 0
    tcut = int(avg_cadence*1.0)

    strike_probs = gaussian_filter1d(strike_probs, 3, axis = 1)
    
    #Obtain adjusted probs
    strike_probs_adjust = np.zeros(strike_probs.shape)
    strike_probs_adjust = strike_probs[:, :]**3/(np.sum(strike_probs[:,:], axis = 0) + 1e-6)**2

    plot_max = 30   #Do some plotting
    if True:
        fig, axs = plt.subplots(3,4)
        tplots = np.arange(len(strike_probs[bell]))*dt
        for bell in range(nbells):
            ax = axs[bell//4, bell%4]
            ax.plot(tplots, strike_probs_adjust[bell,:])
            ax.set_title(bell+1)
            ax.set_xlim(0,plot_max)
        plt.tight_layout()
        plt.show()
        
    allpeaks = []; allbigs = []; allsigs = []
    for bell in range(nbells):
        
        probs = strike_probs_adjust[bell]  

        probs_smooth = 0.5*gaussian_filter1d(probs, int(1.0/dt))

        peaks, _ = find_peaks(probs)
        
        peaks = peaks[peaks > first_strike_time + avg_cadence*(bell-1)]
        
        prominences = peak_prominences(probs, peaks)[0]
        
        bigpeaks = peaks[prominences > 3.0*probs_smooth[peaks]]  #For getting first strikes, need to mbe more significant
        peaks = peaks[prominences > probs_smooth[peaks]]

        sigs = peak_prominences(probs, peaks)[0]#/probs_smooth[peaks]
        
        sigs = sigs/np.max(sigs)
        
        allpeaks.append(peaks); allbigs.append(bigpeaks); allsigs.append(sigs)

    #Find all peaks to begin with
    #Run through each set of rounds         
    
    handstroke = start_handstroke
    taims = np.zeros(nbells)
    next_end = 0
    
    count = 0
    while next_end < np.max(peaks) - int(5.0/dt):
        plotflag = True
        strikes = np.zeros(nbells)
        confs = np.zeros(nbells)
        count += 1
        if len(allstrikes) == 0:  #Establish first strike
            for bell in range(nbells):
                strikes[bell] = allbigs[bell][0]
                confs[bell] = 1.0
                if bell == nbells - 1: 
                    #Check if this matches first tenor...
                    if abs(strikes[bell] -  start_ref) > 10:
                        handstroke = not(handstroke)
        else:  #Find options in the correct range

            for bell in range(nbells):
                peaks = allpeaks[bell]
                sigs = allsigs[bell]
                peaks_range = peaks[(peaks > start)*(peaks < end)]
                sigs_range = sigs[(peaks > start)*(peaks < end)]
                
                start_bell = taims[bell] - int(3.5*avg_cadence)  #Aim within the change
                end_bell = taims[bell] + int(3.5*avg_cadence)
                #Check physically possible...
                start_bell = max(start_bell, allstrikes[-1][bell] + int(3.0*avg_cadence))
                
                sigs_range = sigs_range[(peaks_range > start_bell)*(peaks_range < end_bell)]
                peaks_range = peaks_range[(peaks_range > start_bell)*(peaks_range < end_bell)]

                if len(peaks_range) == 1:   #Only one time that it could reasonably be
                    strikes[bell] = peaks_range[0]
                    tvalue = 1.0/(abs(peaks_range[0] - taims[bell])/tcut + 1)**alpha
                    confs[bell]  = 1.0
                    
                elif len(peaks_range) > 1:
                                          
                    scores = []
                    for k in range(len(peaks_range)):  #Many options...
                        tvalue = 1.0/(abs(peaks_range[k] - taims[bell])/tcut + 1)**alpha
                        yvalue = sigs_range[k]/np.max(sigs_range)
                        scores.append(tvalue*yvalue**2.0)
                        
                    kbest = scores.index(max(scores))
                    
                    strikes[bell] = peaks_range[kbest]
                    confs[bell] = scores[kbest]**2/np.sum(scores)**2

                    cert = max(scores)/np.sum(scores)
                    if confs[bell] < 0.6:
                        plotflag = True
                        print(bell + 1, 'unsure', confs[bell], cert, peaks_range*dt, scores, taims[bell])
                        
                else:
                    #Pick best peak in the change? Seems to work when things are terrible
                    
                    peaks = allpeaks[bell]
                    sigs = allsigs[bell]
                    peaks_range = peaks[(peaks > start)*(peaks < end)]
                    sigs_range = sigs[(peaks > start)*(peaks < end)]
                    
                    start_bell = max(start_bell, allstrikes[-1][bell] + int(3.0*avg_cadence))
                    end_bell = end
                    
                    sigs_range = sigs_range[(peaks_range > start_bell)*(peaks_range < end_bell)]
                    peaks_range = peaks_range[(peaks_range > start_bell)*(peaks_range < end_bell)]

                    plotflag = True

                    scores = []
                    for k in range(len(peaks_range)):  #Many options...
                        tvalue = 1.0/(abs(peaks_range[k] - taims[bell])/tcut + 1)**alpha
                        yvalue = sigs_range[k]/np.max(sigs_range)
                        scores.append(tvalue*yvalue**2.0)
                        
                    if len(scores) > 0:
                        kbest = scores.index(max(scores))
                        
                        strikes[bell] = peaks_range[kbest]
                        confs[bell] = 0.0
    
                        cert = max(scores)/np.sum(scores)
                        if confs[bell] < 0.6:
                            plotflag = True
                            print(bell + 1, 'unsure', confs[bell], cert, peaks_range*dt, scores, taims[bell])
                    else:
                        #Pick average point in the change
                        print(bell + 1, 'complete guess', confs[bell], cert, peaks_range*dt, scores, taims[bell])

                        strikes[bell] = int(0.5*(start + end))
                        confs[bell] = 0.0
                        
        allstrikes.append(strikes)
        allconfs.append(confs)
        
        yvalues = np.arange(nbells) + 1
        
        order = np.array([val for _, val in sorted(zip(strikes, yvalues), reverse = False)])
        conf_order = np.array([val for _, val in sorted(zip(strikes, confs), reverse = False)])
        print(order, np.array(sorted(strikes))*dt)
        
        if plotflag:  #Plot the probs and things
            plotstart = int(min(strikes)); plotend = int(max(strikes))
            ts = np.arange(plotstart - int(1.0/dt),plotend + int(1.0/dt))*dt
            for bell in range(nbells):
                plt.plot(ts, strike_probs[bell,plotstart - int(1.0/dt):plotend + int(1.0/dt)], c = cmap(bell/(nbells-1)), linestyle = 'dotted')
                plt.plot(ts, strike_probs_adjust[bell,plotstart - int(1.0/dt):plotend + int(1.0/dt)], label = bell + 1, c = cmap(bell/(nbells-1)))
            plt.scatter(start*dt, - 0.1, c = 'green')
            plt.scatter(end*dt,  - 0.1, c = 'red')
            plt.scatter(taims*dt,  - 0.2*np.ones(nbells), c = cmap(np.linspace(0,1,nbells)), marker = 's')
            plt.scatter(strikes*dt,  - 0.3*np.ones(nbells), c = cmap(np.linspace(0,1,nbells)), marker = '*')
            plt.legend()
            
            plt.show()
            #input()
        #Determine likely location of the next change END
        #Need to be resilient to method mistakes etc... 
        #Log the current avg. bell cadences
        allcadences.append((max(strikes) - min(strikes))/(nbells - 1))     

        nrows_count = int(min(len(allcadences), 20))
        cadence_ref = np.mean(allcadences[-nrows_count:])
        
        change_start = np.mean(strikes) - cadence_ref*((nbells - 1)/2)
        change_end = np.mean(strikes) + cadence_ref*((nbells - 1)/2)
        
        rats = (strikes - change_start)/(change_end - change_start)
                
        
        if handstroke:
            taims  = np.array(allstrikes[-1]) + int(nbells*avg_cadence)
            next_start = change_start + int(nbells*cadence_ref)
            next_end = change_end + int(nbells*cadence_ref)
        else:
            taims  = np.array(allstrikes[-1]) + int((nbells + 1)*avg_cadence)
            next_start = change_start + int((nbells+1)*cadence_ref)
            next_end = change_end + int((nbells+1)*cadence_ref)


        taims = next_start + (next_end - next_start)*rats
                   
        handstroke = not(handstroke)
        
        yvalues = np.arange(nbells) + 1
        order = np.array([val for _, val in sorted(zip(strikes, yvalues), reverse = False)])
        
        start = next_start - 1.0*int(avg_cadence)
        end  =  next_end   + 3.0*int(avg_cadence)

    print('Overall confidence', np.sum(allconfs)/np.size(allconfs))
    return np.array(allstrikes).T, np.array(allconfs).T
        
def plot_strikes(all_strikes, strike_certs, nrows = -1):
    #Plots the things
    fig = plt.figure(figsize = (10,7))
    nbells = len(all_strikes)
    if nrows < 0:
        nrows = len(all_strikes[0])
    yvalues = np.arange(nbells) + 1
    
    #for bell in range(nbells):
    #    plt.scatter(all_strikes[bell], yvalues[bell]*np.ones(len(all_strikes[bell])),s=all_confidences[bell]*100)
    
    for row in range(nrows):
        plt.plot(all_strikes[:,row],yvalues)
        order = np.array([val for _, val in sorted(zip(all_strikes[:,row], yvalues), reverse = False)])
        confs = np.array([val for _, val in sorted(zip(all_strikes[:,row], strike_certs[:,row]), reverse = False)])
        print('Strikes', row, order, np.array(sorted(all_strikes[:,row]))*dt)#, confs)
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

def save_strikes(strikes, dt, tower):
    #Saves as a pandas thingummy like the strikeometer does
    allstrikes = []
    allbells = []
    nbells = len(strikes)
    yvalues = np.arange(nbells) + 1

    for row in range(len(strikes[0])):
        order = np.array([val for _, val in sorted(zip(strikes[:,row], yvalues), reverse = False)])
        allstrikes = allstrikes + sorted((strikes[:,row]).tolist())
        allbells = allbells + order.tolist()
       
    allstrikes = 1000*np.array(allstrikes)*dt
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
    def __init__(self, Audio, nominal_freqs, overall_tmin, overall_tmax, rounds_tmax):
                
        self.dt = 0.01
        self.fcut_length = 0.125  #Length of each transform slice (in seconds)
        self.transform_smoothing = 0.05 #Transform smoothing for the initial derivatives of the transform (in seconds)
        self.frequency_range = 1    #Range over which to include frequencies in a sweep (as in, 300 will count between 300-range:frequency+range+1 etc.)
        self.derivative_smoothing = 5  #Smoothing for the derivative (in INTEGER time lumps -- could change if necessary...)
        self.smooth_time = 2.0    #Smoothing over which to apply change-long changes (in seconds)
        self.max_change_time = 3.0 #How long could a single change reasonably be
        self.nrounds_max = 8
        self.strike_smoothing = 3 #How much to smooth the input probability function
        self.strike_tcut = 1.0 #How many times the average cadence to cut off
        self.strike_alpha = 2  #How much to care about timing
        self.strike_gamma = 2  #How much to care about prominence
        self.strike_gamma_init = 1.5  #How much to care about prominence for the initial rounds
        self.freq_tcut = 0.5 #How many times the average cadence to cut off for FREQUENCIES (should be identical strikes really)
        self.freq_smoothing = 5 #How much to smooth the data when looking for frequencies (as an INTEGER)
        self.beta = 2   #How much to care whether strikes are certain when looking at frequencies
        self.freq_filter = 2#How much to filter the frequency profiles (in INT)
        
        self.rounds_tmax = rounds_tmax
        Audio.signal = Audio.signal[int(overall_tmin*Audio.fs):int(overall_tmax*Audio.fs)]
        self.nbells = len(nominal_freqs)
        self.fcut_int = 2*int(self.fcut_length*Audio.fs/2)  #Length of this cut (must be even for symmetry purposes)
        self.tmax =  len(Audio.signal)/Audio.fs
        
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
    nominal_freqs = np.array([1892,1679,1582,1407,1252,1179,1046,930,828,780,693,617])

if tower_number == 2:    
    #fs, data = wavfile.read('audio/brancepeth.wav')
    fname = 'audio/Brancepeth_cambridge.wav'
    nominal_freqs = np.array([1230,1099,977,924,821.5,733])

#Input parameters which may need to be changed for given audio
overall_tmin = 0.0
overall_tmax = 60.0    #Max and min values for the audio signal (just trims overall and the data is then gone)

rounds_tmax = 60.0      #Maximum seconds of rounds

n_reinforces = 2   #Number of times the frequencies should be reinforced

#Import the data
Audio = audio_data(fname)

print('Imported audio length:', len(Audio.signal)/Audio.fs, 'seconds')

#Establish parameters, some of which are hard coded into the class
Paras = parameters(Audio, nominal_freqs, overall_tmin, overall_tmax, rounds_tmax)

print('Trimmed audio length:', len(Audio.signal)/Audio.fs, 'seconds')
print('Running assuming', Paras.nbells, 'bells')

Data = data(Paras, Audio) #This class contains all the important stuff, with outputs and things

#Find strike probabilities from the nominals
Data.strike_probabilities = find_strike_probabilities(Paras, Data, Audio, init=True)
#Find the first strikes based on these probabilities. Hopefully some kind of nice pattern to the treble at least... 

Data.strikes, Data.strike_certs = find_first_strikes(Paras, Data, Audio)

for count in range(n_reinforces):
        
    #Find the probabilities that each frequency is useful. Also plots frequency profile of each bell, hopefully.
    print('Doing frequency analysis,  iteration number', count)
    
    allfreqs, freqprobs = do_frequency_analysis(Paras, Data, Audio)  
    
    stop
    
    np.save('freqs.npy', allfreqs)
    np.save('freqprobs.npy', freqprobs)

    freqprobs = np.load('freqprobs.npy')
    allfreqs = np.load('freqs.npy')
    
    cutmax = int(tmax*fs)

    print('Finding strike probabilities...')
    
    strike_probabilities = find_strike_probs(Paras, Data, Audio, init = False)
    np.save('probs.npy', strike_probabilities)

    strike_probabilities = np.load('probs.npy')
    #strikes, strike_certs = find_strike_times(fs, dt, cut_length, strike_probabilities, first_strike_time) #Finds strike times in integer space
    #strikes, strike_certs = find_strike_times_rounds(fs, dt, cut_length, strike_probabilities, first_strike_time) #Finds strike times in integer space
    
    strikes, strike_certs = find_best_strikes(fs, dt, cut_length, strike_probabilities, strikesmax = 30) #Finds strike times in integer space
    
    count += 1

tmax = len(data)/fs
cutmax = int(tmax*fs)
freqprobs = np.load('freqprobs.npy')
allfreqs = np.load('freqs.npy')

strike_probabilities = find_strike_probs(fs, norm[:cutmax], dt, cut_length, allfreqs, freqprobs, nominal_freqs, init = False)
np.save('probs.npy', strike_probabilities)

strike_probabilities = np.load('probs.npy')

strikes, strike_certs = find_strike_times_rounds(fs, dt, cut_length, strike_probabilities, first_strike_time) #Finds strike times in integer space
maxrows = 0; maxtime = 0.0
for row in range(len(strikes[0])):
    if np.max(strikes[:,row] - np.min(strikes[:,row])) < 3.0/dt:
        maxrows = max(4, row)
        maxtime = np.max(strikes[:,row])*dt
    else:
        break
    
print('Number of probably correct rows: ', maxrows)
strikes = strikes[:, :maxrows]
strike_certs = strike_certs[:, :maxrows]
unit_test(strikes,dt)

#plot_strikes(strikes,strike_certs,  nrows = -1)
save_strikes(strikes, dt, tower_list[tower_number])

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
