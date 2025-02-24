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

from plot_tools import plotamps, plot_log, plot_freq

plt.style.use('default')

def normalise(nbits, raw_input):
    #Normalises the string to the number of bits
    return raw_input/(2**(nbits-1))
    
def transform(fs, norm_cut):
    #Produce the fourier transform of the input data
    trans1 = abs(fft(norm_cut)[:len(norm_cut)//2])
    return 0.5*trans1*fs/len(norm_cut)
    
def frequency_analysis(fs,norm, dt, cut_length, nominal_freqs, strikes):
    #Now takes existing strikes data to do this (to make reinforcing easier)
    
    #__________________________________________________
    nrows = len(strikes[0])
    
    print('Frequency testing on', nrows, 'rows')
    
    cut_start = 0; cut_end = int(cut_length*fs)
    
    nbells = len(nominal_freqs)

    freq_ints = np.array((cut_end - cut_start)*1.0*nominal_freqs/fs).astype('int') + 1   #Integer values for the correct frequencies. One hopes.

    count = 0
    ts = []
    allfreqs = []
        
    while cut_end < len(norm):
        if count%50 == -1:
            print('Analysing, t = ', cut_start/fs)
        trans = transform(fs, norm[cut_start:cut_end])
        
        cut_start = cut_start + int(dt*fs)
        cut_end = cut_start + int(cut_length*fs)
    
        count += 1
        
        ts.append((cut_start + cut_end)/(2*fs))
        
        allfreqs.append(trans)
            
    allfreqs = np.array(allfreqs)
    
    min_freq_int= int(freq_ints[-1]*0.9)  
    max_freq_int = int(freq_ints[0]*4)
    
    #Run through and find the frequencies most prominent at these times? Must be a few of them. Doesn't line up well with nominals...
    
    ts = np.array(ts)
    
    allfreqs_smooth = gaussian_filter1d(allfreqs, int(0.05/dt), axis = 0)
    diffs = np.zeros(allfreqs_smooth.shape)
    diffs[1:,:] = allfreqs_smooth[1:,:] - allfreqs_smooth[:-1,:] 
    
    diffs[diffs < 0.0] = 0.0
     
    #Run through FREQUENCIES and see which match up with increases near the bell time?
    plt.pcolormesh(diffs[:,:max_freq_int].T)
    plt.show()
    
    cs = ['blue', 'red', 'green', 'yellow', 'brown', 'pink']
    
    good_freqs = []; freq_peak_array = []; sig_peak_array = []
    
    last_strike_time = int(np.max(strikes) + 0.5/dt)
    
    print('Last strike times', last_strike_time*dt)
    
    for freq_test in range(min_freq_int, max_freq_int, 1):
    #for freq_test in freq_ints:
        #fig = plt.figure()
        freq_range = 2
        diff_slice = diffs[:last_strike_time,freq_test-freq_range:freq_test+freq_range]
        diff_slice[diff_slice < 0.0] = 0.0
        diffsum = np.sum(diff_slice**2,axis = 1 )
        
        diffsum = gaussian_filter1d(diffsum, 5)
        diffpeaks, _ = find_peaks(diffsum)
        
        prominences = peak_prominences(diffsum, diffpeaks)[0]
        
        diffpeaks = np.array([val for _, val in sorted(zip(prominences, diffpeaks), reverse = True)]).astype('int')
        threshold = np.percentile(diffsum,80)
        
        prominences = sorted(prominences, reverse = True)
        
        diffpeaks = diffpeaks[prominences > threshold]
        prominences = np.array(prominences)[prominences > threshold]
        
        sigpeaks = (prominences - threshold)/(max(prominences) - threshold)
        
        #Number of prominences over a theshold below the max
        
        if freq_test == -1:
                
            plt.plot(ts[:len(diffsum)],diffsum/max(diffsum))
        
            for diffpeak in diffpeaks:
                plt.scatter(ts[diffpeak],1.0, color = 'black')
            
            plt.plot([0.0,ts[len(diffsum)]],threshold*np.ones(2)/max(diffsum))
                    
            plt.title((freq_test, np.sum(diffsum)/np.max(diffsum),len(diffpeaks)))
            plt.show()

        if len(diffpeaks) in range(nrows//2, nrows*3 + 3):
        #if True:
            sig_peak_array.append(sigpeaks.tolist())

            freq_peak_array.append(diffpeaks.tolist())
            good_freqs.append(freq_test)
            
                            
    allprobs = []; best_freqs = []
    #Run through frequencies and see which are closest -- inverse square with time? 
    for fi, freq_test in enumerate(good_freqs):
        tcut = int(0.1/dt)
        probs = np.zeros(nbells)

        npeaks = len(freq_peak_array[fi])
        for bell in range(nbells):
            for peak_test in range(npeaks):
                mindist = np.min(np.abs(freq_peak_array[fi][peak_test] - strikes[bell]))
                prop = mindist/tcut 
                probs[bell] += sig_peak_array[fi][peak_test]**2.0/(prop + 1)**2
            probs[bell] = probs[bell]/sum(sig_peak_array[fi])
            
            
        if max(probs) > 0.4:
            allprobs.append(probs)
            best_freqs.append(freq_test)

    #for fi, freq in enumerate(best_freqs):
    #    print(freq, allprobs[fi], np.where(allprobs[fi] == max(allprobs[fi]))[0][0])

    #Plot probabilities somehow
    for fi, freq in enumerate(best_freqs):
        plt.scatter(freq*np.ones(nbells), np.arange(nbells)+1, s = 100*allprobs[fi]**2)
    plt.show()

    return best_freqs, allprobs
    
def find_strike_probs(fs, norm, dt, cut_length, best_freqs, allprobs, nominal_freqs, doplots = False, init = False):
    #Find times of each bell striking, with some confidence
    
    #Cut_length is the length of the Fourier transform. CENTRE the time around this
    nbells = len(allprobs[0])
    count = 0
    allfreqs = []; ts = []
    
    tmax = len(norm)/fs
    
    
    t = cut_length/2
    trans_length = 2*int(fs*cut_length/2)
    
    while t < tmax - cut_length/2:
        cut_start  = int(t*fs - fs*cut_length/2)
        cut_end    = cut_start + trans_length
        
           
        trans = transform(fs, norm[cut_start:cut_end])
            
        count += 1
        
        ts.append(t)        
        allfreqs.append(trans)
        
        t = t + dt
        
    #Make sure that this transform is sorted in EXACTLY the same way that it's done initially.
    #No tbefores etc., just the derivatives.
    
    allfreqs = np.array(allfreqs)    
    allprobs = np.array(allprobs)
    
    ts = np.array(ts)

    allfreqs_smooth = gaussian_filter1d(allfreqs, int(0.05/dt), axis = 0)
    diffs = np.zeros(allfreqs_smooth.shape)
    diffs[1:,:] = allfreqs_smooth[1:,:] - allfreqs_smooth[:-1,:] 
    
    diffs[diffs < 0.0] = 0.0
     
    difflogs = []; all_diffpeaks = []
    #Produce logs of each FREQUENCY, so don't need to loop
    for fi, freq_test in enumerate(best_freqs):
    #for freq_test in freq_ints:
        #fig = plt.figure()
        
        freq_range = 2
        diff_slice = diffs[:,freq_test-freq_range:freq_test+freq_range]
        diff_slice[diff_slice < 0.0] = 0.0
        diffsum = np.sum(diff_slice**2,axis = 1 )
        
        diffsum = gaussian_filter1d(diffsum, 5)

        diffpeaks, _ = find_peaks(diffsum)
        
        prominences = peak_prominences(diffsum, diffpeaks)[0]
        
        diffpeaks = np.array([val for _, val in sorted(zip(prominences, diffpeaks), reverse = True)]).astype('int')
        if not init:
            threshold = np.percentile(diffsum,90)  #CAN change this
        else:
            threshold = np.percentile(diffsum,10)
        prominences = sorted(prominences, reverse = True)
        
        diffpeaks = diffpeaks[prominences > threshold]
        prominences = np.array(prominences)[prominences > threshold]
        
        #sigpeaks = (prominences - threshold)/(max(prominences) - threshold)
        
        #Number of prominences over a theshold below the max
        if False:
            plt.plot(ts,diffsum/max(diffsum))
        
                
            for diffpeak in diffpeaks:
                plt.scatter(ts[diffpeak],1.0, color = 'black')
            
            plt.plot([0.0,ts[-1]],threshold*np.ones(2)/max(diffsum))
                
            plt.title((freq_test, np.sum(diffsum)/np.max(diffsum),len(diffpeaks)))
            #plt.xlim(30,40)
            plt.show()
            
        difflogs.append(diffsum)
        all_diffpeaks.append(diffpeaks)
        
    if not init:
        overall_bell_probs = np.zeros((nbells, len(diffsum)))
    
        for bell in range(nbells):  #the 4 is the most distinctive
            bell_freqs = allprobs[:,bell]
            all_poss = []; all_probs = []
            for fi, freq_test in enumerate(best_freqs) :
                if bell_freqs[fi] > 0.5:
                    good_peaks = all_diffpeaks[fi]
                    good_peaks = all_diffpeaks[fi]
                    for k in range(len(good_peaks)):
                        all_poss.append(good_peaks[k])
                        all_probs.append(bell_freqs[fi])  #Could also add prominence weighting here?
                        
            all_poss = np.array(all_poss)
            all_probs = np.array(all_probs)
    
            for t_int in range(len(diffsum)):
                
                props = np.abs(all_poss - t_int)/(int(0.1/dt))  #Absolute distance of each peak
                propsum = np.sum(all_probs**1.0/(props + 1)**2)
                overall_bell_probs[bell, t_int] = propsum/np.sum(all_probs)
    
            plt.plot(ts, overall_bell_probs[bell])
            
        plt.xlim(0.0,15.0)
        plt.show()
                
        return overall_bell_probs

    else:
        #This is the initial finding. Just take the unadulterated sum.
        
        overall_bell_probs = np.zeros((nbells, len(diffsum)))
    
        for bell in range(nbells):  #the 4 is the most distinctive
            overall_bell_probs[bell] = difflogs[bell]
    
            plt.plot(ts, overall_bell_probs[bell]/max(overall_bell_probs[bell]), label = bell)
            
        plt.legend()
        plt.xlim(0.0,15.0)
        plt.close()
                
        return overall_bell_probs
        
        
def find_strike_times(fs, dt, cut_length, strike_probs):
    #Using the probabilities, figures out when the strikes actually are
    nbells = len(strike_probs)
    
    bell = nbells - 1 #Assume the tenor is easiest to spot
    
    probs = strike_probs[bell]  
    probs = gaussian_filter1d(probs, 2)
    
    peaks, _ = find_peaks(probs)
    prominences = peak_prominences(probs, peaks)[0]

    peaks = np.array([val for _, val in sorted(zip(prominences, peaks), reverse = True)]).astype('int')
    prominences = sorted(prominences, reverse = True)
    
    #Take those above a certain threshold... 
    peaks = np.array(sorted(peaks[prominences > np.percentile(probs,90)]))
    
    peakdiffs = peaks[2::2] - peaks[:-2:2]
    
    print('Median time between same stroke:', np.percentile(peakdiffs, 50))
    
    avg_cadence = np.percentile(peakdiffs, 50)/(2*nbells + 1) #Avg distance in between bells
        
    #Determine if first change is backstroke or handstroke (bit of a guess, but is usually fine)
    start_handstroke = True
    
    if peaks[1] - peaks[0] > peaks[2] - peaks[1]:
        start_handstroke = False
        
    allstrikes = []; allconfs = []
    minlength = 1e6
    for bell in range(nbells):
        
        handstroke = start_handstroke
        #Find peaks for this bell
        probs = strike_probs[bell]  

        probs = gaussian_filter1d(probs, 5)

        peaks, _ = find_peaks(probs)
        prominences = peak_prominences(probs, peaks)[0]

        #Sort appropriately
        peaks = np.array([val for _, val in sorted(zip(prominences, peaks), reverse = True)]).astype('int')
        prominences = sorted(prominences, reverse = True)
        threshold = np.percentile(probs,40)

        peaks = np.array(sorted(peaks[prominences > threshold]))

        prominences = peak_prominences(probs, peaks)[0]

        bellstrikes = []; bellconfs = []; taims = []
                
        start = 0; end = 0; taim = 0
        alpha =  0.5
        nextend = 0
            
        while nextend < np.max(peaks):
            if len(bellstrikes) == 0:  #Establish first strike
                bellstrikes.append(peaks[0])
                bellconfs.append(1.0)
            else:  #Find options in the correct range
                peaks_range = peaks[(peaks > start)*(peaks < end)]
                if len(peaks_range) == 1:
                    bellstrikes.append(peaks_range[0])
                    bellconfs.append(1.0)
                elif len(peaks_range) > 1:
                      
                    proms_range = prominences[(peaks > start)*(peaks < end)]
                    
                    
                    scores = []
                    for k in range(len(peaks_range)):
                        scores.append((proms_range[k]-threshold)/(abs(peaks_range[k] - taim) + 1)**alpha)
                    kbest = scores.index(max(scores))
                    bellstrikes.append(peaks_range[kbest])
                    bellconfs.append(scores[kbest]/np.sum(scores))
                else:
                    plt.scatter(bellstrikes,np.zeros(len(bellstrikes)))
                    plt.plot(probs)
                    plt.xlim(taim-1000,taim+1000)
                    plt.show()

                    
                    raise Exception('No peaks found in the requested range')
    
            if handstroke:
                taim  = bellstrikes[-1] + int(nbells*avg_cadence)
            else:
                taim  = bellstrikes[-1] + int((nbells + 1)*avg_cadence)
            taims.append(taim)
            start = bellstrikes[-1] + 2*int(avg_cadence)
            end  =  taim + (nbells-1)*int(avg_cadence)
            nextend = bellstrikes[-1] + int(avg_cadence*nbells*1.5)  #End of next change
               
            handstroke = not(handstroke)
            
        plt.scatter(peaks, np.zeros(len(peaks)), c= 'green')
        plt.scatter(taims, np.zeros(len(taims)), c= 'red')
        plt.scatter(bellstrikes, np.zeros(len(bellstrikes)), s = 50*np.array(bellconfs), c= 'black')
        plt.plot(probs)
        plt.title(bell)
        plt.xlim(13000,14000)
        plt.show()
        
        allstrikes.append(bellstrikes)
        allconfs.append(bellconfs)
        minlength = min(minlength, len(bellstrikes))
        
    for bell in range(len(allstrikes)):
        allstrikes[bell] = allstrikes[bell][:minlength]
        allconfs[bell] = allconfs[bell][:minlength]
        
    allstrikes = np.array(allstrikes)[:,:minlength]
    print('Overall confidence', np.sum(allconfs)/np.size(allconfs))

    
    return allstrikes 


def find_first_strikes(fs, norm, dt, cut_length, strikeprobs):
    
    #Takes normalised wave vector, and does some fourier things
        
    nrounds = 4   #Maximum rounds
    print('Finding approximate first strike times in rounds...')
    nbells = len(nominal_freqs)

    tenor_probs = strikeprobs[-1]
    
    tenor_peaks, _ = find_peaks(tenor_probs)
    prominences = peak_prominences(tenor_probs, tenor_peaks)[0]

    #Sort appropriately
    tenor_peaks = np.array([val for _, val in sorted(zip(prominences,tenor_peaks), reverse = True)]).astype('int')
    prominences = sorted(prominences, reverse = True)
    
    threshold = np.percentile(tenor_probs,90)

    tenor_peaks = tenor_peaks[prominences > threshold]

    first_strike = np.min(tenor_peaks)
    tenor_strikes = [first_strike]
    print('Finding tenor strikes, first significant at time', first_strike)
    
    start = first_strike
    end = first_strike + int(3.5/dt)
    for r in range(nrounds + 1):
        #Find most probable tenor strikes
        poss = tenor_peaks[(tenor_peaks > start)*(tenor_peaks < end)]  #Possible strikes in range -- pick biggest
        prominences = peak_prominences(tenor_probs, poss)[0]
        poss = np.array([val for _, val in sorted(zip(prominences,poss), reverse = True)]).astype('int')
        tenor_strikes.append(poss[0])
        start = poss[0]
        end = poss[0] + + int(3.5/dt)   

    print(tenor_strikes)

    tenor_strikes = np.array(tenor_strikes)
    #Determine whether this is the start of the ringing or not...
    difftenors = tenor_strikes[1:] - tenor_strikes[:-1]
    init_aims = []
    
    if first_strike > 1.25*np.mean(difftenors):
        if difftenors[1] > difftenors[0]:
            handstroke = True
        else:
            handstroke = False

        print('Audio starting from the start of ringing')
        tenor_strikes = np.concatenate(([tenor_strikes[0] - difftenors[1]], tenor_strikes))
        print(tenor_strikes)
        #Probably some silence beforehand
    else:
        #Probably not...
        print('Audio starting from mid-way through ringing')
        
    if difftenors[1] > difftenors[0]:
        handstroke = True
    else:
        handstroke = False
        
    init_strikes = np.zeros((nbells, nrounds))
    for rounds in range(nrounds):
        #Interpolate the bells smoothly (assuming steady rounds)
        if handstroke:
            belltimes = np.linspace(tenor_strikes[rounds], tenor_strikes[rounds+1], nbells + 1)
        else:
            belltimes = np.linspace(tenor_strikes[rounds], tenor_strikes[rounds+1], nbells + 2)
            
        cadence = np.mean(belltimes[1:] - belltimes[:-1])
        belltimes = belltimes[-nbells:]
        
        handstroke = not(handstroke)
        
        init_aims.append(belltimes)
                
    plt.plot(tenor_probs)
    for r in range(len(init_aims)):
        plt.scatter(init_aims[r], np.zeros(nbells), c = 'red')
    plt.scatter(tenor_strikes, np.zeros(len(tenor_strikes)), c = 'black')
    plt.title('Initial tenor strike detection')
    plt.show()

    print('Attempted to find ', len(init_aims), ' rounds with ', nrounds*nbells, ' strikes')

    #Use these guesses to find the ACTUAL peaks which should be nearby...
    nrounds = len(init_aims)   #In case this is less (the audio is shorter)
    init_aims = np.array(init_aims)
    init_strikes = np.zeros(init_aims.T.shape)
    for bell in range(nbells):
        bell_peaks, _ = find_peaks(strikeprobs[bell])
        for r in range(nrounds):
            aim = init_aims[r, bell]  #aim time
            if bell == nbells - 1:
                init_strikes[bell, r] = init_aims[r, bell]
                continue
            elif bell > 0:
                start = init_aims[r,bell-1] + int(0.05/dt)
            else:
                start = aim - cadence
            end = init_aims[r,bell+1] - int(0.05/dt)

            poss = bell_peaks[(bell_peaks > start)*(bell_peaks < end)]  #Possible strikes in range -- pick biggest
            prominences = peak_prominences(strikeprobs[bell], poss)[0]
            poss = np.array([val for _, val in sorted(zip(prominences,poss), reverse = True)]).astype('int')
            if len(poss) == 0:
                init_strikes[bell, r] = init_aims[r, bell]
            else:
                init_strikes[bell,r] = poss[0]
            
        
        plt.plot(strikeprobs[bell])
        plt.scatter(init_aims[:,bell], np.zeros(nrounds), c= 'red')
        plt.scatter(init_strikes[bell,:], np.zeros(nrounds), c= 'green')
        plt.title(bell)
        plt.show()
    
    
    
    return init_strikes
    
    

def plot_strikes(all_strikes, nrows = -1):
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
        print('Strikes', row, order, np.array(sorted(all_strikes[:,row]))*dt)
        #print(all_strikes[:,row])
        #print('Louds', row, order, sorted(all_louds[:,row]))


    plt.xlim(0.0,30.0)
    plt.gca().invert_yaxis()
    plt.show(fig)

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
    if maxlength*dt > 2.0:
        raise Exception('Change length not correct.')
    #Check last few changes for rounds
    yvalues = np.arange(nbells) + 1
    order = np.array([val for _, val in sorted(zip(all_strikes[:,nrows-1], yvalues), reverse = False)])
    print('Final change:', order)
    print('_______________')
    return 
#SET THINGS UP
    
cut_length= 0.1 #Time for each cut

if True:
    fs, data = wavfile.read('audio/stedman_nics.wav')
    nominal_freqs = np.array([1439.,1289.5,1148.5,1075.,962.,861.])  #ST NICS
    rounds_cut = [1.0, 9.5]
    
else:
    fs, data = wavfile.read('audio/stockton_stedman.wav')
    nominal_freqs = np.array([1892,1679,1582,1407,1252,1179,1046,930,828,780,693,617])

    rounds_cut = [7.4, 12.1]


print('Audio length', len(data)/fs)
tmax = rounds_cut[1] + 5.0
tmin = 0.0#1.5
cutmax = int(tmax*fs)

import1 = np.array(data)[:,0]

ts = np.linspace(0.0, len(import1)/fs, len(import1))

dt = 0.01  #Time between analyses

audio_length = len(import1)


norm = normalise(16, import1)

dt = 0.01
cut_length = 0.1
cut_time = len(data)/fs - 10.0

#strikes = find_first_strikes(fs, norm[:cutmax], dt, cut_length, nominal_freqs)

#Look into doing strike probabilities just from the nominals?
best_freqs = np.round(nominal_freqs*cut_length).astype('int')

allprobs = np.identity(len(best_freqs))

#Find strike probabilities from the nominals
init_strike_probabilities = find_strike_probs(fs, norm[:int(tmax*fs)], dt, cut_length, best_freqs, allprobs, nominal_freqs, init=True)

strikes = find_first_strikes(fs, norm[:int(tmax*fs)], dt, cut_length, init_strike_probabilities)

print(strikes)

count = 0

tmax = len(data)/fs

while count < 1:
    
    best_freqs, allprobs = frequency_analysis(fs, norm[:cutmax], dt, cut_length, nominal_freqs, strikes[:,:1])
    
    strike_probabilities = find_strike_probs(fs, norm[:int(tmax*fs)], dt, cut_length, best_freqs, allprobs, nominal_freqs)
    np.save('probs.npy', strike_probabilities)
    
    strike_probabilities = np.load('probs.npy')
    strikes = find_strike_times(fs, dt, cut_length, strike_probabilities) #Finds strike times in integer space
        
    np.save('strikes.npy', np.array(strikes))
    count += 1
   
strikes = np.load('strikes.npy')
    
plot_strikes(strikes, nrows = -1)
 
unit_test(strikes,dt)



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
