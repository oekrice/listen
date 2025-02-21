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

plt.style.use('ggplot')

def normalise(nbits, raw_input):
    #Normalises the string to the number of bits
    return raw_input/(2**(nbits-1))
    
def transform(fs, norm_cut):
    #Produce the fourier transform of the input data
    trans1 = abs(fft(norm_cut)[:len(norm_cut)//2])
    return 0.5*trans1*fs/len(norm_cut)


def find_first_strikes(fs,norm, dt, cut_length, nominal_freqs):
    
    #Takes normalised wave vector, and does some fourier things
    
    cut_start = 0; cut_end = int(cut_length*fs)
    
    nbells = len(nominal_freqs)

    freq_ints = np.array((cut_end - cut_start)*1.0*nominal_freqs/fs).astype('int') + 1   #Integer values for the correct frequencies. One hopes.

    count = 0
    ts = []
    allfreqs = []
    
    rounds_start = 1.0
    rounds_end = 9.5
    nrounds = 4
    
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
    plot_cut = np.array(allfreqs)[:,:300]
    freq_scale = np.arange(plot_cut.shape[1])
    time_scale = ts[:plot_cut.shape[0]]
    
    #Find logs of what's happening within those ranges
                
            
    low_filter = int(freq_ints[-1]*1)   #These appear to work well...
    high_filter = freq_ints[0]*2
    alpha = 5.0
    print(freq_ints)
    
    ts = np.array(ts)
    
    allfreqs_smooth = gaussian_filter1d(allfreqs, int(0.05/dt), axis = 0)
    diffs = np.zeros(allfreqs_smooth.shape)
    diffs[1:,:] = allfreqs_smooth[1:,:] - allfreqs_smooth[:-1,:] 
    
    diffs[diffs < 0.0] = 0.0
     
    diffsums = np.zeros(len(ts))
    for i, t in enumerate(ts):
        skew_values = np.linspace(1.0,2.0,high_filter-low_filter)
        diff_slice = diffs[i,low_filter:high_filter]
        diff_slice[diff_slice < 0.0] = 0.0
        diffsum = np.sum(diff_slice**2*skew_values**alpha)/len(diff_slice)
        diffsums[i] = diffsum
        
    diffsums = diffsums/max(diffsums)
    #diffsums = gaussian_filter1d(diffsums, int(0.01/dt))
        
        
    peaks, _ = find_peaks(diffsums)

    peaks = peaks[peaks > int(rounds_start/dt)]
    peaks = peaks[peaks < int(rounds_end/dt)]
    
    prominences = peak_prominences(diffsums, peaks)[0]

    peaks = np.array([val for _, val in sorted(zip(prominences, peaks), reverse = True)]).astype('int')
    prominences = sorted(prominences, reverse = True)
    
    peaks = peaks[:nbells*nrounds]
    prominences = prominences[:nbells*nrounds]

    prom_sort = np.array(sorted(peaks[:nbells*nrounds]))*dt

    print('Initial peaks', prom_sort)
    
    
    peaks = np.array(sorted(peaks))
    
    plt.plot(ts, diffsums/max(diffsums))
    #plt.plot(ts, energies/max(energies))
    
    peakdiffs = np.array(peaks[1:] - peaks[:-1])*dt
    for k in range(nbells*nrounds):
        plt.scatter(ts[peaks[k]],0.0)
        
    plt.title('Finding initial rounds')
    plt.show()

    print('Attempted to find ', nrounds, ' rounds with ', nrounds*nbells, ' strikes')
    print('Min, max, avg time between strikes:', min(peakdiffs), max(peakdiffs), np.mean(peakdiffs))

    #Plotting colormesh
    for i in range(nbells):
        plt.plot([freq_ints[i], freq_ints[i]], [0.0,10.0], c = 'red', linestyle = 'dotted')
    plt.pcolormesh(freq_scale, time_scale, plot_cut)
    
    for k in range(len(peaks)):
        plt.plot([0,2000], [peaks[k]*dt,peaks[k]*dt], c = 'red')
        
    plt.xlim(0.0,freq_scale[-1])
    plt.ylim(min(peaks)*dt - 0.5, max(peaks)*dt + 0.5)

    tbefore = 0.1  #These could theoretically be optimised
    tafter = 0.1
    
    for k in range(len(peaks)):
        if k%6 == 0:
              plt.plot([0,2000], [peaks[k]*dt-tbefore,peaks[k]*dt-tbefore], c = 'green')
              plt.plot([0,2000], [peaks[k]*dt+tafter,peaks[k]*dt+tafter], c = 'green')
         
    plt.gca().invert_yaxis()
    plt.show()

    init_strikes = np.zeros((nbells, nrounds))
    
    for bell in range(nbells):
        init_strikes[bell,:] = peaks[bell::nbells]
    
    print('Init strikes', init_strikes)
    
    return init_strikes
    
    
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
    
def find_strike_probs(fs, norm, dt, cut_length, best_freqs, allprobs, nominal_freqs, doplots = False):
    #Find times of each bell striking, with some confidence
    
    #Cut_length is the length of the Fourier transform. CENTRE the time around this
    nbells = len(allprobs[0])
    count = 0
    allfreqs = []; ts = []
    
    tmax = len(norm)/fs
    
    print('tmax', tmax)
    
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
    
    nominal_ints = np.array((cut_end - cut_start)*1.0*nominal_freqs/fs).astype('int') + 1   #Integer values for the correct frequencies. One hopes.

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
        threshold = np.percentile(diffsum,90)  #CAN change this
        
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
        
    plt.xlim(30.0,40.0)
    plt.show()
            
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
    
def find_row_times(fs, dt, cut_length, strike_probs):
    
    
    
    
    return allstrikes
    
    

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

#SET THINGS UP
    

cut_length= 0.1 #Time for each cut
#freqs_ref = np.array([1899,1692,1582,1411,1252,1179,1046,930,828,780,693,617])
#nominal_freqs = np.array([1031,918,857,757,676]) #MEANWOOD
nominal_freqs = np.array([1439.,1289.5,1148.5,1075.,962.,861.])  #ST NICS

#fs, data = wavfile.read('audio/meanwood_all.wav')
fs, data = wavfile.read('audio/stedman_nics.wav')

print('Audio length', len(data)/fs)
tmax = 10.5
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

strikes = find_first_strikes(fs, norm[:cutmax], dt, cut_length, nominal_freqs)

count = 0

tmax = len(data)/fs

while count < 1:
    
    best_freqs, allprobs = frequency_analysis(fs, norm[:cutmax], dt, cut_length, nominal_freqs, strikes[:,:2])
    

    strike_probabilities = find_strike_probs(fs, norm[:int(tmax*fs)], dt, cut_length, best_freqs, allprobs, nominal_freqs)
    np.save('probs.npy', strike_probabilities)
    
    strike_probabilities = np.load('probs.npy')
    strikes = find_strike_times(fs, dt, cut_length, strike_probabilities) #Finds strike times in integer space
        
    np.save('strikes.npy', np.array(strikes))
    count += 1
   
strikes = np.load('strikes.npy')
    
plot_strikes(strikes, nrows = -1)
 
'''
if True:
    #Using crude initial analysis, find bell frequencies

    #Run through some dt and cut lengths to see things

    #Then do Fourier analysis on the whole thing
    all_strikes, all_louds, all_confidences = find_strike_times(fs, norm[:], dt, cut_length, bell_frequencies, first_strikes, nominal_freqs)
    
    np.save('allstrikes.npy', all_strikes)
    np.save('allconfs.npy', all_confidences)
else:
    all_strikes = np.load('allstrikes.npy')
    all_confidences = np.load('allconfs.npy')
    

plot_strikes(all_strikes, all_louds, all_confidences, nrows = -1)

print('Confidence', np.sum(all_confidences)/np.size(all_confidences))
  
for nchanges in [2,2,2]:
        
    bell_frequencies = reinforce_frequencies(fs, norm, dt, cut_length, all_strikes, all_confidences, nominal_freqs, nchanges = nchanges)
    
    print('New frequencies', bell_frequencies)
    
    all_strikes,  all_louds, all_confidences = find_strike_times(fs, norm, dt, cut_length, bell_frequencies, first_strikes, nominal_freqs)
    
    plot_strikes(all_strikes, all_louds, all_confidences, nrows = -1)
    print('Confidence', np.sum(all_confidences)/np.size(all_confidences))
    
'''
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
