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

from plot_tools import plotamps, plot_log, plot_freq

def normalise(nbits, raw_input):
    #Normalises the string to the number of bits
    return raw_input/(2**(nbits-1))
    
def transform(fs, norm_cut):
    #Produce the fourier transform of the input data
    trans1 = abs(fft(norm_cut)[:len(norm_cut)//2])
    return 0.5*trans1*fs/len(norm_cut)

def initial_analysis(fs,norm, dt, cut_length, nominal_freqs):
    #Takes normalised wave vector, and does some fourier things
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
    plot_cut = np.array(allfreqs)[:,:200]
    freq_scale = np.arange(plot_cut.shape[1])
    time_scale = ts[:plot_cut.shape[0]]
    
    
    #Find logs of what's happening within those ranges
    freq_range = 3 #Either way in integer lumps
    

    nominal_ranges = np.zeros((nbells,2),dtype = 'int')  #max and min integers to get the nominals
    for i in range(nbells):
        for ui, u in enumerate([-1,1]):
            nominal_ranges[i, ui] = int(freq_ints[i] + u*freq_range)  #Can check plot to make sure these are within. Should be for meanwood
            
    #Get times of initial rounds just from overall volume?! And can optimise...
    
    nrounds = 2  #Number of rounds in the cut
    rounds_start = 4.0
    rounds_end = 8.5
    
    fig, axs = plt.subplots(nbells+1)

    maxmin = 0.0
   # start_freq = allfreqs[-1]*0.75
    #end_freq = allfreqs[]
    #Want to optimise this range... Find minimum prominence?
    for start_test in range(0, 60 ,5):
        for end_test in range(start_test, 1000,5):
            #print(start_test, end_test, maxmin, end_test/freq_ints[0])

            allsum = np.sum(allfreqs[:,start_test:end_test], axis = 1)
            peaks, _ = find_peaks(allsum)
        
            peaks = peaks[peaks > int(rounds_start/dt)]
            peaks = peaks[peaks < int(rounds_end/dt)]
            
            prominences = peak_prominences(allsum, peaks)[0]
        
            peaks = np.array([val for _, val in sorted(zip(prominences, peaks), reverse = True)]).astype('int')
            prominences = sorted(prominences, reverse = True)
            
            if len(prominences) > 0:
                if min(prominences[:nbells*nrounds]) > maxmin:
                    maxmin = min(prominences[:nbells*nrounds])
                    start_best = start_test; end_best = end_test
            
    #start_best = 0; end_best = 1000

    allsum = np.sum(allfreqs[:,start_best:end_best], axis = 1)
    peaks, _ = find_peaks(allsum)

    peaks = peaks[peaks > int(rounds_start/dt)]
    peaks = peaks[peaks < int(rounds_end/dt)]
    
    prominences = peak_prominences(allsum, peaks)[0]

    peaks = np.array([val for _, val in sorted(zip(prominences, peaks), reverse = True)]).astype('int')
    prominences = sorted(prominences, reverse = True)
    
    peaks = peaks[:nbells*nrounds]
    prominences = prominences[:nbells*nrounds]

    peaks = sorted(peaks)
    
    
    for k in range(nbells*nrounds):
        axs[0].scatter(ts[peaks[k]],0.0)
        
    nominal_logs = np.zeros((nbells, len(ts)))
    axs[0].plot(ts, allsum)
    axs[0].set_xlim(rounds_start, rounds_end)
    
    for i in range(nbells):
        ax = axs[i+1]
        #Do logs of these transforms in time
        nominal_logs[i] = np.max(allfreqs[:,nominal_ranges[i,0:nominal_ranges[i,1]]],axis = 1)
        
        ax.plot(ts, nominal_logs[i])
        ax.set_xlim(rounds_start, rounds_end)
        
    plt.tight_layout()
    plt.show()
         

    for i in range(nbells):
        plt.plot([freq_ints[i], freq_ints[i]], [0.0,10.0], c = 'red', linestyle = 'dotted')
    plt.pcolormesh(freq_scale, time_scale, plot_cut)
    for i in range(nbells):
        for u in [-1,1]:
            plt.plot([freq_ints[i] + u*freq_range, freq_ints[i] + u*freq_range], [0.0,0.5], c = 'green', linestyle = 'dotted')
    
    for k in range(len(peaks)):
        plt.plot([0,2000], [peaks[k]*dt,peaks[k]*dt], c = 'red')
        
    plt.xlim(0.0,freq_scale[-1])
    plt.ylim(min(peaks)*dt - 0.5, max(peaks)*dt + 0.5)

    tbefore = 0.1  #These could theoretically be optimised
    tafter = 0.1
    
    for k in range(len(peaks)):
        if k%5 == -1:
              plt.plot([0,2000], [peaks[k]*dt-tbefore,peaks[k]*dt-tbefore], c = 'green')
              plt.plot([0,2000], [peaks[k]*dt+tafter,peaks[k]*dt+tafter], c = 'green')
         
    plt.gca().invert_yaxis()
    plt.show()

    min_freq_int = int(freq_ints[-1]*0.75)
    max_freq_int = int(freq_ints[0]*4)
    ntests = 6
    bell_frequencies = []; first_strikes = []
    #Run through and find the frequencies most prominent at these times? Must be a few of them. Doesn't line up well with nominals...
    for bell in range(nbells):
        freq_picks = []
        for rounds in range(nrounds):
            peak_id = rounds*nbells + bell
            #print('Bell', bell, 'time', peaks[peak_id])
            
            start = int(peaks[peak_id]-tbefore/dt); end = int(peaks[peak_id] + tafter/dt)
            
            if rounds == 0:
                first_strikes.append(peaks[peak_id]*dt)
                
            diff = allfreqs[end,:] - allfreqs[start,:] 
            #Cut out negatives
            diff[diff < 0.0] = 0.0
            
            #Shift diff to favour lower frequencies
            xs = np.linspace(0.0, 2*len(diff)/max_freq_int, len(diff))

            diff = diff*xs**1.05
            
            plt.plot(diff[:max_freq_int])

            peak_freqs, _ = find_peaks(diff)

            peak_freqs = peak_freqs[peak_freqs > min_freq_int]
            peak_freqs = peak_freqs[peak_freqs < max_freq_int]
            
            prominences = peak_prominences(diff, peak_freqs)[0]

            peak_freqs = np.array([val for _, val in sorted(zip(prominences, peak_freqs), reverse = True)]).astype('int')
            prominences = sorted(prominences, reverse = True)

            sorted_tests = sorted(peak_freqs[:ntests])
            freq_picks.append(sorted_tests)
            
            #print(peaks[:ntests])
        
            plt.close()

            #Log (as a LIST) the frequencies which have consistently increased a lot here (at all rounds)
            #Needs to be consistent across everything though -- important
        fudge = 2 #Leeway either side
        confirmed_picks = []
        current_freq = 0.0
        for freq in freq_picks[0]:
            allfine = True
            for i2 in range(1, len(freq_picks)):
                fine = False
                for j2 in range(len(freq_picks[i2])):
                    if abs(freq_picks[i2][j2] - freq) <= fudge:
                        current_freq = (freq + freq_picks[i2][j2])/2
                        fine = True
                if not fine:
                     allfine = False
            if allfine:
                confirmed_picks.append(current_freq*fs/(cut_end-cut_start))
        bell_frequencies.append(confirmed_picks)
        
    return bell_frequencies, first_strikes
    
def find_strike_times(fs,norm, dt, cut_length, bell_frequencies, first_strikes):
    #Find times of each bell striking, with some confidence
    
    #Cut_length is the length of the Fourier transform. CENTRE the time around this
    nbells = len(bell_frequencies)
    count = 0
    allfreqs = []; ts = []
    tmax = len(norm)/fs
    
    t = cut_length/2
    trans_length = 2*int(fs*cut_length/2)
    
    while t < tmax - cut_length/2:
        cut_start  = int(t*fs - fs*cut_length/2)
        cut_end    = cut_start + trans_length
        
        if count%50 == -1:
            print('Analysing, t = ', t)
            
        trans = transform(fs, norm[cut_start:cut_end])
            
        count += 1
        
        ts.append(t)        
        allfreqs.append(trans)
        
        t = t + dt
        
    allfreqs = np.array(allfreqs)    
    
    #Run through bells and see what happens
    tbefore = 0.1  #These could theoretically be optimised
    tafter = 0.1

    error_range = 0.75 #Allowable variation in speed (dictaed by ringing things)
    fd = int(0.1/dt)   #Allowable variation in frequency time (delay for hum note etc.)
    
    nrounds = 100
    
    all_strikes = []; all_confidences = []
    for bell in range(5):
        
        bell_strikes = []; bell_confidences = []
        
        freqs = bell_frequencies[bell]
        freq_ints =  np.array(trans_length*np.array(freqs)/fs).astype('int')
        
        fig, axs = plt.subplots(len(freqs))

        all_logdiffs = []
        
        for fi, freq_test in enumerate(freq_ints):
            log = np.sum(allfreqs[:,freq_test-1:freq_test+1], axis = 1)
            log = gaussian_filter1d(log, sigma = 0.1/dt)
            logdiffs = np.zeros(len(log))
            for i in range(len(logdiffs)):
                logdiffs[i] = log[int(i + tafter*dt)] -   log[int(i - tbefore*dt)]     
                
            #Find prominences of logdiffs in this range

            #Cut out negatives
            logdiffs[logdiffs < 0.0] = 0.0
            #logdiffdiffs = logdiffs[1:] - logdiffs[:-1]
            
            axs[fi].plot(ts,log/np.max(log))
            axs[fi].plot(ts,logdiffs/np.max(logdiffs))
            axs[fi].set_xlim(20.0,40.0)

            all_logdiffs.append(logdiffs)
            
        for ri in range(nrounds):
            all_poss = []

            if ri == 0:  #Use 'first strike' time to give range
                mint = int((first_strikes[bell] - error_range)/dt)
                maxt = int((first_strikes[bell] + error_range)/dt)
    
            else:   #Use previous strike to inform this
                mint = int((bell_strikes[-1] + 2.2 - error_range)/dt)
                maxt = int((bell_strikes[-1] + 2.2 + error_range)/dt)
                
                if maxt > len(logdiffs):
                    break

            for fi, freq_test in enumerate(freq_ints):
                logdiffs = all_logdiffs[fi]
                   
                #Time to log OVERALL is when the second derivative of the frequency closest to the time increases fastest
                poss_times, _ = find_peaks(logdiffs[mint:maxt], prominence = 0.1*np.max(logdiffs))
                                    
                all_poss.append(poss_times)
                axs[fi].scatter(mint*dt, 0.0, c = 'blue')
                axs[fi].scatter(maxt*dt, 0.0, c = 'yellow')
                    
                    
                bestn = 0; besttime = 0
                for test in range(maxt-mint):
                    #Find peaks which lie in this area, with confidence?
                    n = 0; sumtime = 0
                    for times in all_poss:
                        if any(test-fd < num < test+fd for num in times):
                            sumtime += times[(times > test-fd)*(times < test+fd)][0]
                            n += 1
                    if n > bestn:
                        bestn = n
                        besttime = sumtime/n
                    
            if bestn/len(all_poss) < 0.5:  #Don't take this one, as it's probably wrong...
                bell_confidences.append(0.0)
                bell_strikes.append(bell_strikes[-1] + 2.2)
                
            else:
                bell_strikes.append(dt*(besttime + mint))
                bell_confidences.append(bestn/len(all_poss))
        
        for fi in range(len(freq_ints)):
            axs[fi].scatter(bell_strikes, np.zeros(len(bell_strikes)), c= 'green')

        plt.suptitle('Bell %d' % (bell))
        plt.tight_layout()
        plt.show()
        
        all_strikes.append(bell_strikes)
        all_confidences.append(bell_confidences)
        
        print(len(bell_strikes))
    #Trim so there's the right amount of rounds
    min_length = 1e6
    for bell in range(nbells):
        min_length = min(min_length, len(all_strikes[bell]))
    
    for bell in range(nbells):
        all_strikes[bell] = all_strikes[bell][:min_length]
        all_confidences[bell] = all_confidences[bell][:min_length]

    
    all_strikes = np.array(all_strikes) 
    all_confidences = np.array(all_confidences)
        
    return all_strikes, all_confidences

def plot_strikes(all_strikes, all_confidences):
    #Plots the things
    fig = plt.figure(figsize = (10,7))
    nbells = len(all_strikes)
    nrows = len(all_strikes[0])
    yvalues = np.arange(nbells) + 1
    
    for bell in range(nbells):
        plt.scatter(all_strikes[bell], yvalues[bell]*np.ones(len(all_strikes[bell])),s=all_confidences[bell]*100)
    
    for row in range(nrows):
        plt.plot(all_strikes[:,row],yvalues)
        order = np.array([val for _, val in sorted(zip(all_strikes[:,row], yvalues), reverse = False)])
        print(row, min(all_strikes[:,row]), order)

    plt.xlim(10.0,100.0)
    plt.gca().invert_yaxis()
    plt.show()
    
def reinforce_frequencies(fs, norm, dt, cut_length, all_strikes, all_confidences, nominal_freqs):
    #Use confident picks to get better values for the frequencies... Theoretically. 
    #Similar to the initial frequency finder.

    nbells = len(nominal_freqs)
    cut_start = 0; cut_end = int(cut_length*fs)
    freq_ints = np.array((cut_end - cut_start)*1.0*nominal_freqs/fs).astype('int') + 1   #Integer values for the correct frequencies. One hopes.

    count = 0
    ts = []
    allfreqs = []
    
    while cut_end < len(norm):
        trans = transform(fs, norm[cut_start:cut_end])
        
        cut_start = cut_start + int(dt*fs)
        cut_end = cut_start + int(cut_length*fs)
    
        count += 1
        
        ts.append((cut_start + cut_end)/(2*fs))
        
        allfreqs.append(trans)
        
    
    allfreqs = np.array(allfreqs)
    plot_cut = np.array(allfreqs)[:,:200]
    freq_scale = np.arange(plot_cut.shape[1])
    time_scale = ts[:plot_cut.shape[0]]
    
    nrows = len(all_strikes[0])
    
    min_freq_int = int(freq_ints[-1]*0.75)
    max_freq_int = int(freq_ints[0]*5)
    
    ntests = 10
    
    bell_frequencies = []; first_strikes = []
    
    tbefore = 0.1  #These could theoretically be optimised
    tafter = 0.1

    #Run through and find the frequencies most prominent at these times? Must be a few of them. Doesn't line up well with nominals...
    for bell in range(nbells):
        freq_picks = []
        for row in range(nrows):
                        
            start = int(all_strikes[bell,row]/dt-tbefore/dt); end = int(all_strikes[bell,row]/dt + tafter/dt)
                            
            diff = allfreqs[end,:] - allfreqs[start,:] 
            #Cut out negatives
            diff[diff < 0.0] = 0.0
            
            #Shift diff to favour lower frequencies
            xs = np.linspace(0.0, 2*len(diff)/max_freq_int, len(diff))

            diff = diff*xs**1.05
            
            plt.plot(diff[:max_freq_int])

            peak_freqs, _ = find_peaks(diff)

            peak_freqs = peak_freqs[peak_freqs > min_freq_int]
            peak_freqs = peak_freqs[peak_freqs < max_freq_int]
            
            prominences = peak_prominences(diff, peak_freqs)[0]

            peak_freqs = np.array([val for _, val in sorted(zip(prominences, peak_freqs), reverse = True)]).astype('int')
            prominences = sorted(prominences, reverse = True)

            sorted_tests = sorted(peak_freqs[:ntests])
            freq_picks.append(sorted_tests)
            
            #print(peaks[:ntests])

            #Log (as a LIST) the frequencies which have consistently increased a lot here (at all rounds)
            #Needs to be consistent across everything though -- important
        nout_max = int(nrows//10)
        fudge = 2 #Leeway either side
        confirmed_picks = []
        current_freq = 0.0
        for freq in freq_picks[0]:
            allfine = True
            nout = 0
            for i2 in range(1, len(freq_picks)):
                fine = False
                for j2 in range(len(freq_picks[i2])):
                    if abs(freq_picks[i2][j2] - freq) <= fudge:
                        current_freq = (freq + freq_picks[i2][j2])/2
                        fine = True
                if not fine:
                     nout += 1
            if nout < nout_max:
                confirmed_picks.append(current_freq*fs/(cut_end-cut_start))
        print(bell, freq_ints[bell], np.array(confirmed_picks)/fs*(cut_end-cut_start))
        bell_frequencies.append(confirmed_picks)   
        
        plt.scatter(np.array(confirmed_picks)/fs*(cut_end-cut_start),np.zeros(len(confirmed_picks)), c = 'black')
        plt.title(bell)
        plt.close()

    return bell_frequencies
    
    
    

cut_length= 0.1 #Time for each cut
#freqs_ref = np.array([1899,1692,1582,1411,1252,1179,1046,930,828,780,693,617])
nominal_freqs = np.array([1031,918,857,757,676])


fs, data = wavfile.read('audio/meanwood.wav')

print('Audio length', len(data)/fs)
tmax = 10.5
tmin = 0.0#1.5
cutmin = int(tmin*fs)
cutmax = int(tmax*fs)

import1 = np.array(data)[:,0]

ts = np.linspace(0.0, len(import1)/fs, len(import1))

dt = 0.01  #Time between analyses

audio_length = len(import1)


norm = normalise(16, import1)

dt = 0.01
cut = 0.1

bell_frequencies, first_strikes = initial_analysis(fs, norm[cutmin:cutmax], dt, cut_length, nominal_freqs)

print('Initial frequencies', bell_frequencies)

if False:
    #Using crude initial analysis, find bell frequencies

    #Run through some dt and cut lengths to see things

    #Then do Fourier analysis on the whole thing
    all_strikes, all_confidences = find_strike_times(fs, norm, dt, cut_length, bell_frequencies, first_strikes)
    
    np.save('allstrikes.npy', all_strikes)
    np.save('allconfs.npy', all_confidences)
else:
    all_strikes = np.load('allstrikes.npy')
    all_confidences = np.load('allconfs.npy')
    
plot_strikes(all_strikes, all_confidences)
        
print('Confidence', dt, cut_length, np.sum(all_confidences)/np.size(all_confidences))

bell_frequencies = reinforce_frequencies(fs, norm, dt, cut_length, all_strikes, all_confidences, nominal_freqs)

print('New frequencies', bell_frequencies)

all_strikes, all_confidences = find_strike_times(fs, norm, dt, cut_length, bell_frequencies, first_strikes)

plot_strikes(all_strikes, all_confidences)

print('Confidence', dt, cut_length, np.sum(all_confidences)/np.size(all_confidences))

#Things are imported -- find initial bell amplitudes from the reference frequencies. Will be a bit rubbish.

#ts, logs = find_bell_amps(fs,norm, dt, cut_length, freqs_ref, freqs_ref)   #Find INITIAL profile based on frequency guess

'''
allstrikes, allmags = find_strikes(ts, logs, upness_flag = True, doplot = True)   #Upness is DEFINITELY better

rounds_times, allrows, allcerts  = determine_rhythm(allstrikes, allmags, rounds = True)
    
for row in allrows:
    plt.scatter(np.linspace(1,len(row),len(row)), row)

plt.ylabel('Time')
plt.xlabel('Bells')
plt.ylim(0,30)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('rounds.png')
plt.show()

#From these initial rounds times, attempt to build some better frequency profiles. Need both hand and back really, but we'll see

new_freqs = determine_frequencies(fs, norm, dt, cut_length, freqs_ref, freqs_ref, rounds_times, allcerts, upness_flag = True)

ts, logs = find_bell_amps(fs,norm, dt, cut_length, new_freqs, freqs_ref)   #Find INITIAL profile based on frequency guess

allstrikes, allmags = find_strikes(ts, logs, upness_flag = True, doplot = True)   #Upness is DEFINITELY better

rounds_times, allrows, allcerts  = determine_rhythm(allstrikes, allmags, rounds = True)
    
#for row in allrows:
#    plt.scatter(np.linspace(1,len(row),len(row)), row)

plt.ylabel('Time')
plt.xlabel('Bells')
plt.ylim(0,30)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('rounds.png')
plt.show()

'''


#np.save('all_freqs.npy', new_freqs)


#np.save('all_freqs.npy', init_freqs)

    
#all_freqs = np.load('all_freqs.npy')

#do_rounds_etc(fs, norm, dt, cut_length, all_freqs)
#print(np.shape(all_freqs))





#def determine_rhythm(allstrikes, allmags):
    #Whittle down which are the correct strikes using the rhythm of the rounds. T


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
