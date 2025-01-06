# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 19:18:11 2025

@author: eleph
"""

import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.io import wavfile
import numpy as np
from scipy.signal import find_peaks, peak_prominences
import statistics

fs, data = wavfile.read('stockton_roundslots.wav')
#fs, data = wavfile.read('bellsound_deep.wav')

import1 = np.array(data)[:,0]

dt = 0.015  #Time between analyses
cut_length= 0.175 #Time for each cut

audio_length = len(import1)
print(audio_length)

ncuts = int(audio_length/cut_length)

nsamples = int(cut_length*fs)


ts = np.linspace(0.0, len(import1)/fs, len(import1))

def normalise(nbits, raw_input):
    #Normalises the string to the number of bits
    return raw_input/(2**(nbits-1))
    
def transform(fs, norm_cut):
    #Produce the fourier transform of the input data
    trans1 = abs(fft(norm_cut)[:len(norm_cut)//2])
    return 0.5*trans1*fs/len(norm_cut)



norm = normalise(16, import1)

cut_start = 0; cut_end = int(cut_length*fs)
count = 0

all_peaks = []; ts = []
#freqs = [617,693,780,828,930,1046,1179,1252,1407,1582,1679,1892]   #Frequencies of the bells (in descending order)
freqs = np.array([1892,1679,1582,1407,1252,1179,1046,930,828,780,693,617])
freq_ints = np.array((cut_end - cut_start)*1.0*freqs/fs).astype('int')   #Integer values for the correct frequencies

logs = [[] for _ in freqs]

if False:
    while cut_end < len(norm):
        print(cut_end, len(norm))
        trans = transform(fs, norm[cut_start:cut_end])
        
        
        cut_start = cut_start + int(dt*fs)
        cut_end = cut_start + int(cut_length*fs)
    
        count += 1
    
        npeaks = 5
    
        peaks, _ = find_peaks(trans)
        prominences = peak_prominences(trans, peaks)[0]
        
        top_freqs = np.array([val for _, val in sorted(zip(prominences, peaks), reverse = True)])
    
        top_freqs = 0.5*top_freqs*fs/len(trans)
        
        
        #plt.plot(ts[:nsamples], norm[:nsamples])
        #plt.show()
        all_peaks.append(top_freqs[:npeaks])
        ts.append((cut_start + cut_end)/(2*fs))
        
        for bell in range(len(freq_ints)):
            logs[bell].append(max(trans[freq_ints[bell]-1:freq_ints[bell]+2]))
                
        if False:
            fig = plt.figure(figsize = (10,7))
            plt.title(cut_start/fs)
            for freq in freqs:
                plt.plot([freq, freq], [0,2000])
            plt.scatter(top_freqs[:npeaks], np.zeros(npeaks) ,c = 'black')
            plt.plot(np.linspace(0.0, fs/2, len(trans)), trans)
            
            #plt.plot(trans)
    
            plt.xlabel('Frequency')
            plt.ylabel('Peakness')
            plt.xscale('log')
            plt.xlim(200,2000)
            plt.ylim(0,2000)
            plt.show()
          
    np.save('logs.npy', logs)
    np.save('ts.npy', ts)

logs = np.load('logs.npy')
ts = np.load('ts.npy')

if False:
    for bell in range(12):
        plt.title((bell+1))
        plt.plot(ts, logs[bell])
        plt.show()

if True:
    allstrikes = []; allmags = []
    fig = plt.figure(figsize = (5,10))
    for bell in range(len(freqs)):
        #Find strike times
        strikes = []; mags = []
        
        peaks, _ = find_peaks(logs[bell])
        prominences = peak_prominences(logs[bell], peaks)[0]
    
        strike_times = np.array([val for _, val in sorted(zip(prominences, peaks), reverse = True)]).astype('int')
        prominences = sorted(prominences, reverse = True)
    
        
        for k in range(len(prominences)):
            if prominences[k] > np.percentile(prominences, 90):
                strikes.append(ts[strike_times[k]])
                mags.append(100.0*prominences[k]/max(prominences))
            
        plt.scatter(np.ones(len(strikes))*(bell+1), strikes, c = 'black', s = mags)
    
        allstrikes.append(strikes)
        allmags.append(mags)
        
def std_dev(strikes):
    diffs = []
    sortstrikes = sorted(strikes)
    for i in range(len(sortstrikes) -1):
        diffs.append(sortstrikes[i+1] - sortstrikes[i])
    if np.mean(diffs) > 1.5 and np.mean(diffs) < 3.0:
        
        return statistics.stdev(diffs)
    else:
        return 1e6
    
def determine_rhythm(allstrikes, allmags):
    #Work off the tenor to begin with as the tenor strikes are pretty obvious
    nbells = len(allstrikes)
    rounds_times = [[] for _ in range(nbells)]

    tenor_strikes = allstrikes[nbells-1]
    tenor_mags = allmags[nbells-1]
    minstd = 1e6
    #Remove strikes until the standard deviation of the difference between strikes is minimised
    for i in range(3,len(tenor_strikes)):
        #print(tenor_strikes[i], tenor_mags[i])
        if std_dev(tenor_strikes[:i]) < minstd:
            imin  = i
            minstd = std_dev(tenor_strikes[:i])
    tenor_strikes = sorted(tenor_strikes[:imin])   #These are now probably the correct tenor strikes. Maybe...
    print('Peal speed', (5000/3600)*(tenor_strikes[-1] - tenor_strikes[0])/(len(tenor_strikes)-1))
    #How far out is one 'blow'? Should narrow down the time available for bells to strike reasonably.
    blow_deviation = 2*((tenor_strikes[-1] - tenor_strikes[0])/(len(tenor_strikes)-1))/(nbells * 2 + 1)
    tchange = (tenor_strikes[-1] - tenor_strikes[0])/(len(tenor_strikes)-1)
    #Start counting after first tenor strike. Assume no handstorke gap for now.
    
    for row in range(1,2):   #Do based on bell position after this
        start = tenor_strikes[row-1]
        end = tenor_strikes[row]
        #print('row', start, end)
        for bell in range(nbells):
            predict = start + (end - start)*(bell + 1)/nbells
            startbell = predict - blow_deviation*2
            endbell = predict + blow_deviation*2
            #Find best strike in this range
            minmag = 0.0
            for k, strike in enumerate(allstrikes[bell]):
                if strike >= startbell and strike <= endbell:
                    #print(strike, allmags[bell][k])
                    if allmags[bell][k] > minmag:
                        minmag = allmags[bell][k]
                        k_strike = k
            if minmag > 0.0:
                rounds_times[bell].append(allstrikes[bell][k_strike])
            else:
                print('Starting rounds not found. Bugger')
                rounds_times[bell].append(-1)
    
    miscount = np.zeros(nbells).astype('int')
    row = 2
    while np.sum(miscount) < nbells:
        for bell in range(nbells):
            
            predict = rounds_times[bell][-miscount[bell] - 1] + tchange*(miscount[bell] + 1)
            startbell = predict - blow_deviation*2.5
            endbell = predict + blow_deviation*2.5
            minmag = 0.0
                
            for k, strike in enumerate(allstrikes[bell]):
                if strike >= startbell and strike <= endbell:
                    if bell == 5:
                        print(strike, allmags[bell][k])
                    if allmags[bell][k] > minmag:
                        minmag = allmags[bell][k]
                        k_strike = k
            if minmag > 0.0:
                rounds_times[bell].append(allstrikes[bell][k_strike])
                miscount[bell] = 0
                print(rounds_times[bell])
            else:
                print('Change not found')
                rounds_times[bell].append(-1)
                miscount[bell] += 1
        row += 1
    #print(tenor_strikes)
    return rounds_times


rounds_times = determine_rhythm(allstrikes, allmags)
    
for bell in range(len(rounds_times)):
    plt.scatter(np.ones(len(rounds_times[bell]))*(bell+1), rounds_times[bell], c = 'red')

plt.ylabel('Time')
plt.xlabel('Bells')
plt.ylim(0,40)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('rounds.png')
plt.show()
    
    
if False:
    fig = plt.figure(figsize = (10,7))
    
    
    for freq in freqs:
        plt.plot([ts[0], ts[-1]], [freq, freq])
        
    for i in range(len(all_peaks)):
    
        plt.scatter(ts[i]*np.ones(len(all_peaks[i])), all_peaks[i], c = 'black', s = 2.5)
    plt.xlabel('Time')
    
    plt.yscale('log')
    
    plt.show()

#def determine_rhythm(allstrikes, allmags):
    #Whittle down which are the correct strikes using the rhythm of the rounds. T
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
