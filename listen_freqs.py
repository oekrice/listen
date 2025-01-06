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



def normalise(nbits, raw_input):
    #Normalises the string to the number of bits
    return raw_input/(2**(nbits-1))
    
def transform(fs, norm_cut):
    #Produce the fourier transform of the input data
    trans1 = abs(fft(norm_cut)[:len(norm_cut)//2])
    return 0.5*trans1*fs/len(norm_cut)

def find_bell_amps(fs,norm, dt, cut_length, freqs):
    #Outputs logs of the amplitudes around the specified bell frequencies.
    
    count = 0
    cut_start = 0; cut_end = int(cut_length*fs)
    all_peaks = []; ts = []
    
    logs = [[] for _ in freqs]

    freq_ints = np.array((cut_end - cut_start)*1.0*freqs/fs).astype('int')   #Integer values for the correct frequencies

    while cut_end < len(norm):
        if count%10 == 9:
            print('Analysing, t = ', cut_start/fs)
        trans = transform(fs, norm[cut_start:cut_end])
        
        cut_start = cut_start + int(dt*fs)
        cut_end = cut_start + int(cut_length*fs)
    
        count += 1
    
        npeaks = 5
    
        peaks, _ = find_peaks(trans)
        prominences = peak_prominences(trans, peaks)[0]
        
        top_freqs = np.array([val for _, val in sorted(zip(prominences, peaks), reverse = True)])
    
        top_freqs = 0.5*top_freqs*fs/len(trans)
        
        
        all_peaks.append(top_freqs[:npeaks])
        ts.append((cut_start + cut_end)/(2*fs))
        
        
        for bell in range(len(freq_ints)):
            freqmin = freq_ints[bell]-1
            freqmax = freq_ints[bell]+2
            logs[bell].append(max(trans[freqmin:freqmax]))
            #logs[bell].append(sum(trans[freqmin:freqmax]))
    
            
            
        if False:
            if cut_start/fs < 6.0 and cut_start/fs > 3.2:
                fig = plt.figure(figsize = (10,7))
                plt.title(cut_start/fs)
                for freq in freqs:
                    plt.plot([freq, freq], [0,2000])
                plt.scatter(top_freqs[:npeaks], np.zeros(npeaks) ,c = 'black')
                plt.plot(np.linspace(0.0, fs/2, len(trans)), trans, c= 'red')
                    
                plt.xlabel('Frequency')
                plt.ylabel('Peakness')
                plt.xscale('log')
                plt.xlim(200,2000)
                plt.ylim(0,2000)
                plt.show()
          
    np.save('logs.npy', logs)
    np.save('ts.npy', ts)



def find_strikes(ts, logs):

    allstrikes = []; allmags = []
    
    test_bell = 1
    #fig = plt.figure(figsize = (5,5))
    
    #plt.plot(ts[:1000], logs[test_bell][:1000])
    #plt.show()
    
    fig = plt.figure(figsize = (5,10))

    funcs = np.arange(0,len(freqs))*0.1 + 1.0

    #funcs[3] = 2.0

    funcs[1] = 2.2
    funcs[7] = 0.75
    #funcs[0] = 1.5
    
    for bell in range(len(freqs)):#range(test_bell, test_bell + 1):#len(freqs)):

        #Find strike times
        strikes = []; mags = []
        if True:#bell > 1 and bell < 10:
                
   
            nprev = int(funcs[bell]/dt)
                
            upness = np.zeros(len(logs[bell]))
            for n in range(0, len(logs[bell])):
                if n < nprev:
                    upness[n] = 0.
                else:
                    upness[n] = logs[bell][n]/(np.mean(logs[bell][n - nprev:n]))
                    
    
            peaks, _ = find_peaks(upness)
            prominences = peak_prominences(upness, peaks)[0]
        else:
            peaks, _ = find_peaks(logs[bell])
            prominences = peak_prominences(logs[bell], peaks)[0]
        
        strike_times = np.array([val for _, val in sorted(zip(prominences, peaks), reverse = True)]).astype('int')
        prominences = sorted(prominences, reverse = True)
        #print(ts[strike_times])

        #print(prominences)
        #plt.plot(ts, upness)

        #plt.show()

        for k in range(len(prominences)):
            if prominences[k] > np.percentile(prominences, 80):# and logs[bell][max(0, strike_times[k] - int(0.1*fs))] < base_threshold:
                strikes.append(ts[strike_times[k]])
                mags.append(100.0*prominences[k]/max(prominences))
            
        #plt.scatter(strikes, np.zeros(len(strikes)), c = 'black')
        plt.scatter(np.ones(len(strikes))*(bell+1), strikes, c = 'black', s = mags)
    
        allstrikes.append(strikes)
        allmags.append(mags)
        

    #plt.show()
    return allstrikes, allmags

def std_dev(strikes):
    diffs = []
    sortstrikes = sorted(strikes)
    for i in range(len(sortstrikes) -1):
        diffs.append(sortstrikes[i+1] - sortstrikes[i])
    if np.mean(diffs) > 1.5 and np.mean(diffs) < 3.0:
        
        return statistics.stdev(diffs)
    else:
        return 1e6
    
def print_row(times):
    #Prints the order of the bells
    bellnames = np.arange(1,len(times) + 1).astype('str')
    
    bellnames[9] == '0'
    bellnames[10] = 'E'
    bellnames[11] = 'T'
    order = [val for _, val in sorted(zip(times, bellnames))]
    if order[-1] != 'T':
        print(order)
    
def determine_rhythm(allstrikes, allmags):
    #Work off the tenor to begin with as the tenor strikes are pretty obvious
    nbells = len(allstrikes)
    rounds_times = [[] for _ in range(nbells)]

    tenor_strikes = allstrikes[nbells-1]
    tenor_mags = allmags[nbells-1]
    minstd = 1e6
    
    rounds_cutoff = 20  #Seconds of rounds to consider for overall speed purposes.
    rounds_strikes = []
    for strike in tenor_strikes:
        if strike < rounds_cutoff:
            rounds_strikes.append(strike)
        
    rounds_strikes = np.array(rounds_strikes)
    #Remove strikes until the standard deviation of the difference between strikes. is minimised
    for i in range(3,len(rounds_strikes)):
        #print(tenor_strikes[i], tenor_mags[i])
        if std_dev(rounds_strikes[:i]) < minstd:
            imin  = i
            minstd = std_dev(rounds_strikes[:i])
            #print(minstd, np.array(sorted(rounds_strikes[:imin]))[1:] - np.array(sorted(rounds_strikes[:imin]))[:-1] )
            
    tenor_strikes = np.array(sorted(rounds_strikes[:imin]))   #These are now probably the correct tenor strikes. Maybe...
    
    #Get an even number
    if len(tenor_strikes)%2 == 1:
        tenor_strikes = tenor_strikes[:-1]
    
    handstroke = False
    #Determine whether it starts at handstroke or backstroke
    diffs = tenor_strikes[1:] - tenor_strikes[:-1]
    diff2s = diffs[1:] - diffs[:-1]
    if np.sum(diff2s[1::2]) - np.sum(diff2s[0::2]) > 0:
        handstroke = True
    print('Is first logged row handstroke?', handstroke)
    
    #print('Peal speed', (5000/3600)*(tenor_strikes[-1] - tenor_strikes[0])/(len(tenor_strikes)-1))
    #How far out is one 'blow'? Should narrow down the time available for bells to strike reasonably.
    blow_deviation = 2*((tenor_strikes[-1] - tenor_strikes[0])/(len(tenor_strikes)-1))/(nbells * 2 + 1)
    tchange = (tenor_strikes[-1] - tenor_strikes[0])/(len(tenor_strikes)-1)
    #Start counting after first tenor strike. Assume no handstorke gap for now.
    
    #Find initial row
    row = 1
    start = tenor_strikes[row-1]
    end = tenor_strikes[row]
    #print('row', start, end)
    current_row = np.zeros(nbells)
    
    for bell in range(nbells):
        if handstroke:
            predict = start + (end - start)*(bell + 2)/(nbells + 1)
        else:
            predict = start + (end - start)*(bell + 1)/(nbells)

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
            current_row[bell] = allstrikes[bell][k_strike]

        else:
            print('Starting rounds not found. Bugger')
            rounds_times[bell].append(-1)
            current_row[bell] = -1
        
    #Find subsequent rows
    miscount = np.zeros(nbells).astype('int')
    row = 2
    
    tboth = 2*(tenor_strikes[-1] - tenor_strikes[0])/(len(tenor_strikes) - 1)
    thand = (nbells+1)*tboth/(2*nbells+1)
    tback = (nbells)*tboth/(2*nbells+1)
    
    print(tboth, thand, tback)
    print('Time for two changes', tboth)

    
    allrows = np.array([current_row])
    
    while np.sum(miscount) < 4:
        handstroke = not(handstroke)
        #Find starts and end of each row
        if handstroke:
            predictstart = sorted(current_row)[1] + thand - 2*blow_deviation
            predictend = sorted(current_row)[nbells-2] + thand + 2*blow_deviation
        else:
            predictstart = sorted(current_row)[1] + tback - 2*blow_deviation
            predictend = sorted(current_row)[nbells-2] + tback + 2*blow_deviation

        print(row, predictstart, predictend)
            
        for bell in range(nbells):

            #Number of whole pulls missed.
            npulls = (miscount[bell] + 1)//2 #number of whole pulls missed
            
            predict = rounds_times[bell][-miscount[bell] - 1] + npulls*tboth
            
            if handstroke:
                predict = predict + thand*((miscount[bell] + 1)%2)
            else:
                predict = predict + tback*((miscount[bell] + 1)%2)
                
            #Add miscounts
            startbell = predictstart
            endbell = predictend
            
            startbell = max(predict - blow_deviation*(2*miscount[bell] + 2.5), predictstart)
            endbell = min(predict + blow_deviation*(2*miscount[bell] + 2.5), predictend)
            
            startbell = predict - blow_deviation*(2*miscount[bell] + 2.5)
            endbell = predict + blow_deviation*(2*miscount[bell] + 2.5)
            
            maxmag = 0.0
                
            for k, strike in enumerate(allstrikes[bell]):
                if strike >= startbell and strike <= endbell:
                    if allmags[bell][k] > maxmag:
                        maxmag = allmags[bell][k]
                        k_strike = k
            if maxmag > 0.0:
                rounds_times[bell].append(allstrikes[bell][k_strike])
                miscount[bell] = 0
                #print(rounds_times[bell])
                current_row[bell] = allstrikes[bell][k_strike]
            else:
                #print('Change not found')
                rounds_times[bell].append(-1)
                miscount[bell] += 1
                current_row[bell] = -1

        row += 1
        if max(current_row) > 0:
            allrows = np.concatenate((allrows, [current_row]), axis = 0)
        
        #print_row(current_row)
    #print(allrows)
        #Recalculate speeds based on the last few rows? Don't bother for now...
        
    #print(tenor_strikes)
    return rounds_times, allrows


fs, data = wavfile.read('stockton_roundslots.wav')
fs, data = wavfile.read('stockton_cambridge.wav')
#fs, data = wavfile.read('bellsound_deep.wav')

tmax = 120
cut = int(tmax*fs)

import1 = np.array(data)[:cut,0]

ts = np.linspace(0.0, len(import1)/fs, len(import1))

dt = 0.02  #Time between analyses
cut_length= 0.2 #Time for each cut

audio_length = len(import1)

norm = normalise(16, import1)

freqs = np.array([1899,1692,1582,1411,1252,1179,1046,930,828,780,693,617])

#find_bell_amps(fs,norm, dt, cut_length, freqs)

logs = np.load('logs.npy')
ts = np.load('ts.npy')


allstrikes, allmags = find_strikes(ts, logs)
rounds_times, allrows = determine_rhythm(allstrikes, allmags)
    
for row in allrows:
    plt.scatter(np.linspace(1,len(row),len(row)), row)
    
#for bell in range(len(rounds_times)):
#    plt.scatter(np.ones(len(rounds_times[bell]))*(bell+1), rounds_times[bell], c = 'red')

plt.ylabel('Time')
plt.xlabel('Bells')
plt.ylim(0, 60)
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
