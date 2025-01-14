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
from scipy.signal import hilbert, chirp
from scipy import signal
from scipy.ndimage import gaussian_filter1d

def normalise(nbits, raw_input):
    #Normalises the string to the number of bits
    return raw_input/(2**(nbits-1))
    
def transform(fs, norm_cut):
    #Produce the fourier transform of the input data
    trans1 = abs(fft(norm_cut)[:len(norm_cut)//2])
    return 0.5*trans1*fs/len(norm_cut)

def find_bell_amps(fs,norm, dt, cut_length, all_freqs):
    #Outputs logs of the amplitudes around the specified bell frequencies.
    
    count = 0
    cut_start = 0; cut_end = int(cut_length*fs)
    all_peaks = []; ts = []; base = []
    
    logs = [[] for _ in range(len(all_freqs))]

    while cut_end < len(norm):
        if count%50 == 49:
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
        
        
        for bell in range(len(all_freqs)):
            #Convolve with thenew frequency data
            conv = trans*all_freqs[bell]
            
            logs[bell].append(np.sum(conv)/np.sum(trans))
            #logs[bell].append(sum(trans[freqmin:freqmax]))
    
            
        base.append(np.mean(trans))
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
    
    if False:
        for bell in range(12):
            plt.title(bell)
            plt.plot(ts, logs[bell])
            plt.show()

    np.save('base.npy', base)
    np.save('logs.npy', logs)
    np.save('ts.npy', ts)


def find_strikes(ts, logs, doplot = False, init = True):

    allstrikes = []; allmags = []
    
    test_bell = 1
    
    if doplot:
        fig = plt.figure(figsize = (5,10))

    funcs = np.arange(0,len(freqs))*0.0 + 0.6
    
    for bell in range(len(freqs)):#range(test_bell, test_bell + 1):#len(freqs)):
        logs_smooth = gaussian_filter1d(logs[bell],10)
        #Find strike times
        strikes = []; mags = []
        if not init:#bell > 1 and bell < 10:
                
            nprev = int(funcs[bell]/dt)
                
            upness = np.zeros(len(logs_smooth))
            for n in range(0, len(logs_smooth)):
                if n < nprev:
                    upness[n] = 0.
                else:
                    upness[n] = logs_smooth[n]/(np.mean(logs_smooth[n - nprev:n]))
    
            peaks, _ = find_peaks(upness)
            prominences = peak_prominences(upness, peaks)[0]
            
        else:
            peaks, _ = find_peaks(logs_smooth)
            prominences = peak_prominences(logs_smooth, peaks)[0]
        
        strike_times = np.array([val for _, val in sorted(zip(prominences, peaks), reverse = True)]).astype('int')
        prominences = sorted(prominences, reverse = True)
        #print(ts[strike_times])

        #print(prominences)

        for k in range(len(prominences)):
            if prominences[k] > 0.25*max(prominences):# and logs[bell][max(0, strike_times[k] - int(0.1*fs))] < base_threshold:
                strikes.append(ts[strike_times[k]])
                mags.append((prominences[k]/max(prominences)))
            
        if doplot:
            #plt.scatter(strikes, np.zeros(len(strikes)), c = 'black')
            plt.scatter(np.ones(len(strikes))*(bell+1), strikes, c = 'black', s = 100*np.array(mags)**2)
    
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
    if order[-2] != 'E' or order[0] != '1':
        print(max(times), order)
    
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
    if len(tenor_strikes)%2 == 0:
        tenor_strikes = tenor_strikes[:-1]
    
    print('Tenor', tenor_strikes)
    print('Diffs', tenor_strikes[1:] - tenor_strikes[:-1])

    handstroke = True
    #Determine whether it starts at handstroke or backstroke
    diffs = tenor_strikes[1:] - tenor_strikes[:-1]
    diff2s = diffs[1:] - diffs[:-1]
    if np.sum(diff2s[1::2]) - np.sum(diff2s[0::2]) > 0:
        handstroke = False
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
    
    print('Time for two changes', tboth)

    
    allrows = np.array([current_row])
    allcerts = [np.ones((len(current_row)))]
   
    current_certs = np.zeros(nbells)
    bellcerts = [[] for _ in range(nbells)]

    while np.sum(miscount) < 4:
        handstroke = not(handstroke)
        last_end = max(current_row)
        #Find starts and end of each row
        if handstroke:
            predictstart = sorted(current_row)[1] + thand - 3*blow_deviation
            predictend = sorted(current_row)[nbells-2] + thand + 3*blow_deviation
        else:
            predictstart = sorted(current_row)[1] + tback - 3*blow_deviation
            predictend = sorted(current_row)[nbells-2] + tback + 3*blow_deviation
            
        for bell in range(nbells):

            #Number of whole pulls missed.
            npulls = (miscount[bell] + 1)//2 #number of whole pulls missed
            
            predict = rounds_times[bell][-miscount[bell] - 1] + npulls*tboth
            
            if handstroke:
                predict = predict + thand*((miscount[bell] + 1)%2)
            else:
                predict = predict + tback*((miscount[bell] + 1)%2)
                
            #Add miscounts
            #startbell = predictstart
            #endbell = predictend
            
            startbell = predict - blow_deviation*(2*miscount[bell] + 4.0)
            endbell = predict + blow_deviation*(2*miscount[bell] + 4.0)
            #startbell = max(predict - blow_deviation*(2*miscount[bell] + 2.5), predictstart)
            #endbell = min(predict + blow_deviation*(2*miscount[bell] + 2.5), predictend)
            
            #startbell = last_end - blow_deviation*2
            #endbell = last_end + thand*1.25
                      
            #startbell = max(startbell, current_row[bell] + 1.0)
            
            if bell == 0 and predict < 20.0:
                print(predict,  blow_deviation, startbell, endbell)
            #input()
            maxmag = 0.0
            mags_poss = []
            strikes_poss = []
            alpha = 1.0
            for k, strike in enumerate(allstrikes[bell]):
                if strike >= startbell and strike <= endbell:
                    strikes_poss.append(strike)
                    if abs(strike-predict) > 1e-6:
                        mag_adjust = allmags[bell][k]/(abs(strike-predict))**alpha
                        mags_poss.append(mag_adjust)
                    else:
                        k_strike = k
                        maxmag = 1e6
                        mags_poss.append(1e6)

                    if mag_adjust > maxmag:
                        maxmag = mag_adjust
                        k_strike = k
            #if bell == 0 and startbell < 20.0 and endbell > 20.0:
            #    print(strikes_poss, mags_poss, predict, startbell, endbell)

            if len(mags_poss) == 1:
                current_certs[bell] = 1
            elif len(mags_poss) == 0:
                current_certs[bell] = 0
            else:
                current_certs[bell] = 1.0 - sorted(mags_poss)[-2]/sorted(mags_poss)[-1]
            #if bell == 0 and startbell < 20.0 and endbell > 20.0:

            #    print('cert', current_certs[bell], allstrikes[bell][k_strike])
            if current_certs[bell] > 0.25:
                rounds_times[bell].append(allstrikes[bell][k_strike])
                miscount[bell] = 0
                #print(rounds_times[bell])
                current_row[bell] = allstrikes[bell][k_strike]
                if len(mags_poss) == 1:
                    current_certs[bell] = 1
                else:
                    frac = sorted(mags_poss)[-2]/sorted(mags_poss)[-1]
                    current_certs[bell] = 1.0 - sorted(mags_poss)[-2]/sorted(mags_poss)[-1]
                
            else:
                #print('Change not found')
                rounds_times[bell].append(predict)
                miscount[bell] += 1
                current_row[bell] = predict
                current_certs[bell] = 0

            bellcerts[bell].append(current_certs[bell])
            
        row += 1
        if max(current_row) > 0:
            allrows = np.concatenate((allrows, [current_row]), axis = 0)
            allcerts = np.concatenate((allcerts, [current_certs]), axis = 0)

        print_row(current_row)
        
    allcerts = 0
    for bell in range(12):
        print('Clarity for bell', bell + 1, np.mean(bellcerts[bell]))
        allcerts += np.mean(bellcerts[bell])
    print('Total', allcerts/12)
        #print('misses', miscount)
    #print(allrows)
        #Recalculate speeds based on the last few rows? Don't bother for now...
        
    #print(tenor_strikes)
    return rounds_times, allrows, allcerts


def determine_frequencies(freqs, cut_length):
    #Using the sample of rounds, attempts to determine the frequency profile of each bell
    fs, data = wavfile.read('stockton_roundstest.wav')
    import_rounds = np.array(data)[:,0]
    norm_rounds = normalise(16, import_rounds)
    ts = np.linspace(0.0, len(import_rounds)/fs, len(import_rounds))

    nbells = len(freqs)
    #plt.plot(ts, norm_rounds)
    #plt.show()

    #Determine the approximate times of strikes using the frequency 'delta' functions
    dt = 0.01  #Time between analyses

    count = 0
    cut_start = 0; cut_end = int(cut_length*fs)
    
    freq_ints = np.array((cut_end - cut_start)*1.0*freqs/fs).astype('int')   #Integer values for the correct frequencies

    
    treble_log = []; tenor_log = []; log_ts = []
    alllog =[[] for _ in range(nbells)]
    #Do Fourier transforms to get frequencies
    while cut_end < len(norm_rounds):
        if count%10 == 9:
            print('Analysing, t = ', cut_start/fs)
        trans = transform(fs, norm_rounds[cut_start:cut_end])
        t_centre = (ts[cut_start] + ts[cut_end])/2
        
        cut_start = cut_start + int(dt*fs)
        cut_end = cut_start + int(cut_length*fs)
        
        treble_log.append(sum(trans[freq_ints[0]-1:freq_ints[0]+2]))
        tenor_log.append(sum(trans[freq_ints[-1]-1:freq_ints[-1]+2]))
        
        for bell in range(nbells):
            alllog[bell].append(sum(trans[freq_ints[bell]-1:freq_ints[bell]+2]))
            
        log_ts.append(t_centre)
        count += 1
        
        if False:

            for freq_int in freq_ints:
                plt.plot([freq_ints, freq_ints], [0,2000])
                
            plt.xlim(0,600)
            plt.title(t_centre)
            plt.plot(trans)
            plt.show()
    
    #Find treble and tenor times and interpolate    

    #Find time of uptick in each one. 
    nbefore = 10
    treble_upness = [0 for _ in range(nbefore)]
    tenor_upness = [0 for _ in range(nbefore)]
    for k in range(nbefore, len(treble_log)): 
        treble_upness.append(treble_log[k]/np.mean(treble_log[k-nbefore:k]))
        tenor_upness.append(tenor_log[k]/np.mean(tenor_log[k-nbefore:k]))

    #Assuming treble in the first half:
    treble_time = np.where(treble_upness[:len(treble_upness)//2] == max(treble_upness[:len(treble_upness)//2]))[0][0]
    tenor_time = np.where(tenor_upness[:len(tenor_upness)] == max(tenor_upness[:len(tenor_upness)]))[0][0]

    predicted_inds = np.linspace(treble_time, tenor_time, nbells).astype('int')
    error = (predicted_inds[1] - predicted_inds[0])
        
    print('Length of change', log_ts[tenor_time] - log_ts[treble_time])
    alltimes = -1*np.ones((nbells))
    #Use these times to find more precise picks for the individual bells.
    for bell in range(nbells):
        
        #Find time of uptick in each one. 
        nbefore = 10
        upness = [0 for _ in range(nbefore)]
        for k in range(nbefore, len(alllog[bell])): 
            upness.append(alllog[bell][k]/np.mean(alllog[bell][k-nbefore:k]))
    
        start_slice = predicted_inds[bell] - error
        end_slice = predicted_inds[bell] + error

        #alltimes[bell] = log_ts[np.where(upness[start_slice:end_slice] == max(upness[start_slice:end_slice]))[0][0] + start_slice]
        alltimes[bell] = log_ts[np.where(alllog[bell][start_slice:end_slice] == max(alllog[bell][start_slice:end_slice]))[0][0] + start_slice]
    #Ensuring the cadences are kept the same, use these times to determine a coherent frequeny profile for each bell
    #Probably trannform into frequency space but don't really need to for now.
    cut_inds = int(cut_length*fs)   #Length of the fourier cut
    all_freqs = [[] for _ in range(nbells)] 
    
    for bell in range(nbells):
        
        cutmin = int(0.8*freq_ints[bell]); cutmax = int(1.2*freq_ints[bell])

        #Treble first as treble is nice.
        
        left = 0.2; right = 0.0   #Time indices to check before and after the 'peak'. Alter as necessary as not done empirically.

        #left = 0.1, right = 0.15 WORKS
        t0 = alltimes[bell] - left
        t1 = alltimes[bell] + right
        cut_0_min = int(t0*fs) - cut_inds//2
        cut_0_max = int(t0*fs) + cut_inds//2
    
        trans0 = transform(fs, norm_rounds[cut_0_min:cut_0_max])
        
        cut_1_min = int(t1*fs) - cut_inds//2
        cut_1_max = int(t1*fs) + cut_inds//2
    
        trans1 = transform(fs, norm_rounds[cut_1_min:cut_1_max])

        diff = np.maximum(0.0, trans1- trans0)
        
        #Remove frequencies below the arbitrary cutoff
        diff[:cutmin] = 0.0
        diff[cutmax:] = 0.0
        peak_max = max(diff[freq_ints[bell]-2:freq_ints[bell]+2])

        if False:
            for freq_int in freq_ints:
                plt.plot([freq_ints, freq_ints], [0,1000])

            plt.plot(diff)
            plt.scatter(freq_ints[bell], peak_max, c= 'black')
            plt.xlim(100,450)
            plt.ylim(0,1000)
            plt.title((bell, diff[freq_ints[bell]], alltimes[bell]))
            #plt.xscale('log')
            plt.show()

        all_freqs[bell] = diff/max(diff)
    
    np.save('init_freqs.npy', np.array(all_freqs))
    return np.array(all_freqs)
        
    #plt.plot(log_ts, treble_log)
    #plt.plot(log_ts, tenor_log)
    
    #plt.plot(log_ts, treble_upness)
    #plt.plot(log_ts, tenor_upness)
    #plt.show()

def reinforce_frequencies(fs, norm, rounds_times, tmin = 0, tmax = 60):
    #Outputs (hopefully) more accurate frequency stuff based on more han the initial rounds
    
    print('Reinforcing')
    freqs = np.array([1899,1692,1582,1411,1252,1179,1046,930,828,780,693,617])

    nbells = len(rounds_times)
    tlim = 60
    left = 0.2; right = 0.0   #Time indices to check before and after the 'peak'. Alter as necessary as not done empirically.

    
    #0.3 WORKS
    cut_inds = int(cut_length*fs)   #Length of the fourier cut
    all_freqs = np.zeros((nbells, (cut_inds)//2))
    freq_ints = np.array((cut_inds)*1.0*freqs/fs).astype('int')   #Integer values for the correct frequencies
    cutoff = int(0.8*freq_ints[-1])

    for bell in range(nbells):
        cutmin = int(0.8*freq_ints[bell]); cutmax = int(2.0*freq_ints[bell])

        alldiff = np.zeros((cut_inds)//2)

        count = 0
        for time in rounds_times[bell][:]:
            
            if time > tmin and time < tmax:
                #print(bell, time)
                t0 = time - left
                t1 = time + right
                cut_0_min = int(t0*fs) - cut_inds//2
                cut_0_max = int(t0*fs) + cut_inds//2
            
                trans0 = transform(fs, norm[cut_0_min:cut_0_max])
    
                cut_1_min = int(t1*fs) - cut_inds//2
                cut_1_max = int(t1*fs) + cut_inds//2
            
                trans1 = transform(fs, norm[cut_1_min:cut_1_max])
    
            #plt.plot(trans1-trans0)
            alldiff = alldiff + np.maximum(0.0, trans1- trans0)
            

            count += 1
    
        #for freq_int in freq_ints:
        #    plt.plot([freq_ints, freq_ints], [0,1000])

        
        #plt.xlim(0,1000)
        #plt.title((bell, time))
        #plt.xlim(100,450)
        #plt.ylim(0,1000)

        #plt.show()
        alldiff = np.maximum(0.0, alldiff/count)
        
        alldiff[:cutmin] = 0.0
        alldiff[cutmax:] = 0.0

        if True:
            for freq_int in freq_ints:
                plt.plot([freq_ints, freq_ints], [0,400])
        
            plt.plot(alldiff/count, c = 'black')
            plt.xlim(0,1000)
            plt.title(bell)
            plt.show()
        
        all_freqs[bell] = alldiff
        
    all_freqs = np.array(all_freqs)
    
    #plt.plot(all_freqs[bell])
    return all_freqs
      

def plotamps(ts, amps, strikes, bell, xmin = 0, xmax = 30):
    plt.plot(ts, amps[bell])
    fact = len(ts)/ts[-1]
    ampsplot = (np.array(strikes[bell])*fact).astype('int')
    plt.scatter(strikes[bell],amps[bell][ampsplot], c = 'black')
    plt.xlim(xmin, xmax)
    plt.title(bell)
    plt.show()
    
def plot_freq(toplot, fs, freqs_reference, title = 0):
    #Plots the frequency profile with the bell references
    fig = plt.figure(figsize = (5,5))
    freq_ints = np.array((len(toplot))*2.0*freqs_reference/fs).astype('int')   #Integer values for the correct frequencies

    xs = np.linspace(0,1,len(toplot))*fs
    for freq in freq_ints:
        plt.plot([freq,freq], [0,1])
    plt.plot(toplot, c= 'black')
    plt.title(title)
    plt.xlim(0,max(freq_ints)*1.2)
    plt.show()
    
def plot_log(ts, log, title = 0, strikes = [], mags = []):
    fig = plt.figure(figsize = (5,5))

    plt.plot(ts, log)
    for i, strike in enumerate(strikes):
        plt.scatter([strike], [0], s = mags[i]*100)
    plt.title(title)
    plt.show()
    
def update_frequencies(norm, init_freqs, cut_length, t_end, k = 0, rounds_times = []):
    #Run through each bell (the whole time) and try to improve freqency profile as it goes
    #t_end = 180
    print('Update number', k)
    cut_start = 0; cut_end = int(cut_length*fs)
    all_peaks = []; ts = []; base = []
    
    logs = [[] for _ in range(len(init_freqs))]

    count = 0
    #If have definite rounds times, use these as a basis.
    while cut_end/fs < t_end:
        #if count%50 == 49:
         #   print('Analysing, t = ', cut_start/fs)
        trans = transform(fs, norm[cut_start:cut_end])
        
        cut_start = cut_start + int(dt*fs)
        cut_end = cut_start + int(cut_length*fs)
    
        count += 1
        
        peaks, _ = find_peaks(trans)
        prominences = peak_prominences(trans, peaks)[0]
                
        ts.append((cut_start + cut_end)/(2*fs))
        
        for bell in range(len(init_freqs)):
            #Convolve with the new frequency data
            conv = trans*init_freqs[bell]
            
            logs[bell].append(np.sum(conv)/np.sum(trans))
            #logs[bell].append(sum(trans[freqmin:freqmax]))
    
                
    if k < 1:
        allstrikes, allmags = find_strikes(ts, logs, doplot = False, init = True)
    else:
        allstrikes, allmags = find_strikes(ts, logs, doplot = False, init = False)

    alltimes = [[] for _ in range(len(init_freqs))]   #Verified strike times
    alltimes = rounds_times.copy()
    
    for bell in range(12):
        plot_log(ts, logs[bell], title = bell, strikes = allstrikes[bell], mags= allmags[bell])

    error = 0.3
    for bell in range(len(init_freqs)):
        if len(rounds_times[bell]) == 0:
            #First strike
            if bell == 0:
                tmax = sorted(allstrikes[bell])[1] + error
                tmin = sorted(allstrikes[bell])[1] - error
                for i, strike in enumerate(allstrikes[bell]):
                    if strike < tmax and strike > tmin:
                        if allmags[bell][i] > 0.1*max(allmags[bell]):
                        #This is probably it...
                            alltimes[bell].append(strike)
                            break  
                        
                print('First strike found', strike)
            else:
                tmax = alltimes[bell -1][-1] + error*2
                tmin = alltimes[bell -1][-1] - error/2
                for i, strike in enumerate(allstrikes[bell]):

                    if strike < tmax and strike > tmin:
                        if allmags[bell][i] > 0.1*max(allmags[bell]):
                        #This is probably it...
                            alltimes[bell].append(strike)
                            break      
    print('Initial rounds found')
    print('Rounds times', alltimes)

    dbell = sorted(allstrikes[0])[1] - sorted(allstrikes[0])[0]
          
    if k < 1:
        error = 0.4
    else:
        error = 0.6
    track = []
    for bell in range(len(init_freqs)):
        #Find the rest of the strikes with this current frequency
        if True:   #Find things sensibly and make sure the rounds are correct
            go = True
            while go:
                #Subsequent strikes
                tmax = alltimes[bell][-1] + dbell + error
                tmin = alltimes[bell][-1] + dbell - error
                strikes = []
                if bell == 2:
    
                    print(tmin, tmax)
    
                for i, strike in enumerate(allstrikes[bell]):
                    if strike < tmax and strike > tmin:
                        if bell == 2:
                            print(tmin, tmax, strike, allmags[bell][i], max(allmags[bell]))
    
                        if k < 1:
                            if allmags[bell][i] > 0.2*max(allmags[bell]):
                                strikes.append(strike)
                        else:
                            if allmags[bell][i] > 0.2*max(allmags[bell]):
                                strikes.append(strike)
                        #This is probably it...
                if len(strikes) == 1:
                    alltimes[bell].append(strikes[0])
                else:
                    go = False
        else:
            #Subsequent strikes
            alltimes[bell] = alltimes[bell][:20]
            tmax = 1e6
            tmin = alltimes[bell][-1] + dbell - error
            strikes = []

            for i, strike in enumerate(allstrikes[bell]):
                if strike < tmax and strike > tmin:

                    alltimes[bell].append(strike)

                    if allmags[bell][i] > 0.9*max(allmags[bell]):
                        alltimes[bell].append(strike)
                    #This is probably it...
                    
        track.append(len(alltimes[bell]))
        
    left = 0.2; right = 0.0   #Time indices to check before and after the 'peak'. Alter as necessary as not done empirically.

    nbells = len(alltimes)
    #0.3 WORKS
    cut_inds = int(cut_length*fs)   #Length of the fourier cut
    all_freqs = np.zeros((nbells, (cut_inds)//2))
    freq_ints = np.array((cut_inds)*1.0*freqs/fs).astype('int')   #Integer values for the correct frequencies
    cutoff = int(0.8*freq_ints[-1])

    for bell in range(nbells):
        
        cutmin = int(0.8*freq_ints[bell]); cutmax = int(2*freq_ints[bell])

        cutmin = 0; cutmax = 4000
        alldiff = np.zeros((cut_inds)//2)

        count = 0
        for time in alltimes[bell][:]:
            
            #print(bell, time)
            t0 = time - left
            t1 = time + right
            cut_0_min = int(t0*fs) - cut_inds//2
            cut_0_max = int(t0*fs) + cut_inds//2
        
            trans0 = transform(fs, norm[cut_0_min:cut_0_max])

            cut_1_min = int(t1*fs) - cut_inds//2
            cut_1_max = int(t1*fs) + cut_inds//2
        
            trans1 = transform(fs, norm[cut_1_min:cut_1_max])

            alldiff = alldiff + (trans1 - trans0)
            count += 1
    
        alldiff[:cutmin] = 0.0
        alldiff[cutmax:] = 0.0
        alldiff = np.maximum(0.0, alldiff/count)
        
        all_freqs[bell] = alldiff
        
    all_freqs = np.array(all_freqs)
    
    print('Times determined, successes:')
    print(track, sum(track))
    
    if False:
        track_old = np.load('track.npy')
    
        print('Previous:')
        print(track_old, sum(track_old))

        for bell in range(nbells):
            if track_old[bell] > track[bell]:
                all_freqs[bell] = init_freqs[bell]  #Don't update
    
    #Use these new 'alltimes' to update the freqencies and feed back
    np.save('all_freqs.npy', all_freqs)
    np.save('track.npy', track)
    
    minbell = 1e6
    for bell in range(nbells):
        minbell = min(minbell, sorted(alltimes[bell])[-1])
        
    return alltimes
   

def do_rounds_etc(fs, norm, dt, cut_length, all_freqs):

    #Finds the rounds properly and plots. Just needs the frequencies
    find_bell_amps(fs,norm, dt, cut_length, all_freqs)
    
    logs = np.load('logs.npy')
    ts = np.load('ts.npy')
    base = np.load('base.npy')
    
    
    #plotamps(ts, logs, allstrikes, 11)
    
    allstrikes, allmags = find_strikes(ts, logs, doplot = False, init = False)
    
    #for bell in range(12):
    #    plotamps(ts, logs, allstrikes, bell, xmin = 0, xmax = 30)
    
    allstrikes, allmags = find_strikes(ts, logs,  doplot = True, init = False)
    
    rounds_times, allrows, allcerts  = determine_rhythm(allstrikes, allmags)
    
    for row in allrows:
        plt.scatter(np.linspace(1,len(row),len(row)), row)
    
    plt.ylabel('Time')
    plt.xlabel('Bells')
    plt.ylim(0,30)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('rounds.png')
    plt.show()

    
    
cut_length= 0.15 #Time for each cut
freqs = np.array([1899,1692,1582,1411,1252,1179,1046,930,828,780,693,617])
all_freqs = determine_frequencies(freqs, cut_length)   #Outputs the frequency profiles of each bell

fs, data = wavfile.read('stockton_stedman.wav')
#fs, data = wavfile.read('bellsound_deep.wav')

tmax = 30.0
tmin = 0.0#1.5
cut = int(tmax*fs)
cut1 = int(tmin*fs)

import1 = np.array(data)[cut1:cut,0]

ts = np.linspace(0.0, len(import1)/fs, len(import1))

dt = 0.005  #Time between analyses

audio_length = len(import1)

norm = normalise(16, import1)

init_freqs = np.load('init_freqs.npy')  #Initial frequencies determined by the FIRST rounds


for bell in range(len(init_freqs)):
    plot_freq(init_freqs[bell], fs, freqs, title = bell)
rounds_times = [[] for _ in range(len(freqs))]

t_stop = tmax - 1.0
if True:
    rounds_times = update_frequencies(norm, init_freqs, cut_length, t_end = tmax, rounds_times = rounds_times)   
    
    for bell in range(len(freqs)):
        print('Latest strike bell', bell, max(rounds_times[bell]))
        
    for k in range(0):
        #t_stop = t_stop + 20.0
    
        for bell in range(len(init_freqs)):
            plot_freq(init_freqs[bell], fs, freqs, title = (k, bell))
    
        print(init_freqs[bell])
        allfreqs = np.load('all_freqs.npy')
        
        for bell in range(len(init_freqs)):
            plot_freq(all_freqs[bell], fs, freqs, title = (k, bell))

        rounds_times = update_frequencies(norm, allfreqs, cut_length, t_end = t_stop, k = k, rounds_times = rounds_times)   
        for bell in range(len(freqs)):
            print('Latest strike bell', bell, max(rounds_times[bell]))


#np.save('all_freqs.npy', init_freqs)

    
all_freqs = np.load('all_freqs.npy')

do_rounds_etc(fs, norm, dt, cut_length, all_freqs)
#print(np.shape(all_freqs))





#def determine_rhythm(allstrikes, allmags):
    #Whittle down which are the correct strikes using the rhythm of the rounds. T

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
