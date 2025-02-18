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

def find_bell_amps(fs,norm, dt, cut_length, all_freqs, freqs_ref):
    #Outputs logs of the amplitudes around the specified bell frequencies.
    
    count = 0
    cut_start = 0; cut_end = int(cut_length*fs)
    all_peaks = []; ts = []; base = []
    
    logs = [[] for _ in range(len(all_freqs))]

    nbells = len(all_freqs)

    if np.size(all_freqs) == nbells:
        all_freqs = [np.zeros((cut_end - cut_start)//2) for _ in range(nbells)]
        freq_ints = np.array((cut_end - cut_start)*1.0*freqs_ref/fs).astype('int')   #Integer values for the correct frequencies

        for bell in range(nbells):
            all_freqs[bell][freq_ints[bell]-2:freq_ints[bell]+2] = 1.0
            all_freqs[bell] = all_freqs[bell]/np.sum(all_freqs[bell]) 
    else:
        all_freqs = all_freqs

    while cut_end < len(norm):
        if count%50 == 49:
            print('Analysing, t = ', cut_start/fs)
        trans = transform(fs, norm[cut_start:cut_end])
        
        cut_start = cut_start + int(dt*fs)
        cut_end = cut_start + int(cut_length*fs)
    
        count += 1
        
        ts.append((cut_start + cut_end)/(2*fs))
        
        for bell in range(len(all_freqs)):
            #Convolve with the frequency data
            conv = trans*all_freqs[bell]
            
            logs[bell].append(np.sum(conv)/np.sum(trans))
            #logs[bell].append(sum(trans[freqmin:freqmax]))
            
    for bell in range(nbells):
        plot_log(ts, logs[bell], title = ('init', bell))
        
    return ts, logs

def find_strikes(ts, logs, doplot = False, upness_flag = True):

    print('Finding strike times...')
    allstrikes = []; allmags = []
    
    
    nbells = len(logs)
    
    if doplot:
        fig = plt.figure(figsize = (5,10))

    
    for bell in range(len(logs)):#range(test_bell, test_bell + 1):#len(freqs)):
        
        
        logs_smooth = gaussian_filter1d(logs[bell],10)
        #Find strike times
        strikes = []; mags = []
        
        #Either do the upness to determine strikes, or just the absolute values.
        if upness_flag:
                
            nprev = int(0.2/dt)
                
            upness = np.zeros(len(logs_smooth))
            for n in range(0, len(logs_smooth)):
                if n < nprev:
                    upness[n] = 0.
                else:
                    upness[n] = logs_smooth[n]/(np.mean(logs_smooth[n - nprev:n-nprev//2]))
    
            peaks, _ = find_peaks(upness)
            prominences = peak_prominences(upness, peaks)[0]
            
        else:
            peaks, _ = find_peaks(logs_smooth)
            prominences = peak_prominences(logs_smooth, peaks)[0]
        
        #plot_log(ts, logs_smooth, title = bell, strikes = peaks, mags = prominences)

        strike_times = np.array([val for _, val in sorted(zip(prominences, peaks), reverse = True)]).astype('int')
        prominences = sorted(prominences, reverse = True)

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
    
def determine_rhythm(allstrikes, allmags, rounds = True):
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
    
    tenor_strikes = tenor_strikes[1:]
    #Get an even number
    if len(tenor_strikes)%2 == 0:
        tenor_strikes = tenor_strikes[:-1]
    
    print('Tenor', tenor_strikes)
    print('Diffs', tenor_strikes[1:] - tenor_strikes[:-1])

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
    
    print('Time for two changes', tboth)

    
    allrows = np.array([current_row])
    allcerts = [np.ones((len(current_row)))]
   
    current_certs = np.zeros(nbells)
    bellcerts = [[1.0] for _ in range(nbells)]

    if rounds:
        deviation = 0.75*blow_deviation
    else:
        deviation = 3*blow_deviation
        
    while np.sum(miscount) < 4:
        handstroke = not(handstroke)
        last_end = max(current_row)
        #Find starts and end of each row
        if handstroke:
            predictstart = sorted(current_row)[1] + thand - deviation
            predictend = sorted(current_row)[nbells-2] + thand + deviation
        else:
            predictstart = sorted(current_row)[1] + tback - deviation
            predictend = sorted(current_row)[nbells-2] + tback + deviation
            
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
            
            startbell = predict - deviation*(1.0 + 2*miscount[bell])
            endbell = predict + deviation*(1.0 + 2*miscount[bell])
            #startbell = max(predict - blow_deviation*(2*miscount[bell] + 2.5), predictstart)
            #endbell = min(predict + blow_deviation*(2*miscount[bell] + 2.5), predictend)
            
            #startbell = last_end - blow_deviation*2
            #endbell = last_end + thand*1.25
                      
            #startbell = max(startbell, current_row[bell] + 1.0)
            
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

        #print_row(current_row)
    allcerts = 0
    for bell in range(12):
        #print('Clarity for bell', bell + 1, np.mean(bellcerts[bell]))
        allcerts += np.mean(bellcerts[bell])
    print('Total clarity', allcerts/12)
        #print('misses', miscount)
    #print(allrows)
        #Recalculate speeds based on the last few rows? Don't bother for now...
    
    #print(tenor_strikes)
    return rounds_times, allrows, bellcerts



def determine_frequencies(fs, norm, dt, cut_length, freqs, freqs_ref, rounds_times, allcerts, upness_flag = False):
    
    #Using the given rounds
    #fs, data = wavfile.read('stockton_roundslots.wav')    
    
    cut_start = 0; cut_end = int(cut_length*fs)
    
    ts = np.linspace(0.0, len(norm)/fs, len(norm))

    count = 0
    nbells = len(freqs)
    
    alllog =[[] for _ in range(nbells)]
    tlog = []
    #Do Fourier transforms to get frequencies
    cut_start = int(0.0*fs)
    cut_end = cut_start + int(cut_length*fs)
    
    freq_length = int((cut_end - cut_start)//2)
    
    meshfreqs = [] 
    freq_ints = np.array((cut_end - cut_start)*1.0*freqs_ref/fs).astype('int')   #Integer values for the correct frequencies

    if np.size(freqs) == nbells:
        all_freqs = [np.zeros((cut_end - cut_start)//2) for _ in range(nbells)]

        for bell in range(nbells):
            all_freqs[bell][freq_ints[bell]-1:freq_ints[bell]+2] = 1.0
            all_freqs[bell] = all_freqs[bell]/np.sum(all_freqs[bell]) 
    else:
        all_freqs = freqs
    
    while cut_end < len(norm):
        if count%500 == 1:
            print('Analysing, t = ', cut_start/fs)

        trans = transform(fs, norm[cut_start:cut_end])
        
        t_centre = (ts[cut_start] + ts[cut_end])/2
        
        cut_start = cut_start + int(dt*fs)
        cut_end = cut_start + int(cut_length*fs)
        tlog.append(t_centre)
        
        meshfreqs.append(trans)
        
        for bell in range(nbells):

            conv = trans*all_freqs[bell]
            
            alllog[bell].append(np.sum(conv)/np.sum(trans))

            #alllog[bell].append(sum(trans[freq_ints[bell]-1:freq_ints[bell]+2]))



        #plot_freq(trans, fs, freqs, title = t_centre)
        count += 1
        
        
    for bell in range(nbells):
        plot_log(tlog, alllog[bell], title = ('new', bell))


    meshfreqs = np.array(meshfreqs)
        
    alllog = np.array(alllog)
    
    #Get new frequency spectrum from this log, somehow...     
    #Strike time definitely isn't loudes time (in this case at least). Very fast uptick over one or two time periods should make it obvious
    tbefore = 0.1
    nprev = int(tbefore/dt)
    tlog = np.array(tlog)


    all_new_freqs = np.zeros((nbells, freq_length))
    
    for bell in range(nbells):
        logs_smooth = gaussian_filter1d(alllog[bell],10)

        upness = np.zeros(len(logs_smooth))
        for n in range(0, len(logs_smooth)):
            if n < nprev:
                upness[n] = 0.
            else:
                upness[n] = logs_smooth[n]/(np.mean(logs_smooth[n - nprev:n-nprev//2]))

        #Find 'frequencies' for each one based on these peaks. 
        
        if upness_flag:
            peaks, _ = find_peaks(upness[:])
            prominences = peak_prominences(upness[:], peaks)[0]
            widths, heights, leftips, rightips = peak_widths(upness, peaks, rel_height=0.25)
        else:
            peaks, _ = find_peaks(logs_smooth)
            prominences = peak_prominences(logs_smooth, peaks)[0]
            widths, heights, leftips, rightips = peak_widths(logs_smooth, peaks, rel_height=0.25)


        
        #Sort based on heights?
        strike_times = np.array([val for _, val in sorted(zip(prominences, peaks), reverse = True)]).astype('int')
        leftips = np.array([val for _, val in sorted(zip(prominences, leftips), reverse = True)]).astype('int')
        widths = np.array([val for _, val in sorted(zip(prominences, widths), reverse = True)]).astype('int')

        prominences = sorted(prominences, reverse = True)
        #Filter out based on the given times and the number of blows
        nstrikes = 4*len(rounds_times[bell])
        strike_times = strike_times[:nstrikes] 
        leftips = leftips[:nstrikes] 

        
        def timetoind(time):
            dt = tlog[1] - tlog[0]
            return int((time - tlog[0])/dt)
            
        rounds_inds = []
        for time in rounds_times[bell]:
            ind = timetoind(time)
            if ind < len(tlog):
                rounds_inds.append(ind)
            
        
        if len(strike_times) > 0:

            #print(bell, strike_times, leftips[:len(strike_times)], widths[:len(strike_times)])
            
            #plot_log(tlog, logs_smooth, title = bell, strikes = strike_times, mags = prominences)
            #plot_log(tlog, logs_smooth, title = bell, strikes = rounds_inds, mags = allcerts[bell])

            #plot_log(tlog, alllog[bell], title = bell, strikes = strike_times, mags = prominences)
            #plot_log(tlog, upness, title = bell, strikes = strike_times, mags = prominences)
            #plot_log(tlog, upness, title = bell, strikes = leftips, mags = prominences)
            
            pass                
    
        
        #Establish 'new' frequency profile based on these times
        
        
        new_freqs = np.zeros((freq_length))
        rounds_number = 0   #Count once each one is confirmed
        tolerance = int(0.3/dt) #Largeish peak nearby is necessariy nearby
        
        print('Tol', tolerance)
        
        freq_tests = []
        
        for i, rounds_strike in enumerate(rounds_inds):
            #Run through rounds inds and see what we get
            best_mag = 0;
            best_strike = 0
            best_left = 0
            for j, strike in enumerate(strike_times):
                
                if abs(strike - rounds_strike) < tolerance:
                    if prominences[j] > best_mag:
                        best_strike = strike
                        best_mag = prominences[j]
                        best_left = leftips[j]
                        
            if best_mag > 0:
                print('Strike found', bell, rounds_strike, tlog[best_strike], tlog[best_left], best_strike - rounds_strike, best_mag)
            #base = np.mean(meshfreqs[strike - nprev:strike-nprev//2], axis = 0)
            #peak = meshfreqs[strike]
            #peak = np.mean(meshfreqs[strike:strike+nprev//2], axis = 0)
            
                base = np.mean(meshfreqs[best_strike-nprev:best_strike-nprev//2, :], axis = 0)
                peak = meshfreqs[best_strike]
                new_freqs += np.maximum(peak-base,0.)
                freq_tests.append(peak-base)
                
                if bell in []:
                    if i < len(rounds_inds) - 1:
                        plot_freq(peak-base, fs, freqs_ref, title = (bell, strike), end = False)
                    else:
                        plot_freq(peak-base, fs, freqs_ref, title = (bell, strike), end = True)
            
                
        
        #Run through the frequency tests and determine the consistent ones -- there is quite some variation
        freq_tests = np.array(freq_tests)
        threshold = 0
        for ti, test in enumerate(freq_tests):
            threshold += np.percentile(np.maximum(test[:int(freq_ints[-1]*0.85)],0),0.0)
            if False:
                if ti < len(freq_tests) - 1:
                    plot_freq(test, fs, freqs_ref, title = ('a', bell, strike), end = False)
                else:
                    plot_freq(test, fs, freqs_ref, title = ('a', bell, strike), end = True)
                    
        if len(freq_tests) > 0:
            
            print(np.shape(freq_tests))

            threshold = threshold/len(freq_tests)
            print('Threshold', bell, threshold)

            new_freqs = np.zeros((freq_length))
            for n in range(len(freq_tests[0])):
                if np.percentile(freq_tests[:,n], 25) > threshold:
                    new_freqs[n] = np.sum(freq_tests[:,n])/len(freq_tests)
                else:
                    new_freqs[n] = 0.0
            #plot_freq(test, fs, new_freqs, title = ('b', bell, strike), end = False)
            #plt.plot(new_freqs)
            #plt.xlim(0,600)
            #plt.title(bell)
            #plt.show()
            
            new_freqs = np.maximum(new_freqs, 0.)
            new_freqs[:int(freq_ints[-1]*0.85)] = 0.0
            #new_freqs[int(freq_ints[0]*1.2):] = 0.0

            new_freqs = new_freqs/np.sum(new_freqs)
            plot_freq(new_freqs, fs, freqs_ref, title = ('total', bell))
        
        all_new_freqs[bell] = new_freqs
        
    return all_new_freqs

def initial_analysis(fs,norm, dt, cut_length, nominal_freqs):
    #Takes normalised wave vector, and does some fourier things
    cut_start = 0; cut_end = int(cut_length*fs)
    
    nbells = len(nominal_freqs)

    freq_ints = np.array((cut_end - cut_start)*1.0*nominal_freqs/fs).astype('int') + 1   #Integer values for the correct frequencies. One hopes.

    count = 0
    ts = []
    allfreqs = []
    
    while cut_end < len(norm):
        if count%50 == 49:
            print('Analysing, t = ', cut_start/fs)
        trans = transform(fs, norm[cut_start:cut_end])
        
        cut_start = cut_start + int(dt*fs)
        cut_end = cut_start + int(cut_length*fs)
    
        count += 1
        
        ts.append((cut_start + cut_end)/(2*fs))
        
        allfreqs.append(trans)
        
    print(np.array(allfreqs).shape)
    print(freq_ints)
    
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
    print('Nominals', nominal_freqs)
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
    print(start_best, end_best)

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
            print('Bell', bell, 'time', peaks[peak_id])
            
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
    
def reinforce_frequencies(fs, norm, dt, cut_length, all_strikes, all_confidences):
    #Use confident picks to get better values for the frequencies... Theoretically. 
    #Similar to the initial frequency finder.
    
    return
    
    
    

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

if False:
    #Using crude initial analysis, find bell frequencies
    bell_frequencies, first_strikes = initial_analysis(fs, norm[cutmin:cutmax], dt, cut_length, nominal_freqs)

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

bell_frequencies = reinforce_frequencies(fs, norm, dt, cut_length, all_strikes, all_confidences)
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


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
