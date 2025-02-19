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
cut_length= 0.1 #Time for each cut
freqs_ref = np.array([1899,1692,1582,1411,1252,1179,1046,930,828,780,693,617])

fs, data = wavfile.read('audio/stockton_roundslots.wav')

tmax = 60.0
tmin = 0.0#1.5
cut = int(tmax*fs)
cut1 = int(tmin*fs)

import1 = np.array(data)[:cut,0]

ts = np.linspace(0.0, len(import1)/fs, len(import1))

dt = 0.005  #Time between analyses

audio_length = len(import1)

norm = normalise(16, import1)

#Things are imported -- find initial bell amplitudes from the reference frequencies. Will be a bit rubbish.

ts, logs = find_bell_amps(fs,norm, dt, cut_length, freqs_ref, freqs_ref)   #Find INITIAL profile based on frequency guess

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




#np.save('all_freqs.npy', new_freqs)


#np.save('all_freqs.npy', init_freqs)

    
#all_freqs = np.load('all_freqs.npy')

#do_rounds_etc(fs, norm, dt, cut_length, all_freqs)
#print(np.shape(all_freqs))





#def determine_rhythm(allstrikes, allmags):
    #Whittle down which are the correct strikes using the rhythm of the rounds. T


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
