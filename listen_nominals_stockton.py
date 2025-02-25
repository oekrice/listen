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
import pandas as pd
import matplotlib


cmap = matplotlib.colormaps.get_cmap('jet')

plt.style.use('default')

def normalise(nbits, raw_input):
    #Normalises the string to the number of bits
    return raw_input/(2**(nbits-1))
    
def transform(fs, norm_cut):
    #Produce the fourier transform of the input data
    trans1 = abs(fft(norm_cut)[:len(norm_cut)//2])
    return 0.5*trans1*fs/len(norm_cut)
    
def frequency_analysis(fs,norm, dt, cut_length, nominal_freqs, strikes, strike_certs):
    #Now takes existing strikes data to do this (to make reinforcing easier)
    
    #__________________________________________________
    nrows = len(strikes[0])
    
    print('Frequency testing on', nrows, 'rows')
    
    cut_start = 0; cut_end = int(cut_length*fs)
    
    nbells = len(nominal_freqs)

    freq_ints = np.round(nominal_freqs*cut_length).astype('int')  #Integer values for the correct frequencies. One hopes.
    
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
    
    min_freq_int = 10
    max_freq_int = len(allfreqs[0])//4
        
    #Run through and find the frequencies most prominent at these times? Must be a few of them. 
    
    ts = np.array(ts)
    
    allfreqs_smooth = gaussian_filter1d(allfreqs, int(0.05/dt), axis = 0)
    diffs = np.zeros(allfreqs_smooth.shape)
    diffs[1:,:] = allfreqs_smooth[1:,:] - allfreqs_smooth[:-1,:] 
    
    diffs[diffs < 0.0] = 0.0
     
    plotmin = int(strikes[0][0] - 0.5/dt)
    plotmax = int(strikes[-1][1] + 0.5/dt)
    plotfmin = int(freq_ints[-1]*0.5)
    plotfmax = int(freq_ints[0]*2.0)
    tplots = np.linspace(plotmin, plotmax, plotmax-plotmin)*dt
    fplots = np.arange(plotfmin, plotfmax)
    #Run through FREQUENCIES and see which match up with increases near the bell time?
    plt.pcolormesh(fplots, tplots, diffs[plotmin:plotmax,plotfmin:plotfmax])
    plt.gca().invert_yaxis()
    plt.title('All frequencies in initial rounds')
    plt.close()
    
        
    last_strike_time = int(np.max(strikes) + 0.5/dt)
    
    cadence = int(np.mean(strikes[1:,0] - strikes[:-1,0]))
    
    alpha = 2 #Time diminishing factor
    tcut = int(cadence/2) #Peak error diminisher
    threshold = 0.3  #Only count picks which are above this
    
    freq_tests = np.arange(min_freq_int, max_freq_int)
    #freq_tests= freq_ints
    
    allprobs = np.zeros((len(freq_tests), nbells))
    
    for fi, freq_test in enumerate(freq_tests):
        #fig = plt.figure()
        diff_slice = diffs[:last_strike_time,freq_test]
        diff_slice[diff_slice < 0.0] = 0.0
        diffsum = diff_slice**2
        
        diffsum = gaussian_filter1d(diffsum, 5)
        
        smoothsum = gaussian_filter1d(diffsum, int(cadence*2))

        diffpeaks, _ = find_peaks(diffsum)
        
        prominences = peak_prominences(diffsum, diffpeaks)[0]
        
        diffpeaks = np.array([val for _, val in sorted(zip(prominences, diffpeaks), reverse = True)]).astype('int')
        
        prominences = sorted(prominences, reverse = True)
                        
        hvalues = prominences/smoothsum[diffpeaks]   #Significance of the peaks relative to the background flow
        
        #Number of prominences over a theshold below the max (or base on underlying smoothness?). Yes, this is better.
        diffpeaks = diffpeaks[hvalues > 1.0]
        hvalues = hvalues/np.max(hvalues)

        #For each frequency, check consistency against confident strikes (threshold can be really high for that -- 99%?)
        for bell in range(nbells):
            bellstrikes = strikes[bell]

            bellstrikes = bellstrikes[strike_certs[bell] > 0.98] #Can do this earlier I suppose, but it's pretty fast
            
            allvalues = np.zeros(len(bellstrikes))
            for si, strike in enumerate(bellstrikes):
                best_value = 0.0
                for pi, peak_test in enumerate(diffpeaks):
                    tvalue = 1.0/(abs(peak_test - strike)/tcut + 1)**alpha
                    best_value = max(best_value, hvalues[pi]*tvalue)
            
                allvalues[si] = best_value
                
            allvalues = allvalues[allvalues > threshold]
            if len(allvalues) > 0:
                quality = np.mean(allvalues)*(len(allvalues)/len(bellstrikes))**2
            else:
                quality = 0.0
                    
            allprobs[fi, bell] = quality

        if max(allprobs[fi]) > 0.8:
                
            plt.plot(diffsum)
            plt.plot(smoothsum)
        
            for diffpeak in diffpeaks:
                plt.scatter(diffpeak,1.0, color = 'black')
            
            for bell in range(nbells):
                plt.scatter(strikes[bell,:], -0.05*(bell+1)*np.max(diffsum)*np.ones(len(strikes[bell,:])), s = 50*strike_certs[bell,:]**4)
                
            #plt.plot([0.0,ts[len(diffsum)]],threshold*np.ones(2)/max(diffsum))
                    
            plt.title((freq_test, max(allprobs[fi])))
            plt.close()

        #Run through and check these against bell times. Still to be determined the best way to do such a thing...
        #Ideally adding more points does not make things worse here, but it usually seems to
    
    #Do a plot of the 'frequency spectrum' for each bell, with probabilities that each one is significant
    fig, axs = plt.subplots(nbells, figsize = (10,10))
    for bell in range(nbells):
        ax = axs[bell]
        ax.plot(freq_tests/cut_length, allprobs[:,bell], label = bell)
        ax.plot([freq_ints[bell]/cut_length, freq_ints[bell]/cut_length], [0,max(allprobs[:,bell])])
        ax.set_title('Bell %d' % (bell + 1))
        if bell != nbells-1:
            ax.set_xticks([])
    plt.suptitle('Frequency Analysis')
    plt.tight_layout()

    plt.close()
    
    
    return freq_tests, allprobs
    
def find_strike_probs(fs, norm, dt, cut_length, best_freqs, allprobs, nominal_freqs, doplots = False, init = False):
    #Find times of each bell striking, with some confidence
    
    #Cut_length is the length of the Fourier transform. CENTRE the time around this
    nbells = len(allprobs[0])
    count = 0
    allfreqs = []; ts = []
    
    tmax = len(norm)/fs
    
    nominal_ints = np.round(nominal_freqs*cut_length).astype('int')

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
     
    difflogs = []; all_diffpeaks = []; all_sigs = []
    #Produce logs of each FREQUENCY, so don't need to loop
    for fi, freq_test in enumerate(best_freqs):
    #for freq_test in freq_ints:
        #fig = plt.figure()
        
        diff_slice = diffs[:,freq_test]
        diff_slice[diff_slice < 0.0] = 0.0
        diffsum = diff_slice**2
        
        diffsum = gaussian_filter1d(diffsum, 5)

        diffpeaks, _ = find_peaks(diffsum)
        
        prominences = peak_prominences(diffsum, diffpeaks)[0]
        
        diffsum_smooth = gaussian_filter1d(diffsum, int(2.0/dt))
        
        if not init:
            threshold = np.percentile(diffsum,90)  #CAN change this
            sigs = prominences[prominences > diffsum_smooth[diffpeaks]]

            diffpeaks = diffpeaks[prominences > diffsum_smooth[diffpeaks]]
            sigs = sigs/diffsum_smooth[diffpeaks]
            
        else:
            threshold = np.percentile(diffsum,10)
            diffpeaks = diffpeaks[prominences > threshold]
        
        #sigpeaks = (prominences - threshold)/(max(prominences) - threshold)
        

        if False:
            plt.plot(ts,difflogs[-1])
            plt.plot(ts,diffsum_smooth)
                
            for diffpeak in all_diffpeaks[-1]:
                plt.scatter(ts[diffpeak],0.0, color = 'black')
                            
            plt.title((freq_test, np.sum(diffsum)/np.max(diffsum),len(diffpeaks)))
            plt.xlim(0,20)
            plt.show()
            
        difflogs.append(diffsum)
        all_diffpeaks.append(diffpeaks)
        if not init:
            all_sigs.append(sigs)
        
    if not init:
        #Need to take into account lots of frequencies, not just the one (which is MUCH easier)
        #Do a plot of the 'frequency spectrum' for each bell, with probabilities that each one is significant
        fig, axs = plt.subplots(nbells, figsize = (10,10))

        npeaks = 10
        final_freqs = []   #Pick out the frequencies to test
        final_freq_ints = []
        best_probs = []   #Pick out the probabilities for each bell for each of these frequencies
        #Filter allprobs nicely
        #Get rid of narrow peaks
        fshift = best_freqs[0]
        for bell in range(nbells):
            ax = axs[bell]

            probs_raw = allprobs[:,bell]
            probs_clean = np.zeros(len(probs_raw))   #Gets rid of the horrid random peaks
            for fi in range(1,len(probs_raw)-1):
                #probs_clean[fi] = np.min(probs_raw[fi-1:fi+2])
                probs_clean[fi] = probs_raw[fi]
                
            probs_clean = gaussian_filter1d(probs_clean, 1) #Stops peaks wiggling around. Can cause oscillation in ability.
            
            ax.plot(best_freqs/cut_length, probs_clean, label = bell, zorder = 10)
            #ax.plot(best_freqs/cut_length, probs_clean_smooth, label = bell, zorder = 5)
            
            peaks, _ = find_peaks(probs_clean)
            
            peaks = np.array(peaks)
            mask = np.array(peaks > int(nominal_ints[0]) - fshift)
            mask = mask + np.array(peaks > (int(nominal_ints[bell]) - fshift)*0.95)*np.array(peaks < (int(nominal_ints[bell]) - fshift)*1.1)
            peaks = peaks[mask]  #Use nominal frequencies here
            
            prominences = peak_prominences(probs_clean, peaks)[0]
            
            peaks = np.array([val for _, val in sorted(zip(prominences, peaks), reverse = True)]).astype('int')
               
            peaks = peaks[:npeaks]
            
            prominences = peak_prominences(probs_clean, peaks)[0]

            peaks = peaks[prominences > 0.1]
            
            ax.scatter(best_freqs[peaks]/cut_length, np.zeros(len(peaks)), c = 'red')

            final_freq_ints = final_freq_ints + peaks.tolist()
            final_freqs = final_freqs + best_freqs[peaks].tolist()
            
        final_freqs = sorted(final_freqs)
        final_freq_ints = sorted(final_freq_ints)
        #Determine probabilities for each of the bells, in case of overlap
        #Only keep definite ones?
        for freq in final_freq_ints:
            bellprobs = np.zeros(nbells)
            for bell in range(nbells):
                bellprobs[bell] = allprobs[freq, bell]**2/np.sum(allprobs[freq,:])
            
            best_probs.append(bellprobs)
        
        best_probs = np.array(best_probs)
        
        for bell in range(nbells):
            ax = axs[bell]
            ax.plot(best_freqs/cut_length, allprobs[:,bell], label = bell)
            ax.set_title('Bell %d' % (bell + 1))
            if bell != nbells-1:
                ax.set_xticks([])
        plt.suptitle('Frequency Analysis After filtering')
        plt.tight_layout()

        plt.show()

          
        overall_bell_probs = np.zeros((nbells, len(diffsum)))
    
        difflogs = np.array(difflogs)

        for bell in range(nbells):  
            final_poss = []; final_sigs = []; final_probs = []
            for fi, freq_test_int in enumerate(final_freq_ints):
                if best_probs[fi,bell] < 0.05:
                    continue

                peaks = all_diffpeaks[freq_test_int]
                sigs = all_sigs[freq_test_int]/np.max(all_sigs[freq_test_int])
                
                final_poss = final_poss + peaks.tolist()
                final_sigs = final_sigs + sigs.tolist()
                for k in range(len(sigs)):
                    final_probs = final_probs + [best_probs[fi,bell]]
                
                if False:
                    plt.plot(ts, difflogs[freq_test_int,:])
                    plt.scatter(ts[peaks], np.zeros(len(peaks)), c= 'black', s = 50*sigs)
                    plt.title(freq_test_int)
                    plt.xlim(0,20)
                    plt.show()
                    
            if False:
                print('Finals:')
                print(final_poss)
                print(final_sigs)
                print(final_probs)
            final_poss = np.array(final_poss)
            final_sigs = np.array(final_sigs)
            final_probs = np.array(final_probs)
            
            alpha = 2
            beta = 1
            gamma = 4
            tcut = int(0.1/dt)
            
            overall_probs = np.zeros(len(diffsum))
                         
            t_ints = np.arange(len(diffsum))
                #Calculate probability at each time
            tvalues = 1.0/(np.abs(final_poss[:,np.newaxis] - t_ints[np.newaxis,:])/tcut + 1)**alpha
            allvalues = tvalues*final_sigs[:,np.newaxis]**beta*final_probs[:,np.newaxis]**gamma
                                
            overall_probs =  np.sum(allvalues, axis = 0)
                
            overall_probs_smooth = gaussian_filter1d(overall_probs, int(3.5/dt), axis = 0)
                
            overall_bell_probs[bell] = overall_probs/overall_probs_smooth
            
            overall_bell_probs[bell] = overall_bell_probs[bell]/np.max(overall_bell_probs)

                             
            plt.plot(ts, overall_bell_probs[bell], label = bell, c = cmap(bell/(nbells-1)))
            
        plt.title('Probability of strike at each time')
        plt.legend()
        plt.xlim(0.0,7.5)
        plt.close()
                
        return overall_bell_probs

    else:
        #This is the initial finding. Just take the unadulterated sum.
        
        overall_bell_probs = np.zeros((nbells, len(diffsum)))
    
        for bell in range(nbells):  #the 4 is the most distinctive
            overall_bell_probs[bell] = difflogs[bell]
    
            plt.plot(ts, overall_bell_probs[bell]/max(overall_bell_probs[bell]), label = bell)
            
        plt.legend()
        plt.xlim(0.0,7.5)
        plt.show()
                
        return overall_bell_probs
        
        
def find_strike_times(fs, dt, cut_length, strike_probs, start_time):
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
    probs_smooth = gaussian_filter1d(probs, int(1.0/dt))
    
    peaks = np.array(sorted(peaks[prominences > probs_smooth[peaks]]))
    
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
        probs_smooth = 0.5*gaussian_filter1d(probs, int(1.0/dt))

        peaks, _ = find_peaks(probs)
        
        peaks = peaks[peaks > start_time]
        
        prominences = peak_prominences(probs, peaks)[0]
        
        bigpeaks = peaks[prominences > 3.0*probs_smooth[peaks]]  #For getting first strikes, need to mbe more significant
        peaks = peaks[prominences > probs_smooth[peaks]]

        sigs = peak_prominences(probs, peaks)[0]/probs_smooth[peaks]
        
        sigs = sigs/np.max(sigs)
        
        peaks = np.array(sorted(peaks))
        bigpeaks = np.array(sorted(bigpeaks))
        
        bellstrikes = []; bellconfs = []; taims = []
                
        start = 0; end = 0; taim = 0
        alpha =  2.0  #Sharp cutoff
        nextend = 0
        tcut = int(avg_cadence*1.5)
            
        while nextend < np.max(peaks):
            if len(bellstrikes) == 0:  #Establish first strike
                bellstrikes.append(bigpeaks[0])
                bellconfs.append(1.0)
            else:  #Find options in the correct range
                peaks_range = peaks[(peaks > start)*(peaks < end)]
                sigs_range = sigs[(peaks > start)*(peaks < end)]
                
                if len(peaks_range) == 1:   #Only one time that it could reasonably be
                    bellstrikes.append(peaks_range[0])
                    bellconfs.append(1.0)
                elif len(peaks_range) > 1:
                                          
    
                    scores = []
                    for k in range(len(peaks_range)):  #Many options...
                        tvalue = 1.0/(abs(peaks_range[k] - taim)/tcut + 1)**alpha
                        yvalue = sigs_range[k]
                        scores.append(tvalue*yvalue)
                        
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
            start = bellstrikes[-1] + 1*int(avg_cadence)
            end  =  taim + (nbells)*int(avg_cadence)
            nextend = bellstrikes[-1] + int(avg_cadence*nbells*1.5)  #End of next change
               
            handstroke = not(handstroke)
            
        plt.scatter(peaks, -0.1*np.ones(len(peaks)), c= 'green')
        plt.scatter(taims, -0.05*np.ones(len(taims)), c= 'red')
        plt.scatter(bellstrikes, -0.15*np.ones(len(bellstrikes)), s = 50*np.array(bellconfs), c= 'black')
        plt.plot(probs)
        plt.plot(probs_smooth)
        plt.title(bell)
        plt.close()
        
        allstrikes.append(bellstrikes)
        allconfs.append(bellconfs)
        minlength = min(minlength, len(bellstrikes))
        
    for bell in range(len(allstrikes)):
        allstrikes[bell] = allstrikes[bell][:minlength]
        allconfs[bell] = allconfs[bell][:minlength]
        
    allstrikes = np.array(allstrikes)[:,:minlength]
    allconfs = np.array(allconfs)[:,:minlength]
    print('Overall confidence', np.sum(allconfs)/np.size(allconfs))

    
    return allstrikes, allconfs 


def find_first_strikes(fs, norm, dt, cut_length, strikeprobs, nrounds_max = 4):
    
    #Takes normalised wave vector, and does some fourier things
        
    print('Finding approximate first strike times in rounds...')
    nbells = len(nominal_freqs)

    tenor_probs = strikeprobs[-1]
    
    tenor_peaks, _ = find_peaks(tenor_probs)
    prominences = peak_prominences(tenor_probs, tenor_peaks)[0]

    #Sort appropriately
    tenor_peaks = np.array([val for _, val in sorted(zip(prominences,tenor_peaks), reverse = True)]).astype('int')
    prominences = sorted(prominences, reverse = True)
    
    threshold = np.percentile(tenor_probs,75)

    
    first_strike = np.min(tenor_peaks[prominences > np.percentile(tenor_probs,90)])
    
    tenor_peaks = tenor_peaks[prominences > threshold]

    tenor_strikes = [first_strike]
    print('Finding tenor strikes, first significant at time', first_strike)
    
    start = first_strike
    end = first_strike + int(3.5/dt)
    nrounds = nrounds_max
    for r in range(nrounds):
        #Find most probable tenor strikes
        poss = tenor_peaks[(tenor_peaks > start)*(tenor_peaks < end)]  #Possible strikes in range -- just pick biggest
        prominences = peak_prominences(tenor_probs, poss)[0]
        poss = np.array([val for _, val in sorted(zip(prominences,poss), reverse = True)]).astype('int')
        if len(poss) < 1:
            break
        tenor_strikes.append(poss[0])
        start = poss[0]
        end = poss[0] + int(3.5/dt)   

    tenor_strikes = np.array(tenor_strikes)
    #Determine whether this is the start of the ringing or not...
    difftenors = tenor_strikes[1:] - tenor_strikes[:-1]
    init_aims = []
    
    if first_strike > 1.25*np.mean(difftenors):
        diffavg1 = np.mean(difftenors[1::2])
        diffavg0 = np.mean(difftenors[0::2])
        if diffavg1 > diffavg0:
            handstroke = True
        else:
            handstroke = False

        print('Audio starting from the start of ringing')
        tenor_strikes = np.concatenate(([tenor_strikes[0] - difftenors[1]], tenor_strikes))
        nrounds = len(tenor_strikes)

        #Probably some silence beforehand
    else:
        #Probably not...
        print('Audio starting from mid-way through ringing')
        nrounds = len(tenor_strikes)

    if difftenors[1] > difftenors[0]:
        handstroke = True
    else:
        handstroke = False
        
    nrounds = len(tenor_strikes) - 1
    
    
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

    #Obtain VERY smoothed probabilities, to compare peaks against
    smoothprobs = gaussian_filter1d(strikeprobs, int(cadence*2), axis = 1)
    
    #Use these guesses to find the ACTUAL peaks which should be nearby...
    nrounds = len(init_aims)   #In case this is less (the audio is shorter)
    init_aims = np.array(init_aims)
    init_strikes = np.zeros(init_aims.T.shape)
    strike_certs = np.zeros(init_strikes.shape)
    
    alpha = 2
    tcut = int(cadence/5)
    for bell in range(nbells):
        bell_peaks, _ = find_peaks(strikeprobs[bell])
        for r in range(min(nrounds, nrounds_max)):
            aim = init_aims[r, bell]  #aim time
                        
            start = aim - cadence*2
            end   = aim + cadence*2

            poss = bell_peaks[(bell_peaks > start)*(bell_peaks < end)]  #Possible strikes in range -- pick biggest
            
            if len(poss) > 1:
                    
                sigs = peak_prominences(strikeprobs[bell], poss)[0]  #Significance of the possible peaks
                hvalues = sigs/smoothprobs[bell][poss]   #Significance of the peaks
                tvalues = 1.0/(abs(poss - aim)/tcut + 1)**alpha
        
                values = hvalues*tvalues
                        
                poss = np.array([val for _, val in sorted(zip(values,poss), reverse = True)]).astype('int')
                values = sorted(values, reverse = True)
                
                init_strikes[bell,r] = poss[0]
                strike_certs[bell, r] = values[0]/(values[1] + values[0])


            elif len(poss) == 1:
                init_strikes[bell,r] = poss[0]
                strike_certs[bell, r] = 1.0
                
            elif len(poss) == 0:
                print('No peak found for initial rounds... Assuming linear and guessing')
                init_strikes[bell, r] = init_aims[r, bell]
                strike_certs[bell, r] = 0.0
            
        plt.plot(strikeprobs[bell])
        plt.plot(smoothprobs[bell])
        
        plt.scatter(init_aims[:,bell], np.zeros(nrounds), c= 'red', label = 'Predicted linear position')
        plt.scatter(init_strikes[bell,:], np.zeros(nrounds), c= 'green', label = 'Probable pick')
        plt.legend()
        plt.title(bell)
        plt.show()
    
    init_strikes = init_strikes[:,:nrounds_max]
    strike_certs = strike_certs[:,:nrounds_max]
    print('Initial picks found, overall certainty:', np.sum(strike_certs)/np.size(strike_certs))
    
    return init_strikes, strike_certs
    
    

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
        print('Strikes', row, order)#, np.array(sorted(all_strikes[:,row]))*dt, strike_certs[:,row])
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

def save_strikes(strikes, dt):
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
    data.to_csv('nics_stedman.csv')  
    return
    
    
#SET THINGS UP
    
cut_length= 0.1 #Time for each cut

if True:
    fs, data = wavfile.read('audio/stedman_nics.wav')
    nominal_freqs = np.array([1439.,1289.5,1148.5,1075.,962.,861.])  #ST NICS
    rounds_cut = [0.0, 30.0]
    
else:
    fs, data = wavfile.read('audio/stockton_stedman.wav')
    nominal_freqs = np.array([1892,1679,1582,1407,1252,1179,1046,930,828,780,693,617])



print('Audio length', len(data)/fs)
tmax = 30.0

tmin = 0.0#1.5
cutmax = int(tmax*fs)

import1 = np.array(data)[:,0]

ts = np.linspace(0.0, len(import1)/fs, len(import1))

dt = 0.01  #Time between analyses

audio_length = len(import1)

norm = normalise(16, import1)

cut_length = 0.1
cut_time = len(data)/fs - 10.0

#strikes = find_first_strikes(fs, norm[:cutmax], dt, cut_length, nominal_freqs)

#Look into doing strike probabilities just from the nominals?
best_freqs = np.round(nominal_freqs*cut_length).astype('int')

allprobs = np.identity(len(best_freqs))

#Find strike probabilities from the nominals
init_strike_probabilities = find_strike_probs(fs, norm[:int(tmax*fs)], dt, cut_length, best_freqs, allprobs, nominal_freqs, init=True)
#Find probable strike times from these arrays
strikes, strike_certs = find_first_strikes(fs, norm[:int(tmax*fs)], dt, cut_length, init_strike_probabilities, nrounds_max = 8)

start_time = max(0, strikes[0][0] - int(0.1/dt))

count = 0
maxits = 10

tmax = len(data)/fs

while count < maxits:
        
    if True:
    #Find the probabilities that each frequency is useful. Also plots frequency profile of each bell, hopefully.
        print('Doing frequency analysis,  iteration number', count)
    
        allfreqs, freqprobs = frequency_analysis(fs, norm[:cutmax], dt, cut_length, nominal_freqs, strikes[:,:], strike_certs[:,:])  
        
        np.save('freqs.npy', allfreqs)
        np.save('freqprobs.npy', freqprobs)
    
    freqprobs = np.load('freqprobs.npy')
    allfreqs = np.load('freqs.npy')
    
    cutmax = int(tmax*fs)

    print('Finding strike probabilities...')
        
    strike_probabilities = find_strike_probs(fs, norm[:cutmax], dt, cut_length, allfreqs, freqprobs, nominal_freqs, init = False)
    np.save('probs.npy', strike_probabilities)

    strike_probabilities = np.load('probs.npy')
    strikes, strike_certs = find_strike_times(fs, dt, cut_length, strike_probabilities, start_time) #Finds strike times in integer space
    
    count += 1
    
    maxrows = 0; maxtime = 0.0; confidence = 1.0
    for row in range(len(strikes[0])):
        if np.max(strikes[:,row] - np.min(strikes[:,row])) < 3.0/dt:
            #Check confidences
            confidence = confidence*np.mean(strike_certs[:,row])
            if confidence > 0.00:    
                maxrows = row
                maxtime = np.max(strikes[:,row])*dt
            else:
                break
        else:
            break
        
    if maxtime > tmax - 5.0:
        maxits = min(maxits, count + 2)
    print('Number of probably correct rows: ', maxrows)
    strikes = strikes[:, :maxrows]
    strike_certs = strike_certs[:, :maxrows]
    unit_test(strikes,dt)

    plot_strikes(strikes,strike_certs,  nrows = -1)


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
