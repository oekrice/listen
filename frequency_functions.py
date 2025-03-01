# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 17:30:49 2025

@author: eleph
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_prominences
import matplotlib.pyplot as plt

def normalise(nbits, raw_input):
    #Normalises the string to the number of bits
    return raw_input/(2**(nbits-1))


def do_frequency_analysis(Paras, Data, Audio):
    #Now takes existing strikes data to do this (to make reinforcing easier)
    #__________________________________________________
    #Takes strike times and reinforces the frequencies from this. Needs nothing else, so works with the rounds too
     
    tcut = int(Data.cadence*Paras.freq_tcut) #Peak error diminisher

    freq_tests = np.arange(0, len(Data.transform[0])//4)
    nstrikes = len(Data.strikes[0])
    allprobs = np.zeros((len(freq_tests), Paras.nbells))
    allvalues = np.zeros((len(freq_tests), len(Data.strikes[0]), Paras.nbells))
        #Try going through in turn for each set of rounds? Should reduce the order of this somewhat
    for si in range(nstrikes):
        #Run through each row
        strikes = Data.strikes[:,si] #Strikes on this row.
        certs = Data.strike_certs[:, si]
        tmin = int(np.min(strikes) - 1.0/Paras.dt); tmax = int(np.max(strikes) + 1.0/Paras.dt)
        all_peaks = []; all_sigs = []
        print('Examining row %d \r' % si)
        for fi, freq_test in enumerate(freq_tests):
            #fig = plt.figure()
            diff_slice = Data.transform_derivative[tmin:tmax,freq_test]
            diff_slice[diff_slice < 0.0] = 0.0
            diffsum = diff_slice**2
            
            diffsum = gaussian_filter1d(diffsum, Paras.freq_smoothing)
            
            smoothsum = gaussian_filter1d(diffsum, int(Paras.smooth_time/Paras.dt))
    
            peaks, _ = find_peaks(diffsum)
            
            prominences = peak_prominences(diffsum, peaks)[0]
            
            sigs = prominences/np.max(prominences)   #Significance of the peaks relative to the background flow
            
            peaks = peaks + tmin
        
            #For each frequency, check consistency against confident strikes (threshold can be really high for that -- 99%?)
            for bell in range(Paras.nbells):
                best_value = 0.0
                pvalue = certs[bell]**Paras.beta
                for pi, peak_test in enumerate(peaks):
                    tvalue = 1.0/(abs(peak_test - strikes[bell])/tcut + 1)**Paras.strike_alpha
                    best_value = max(best_value, sigs[pi]**Paras.strike_gamma_init*tvalue*pvalue)
                      
                allvalues[fi,si,bell] = best_value
                
    allprobs[:,:] = np.mean(allvalues, axis = 1)
    

    #Do a plot of the 'frequency spectrum' for each bell, with probabilities that each one is significant
    fig, axs = plt.subplots(Paras.nbells, figsize = (10,10))
    for bell in range(Paras.nbells):
        ax = axs[bell]
        ax.plot(freq_tests/Paras.fcut_length, allprobs[:,bell], label = bell)
        ax.plot([Data.nominals[bell]/Paras.fcut_length, Data.nominals[bell]/Paras.fcut_length], [0,max(allprobs[:,bell])])
        ax.set_title('Bell %d' % (bell + 1))
        if bell != Paras.nbells-1:
            ax.set_xticks([])
    plt.suptitle('Frequency Analysis')
    plt.tight_layout()

    plt.show()
   
    #Need to take into account lots of frequencies, not just the one (which is MUCH easier)
    #Do a plot of the 'frequency spectrum' for each bell, with probabilities that each one is significant
    fig, axs = plt.subplots(4,3, figsize = (15,10))

    npeaks = 1000
    final_freqs = []   #Pick out the frequencies to test
    final_freq_ints = []
    best_probs = []   #Pick out the probabilities for each bell for each of these frequencies
    #Filter allprobs nicely
    #Get rid of narrow peaks etc.
    for bell in range(Paras.nbells):
        ax = axs[bell//3, bell%3]
        
        probs_raw = allprobs[:,bell]
        probs_clean = np.zeros(len(probs_raw))   #Gets rid of the horrid random peaks
        for fi in range(1,len(probs_raw)-1):
            #probs_clean[fi] = np.min(probs_raw[fi-1:fi+2])
            probs_clean[fi] = probs_raw[fi]
            
        probs_clean = gaussian_filter1d(probs_clean, Paras.freq_filter) #Stops peaks wiggling around. Can cause oscillation in ability.
        
        ax.plot(freq_tests/Paras.fcut_length, probs_clean, label = bell, zorder = 10)
        #ax.plot(best_freqs/cut_length, probs_clean_smooth, label = bell, zorder = 5)
        
        peaks, _ = find_peaks(probs_clean)
        
        peaks = peaks[peaks > int(500*Paras.fcut_length)]  #Use nominal frequencies here?
        
        prominences = peak_prominences(probs_clean, peaks)[0]
        
        peaks = np.array([val for _, val in sorted(zip(prominences, peaks), reverse = True)]).astype('int')
           
        peaks = peaks[:npeaks]
        prominences = peak_prominences(probs_clean, peaks)[0]

        peaks = peaks[prominences > 0.25*np.max(prominences)]
        
        ax.scatter(freq_tests[peaks]/Paras.fcut_length, np.zeros(len(peaks)), c = 'red')

        final_freq_ints = final_freq_ints + peaks.tolist()
        final_freqs = final_freqs + freq_tests[peaks].tolist()

    final_freqs = sorted(final_freqs)
    final_freq_ints = sorted(final_freq_ints)
                   
    #Determine probabilities for each of the bells, in case of overlap
    #Only keep definite ones?
    for freq in final_freq_ints:
        bellprobs = np.zeros(Paras.nbells)
        for bell in range(Paras.nbells):

            freq_range = 1  #Put quite big perhaps to stop bell confusion. Doesn't like it, unfortunately.

            #Put some blurring in here to account for the wider peaks
            top = np.sum(allprobs[freq-freq_range:freq+freq_range + 1, bell])
            bottom = np.sum(allprobs[freq-freq_range:freq+freq_range + 1, :])
                            
            bellprobs[bell] = (top/bottom)**4
        
        best_probs.append(bellprobs)
        
    best_probs = np.array(best_probs)

    nfinals = 10
    
    for bell in range(Paras.nbells):
        best_probs[:,bell] = best_probs[:,bell]/np.max(best_probs[:,bell])

        ax = axs[bell//3, bell%3]
        ax.scatter(np.array(final_freq_ints)/Paras.fcut_length, -0.1*np.ones(len(final_freq_ints)), c = 'green', s = 50*best_probs[:,bell])

        #print('Confident frequency picks for bell', bell+1, ': ', np.sum(best_probs[:,bell] > 0.1), np.array(final_freq_ints)[best_probs[:,bell] > 0.1])

    for bell in range(Paras.nbells):
        #Filter out so there are only a few peaks on each one (will be quicker)
        threshold = sorted(best_probs[:,bell], reverse = True)[nfinals + 1]
        best_probs[:,bell] = best_probs[:,bell]*[best_probs[:,bell] > threshold]
    
    for bell in range(Paras.nbells):
        ax = axs[bell//3, bell%3]
        ax.plot(freq_tests/Paras.fcut_length, allprobs[:,bell], label = bell)
            
        ax.scatter(np.array(final_freq_ints)/Paras.fcut_length, -0.2*np.ones(len(final_freq_ints)), c = 'blue', s = 50*best_probs[:,bell])

        ax.set_title('Bell %d' % (bell + 1))
        if bell != Paras.nbells-1:
            ax.set_xticks([])


    plt.suptitle('Frequency Analysis After filtering')
    plt.tight_layout()

    plt.show()
 
    
    return freq_tests, allprobs
    
def find_first_strikes(Paras, Data, Audio):
    
    #Takes normalised wave vector, and does some fourier things
        
    tenor_probs = Data.strike_probabilities[-1]
    tenor_peaks, _ = find_peaks(tenor_probs) 
    tenor_peaks = tenor_peaks[tenor_peaks < Paras.rounds_tmax/Paras.dt]
    prominences = peak_prominences(tenor_probs, tenor_peaks)[0]
    
    #Test the first few tenor peaks to see if the following diffs are fine...    
    tenor_big_peaks = np.array(tenor_peaks[prominences > 0.25])  
    tenor_peaks = np.array(tenor_peaks[prominences > 0.01])  
        
    for first_test in range(4):
        first_strike = tenor_big_peaks[first_test]
              
        teststrikes = [first_strike]

        start = first_strike + 1
        end = first_strike + int(Paras.max_change_time/Paras.dt)
        
        for ri in range(Paras.nrounds_max):  #Try to find as many as is reasonable here
            #Find most probable tenor strikes
            poss = tenor_peaks[(tenor_peaks > start)*(tenor_peaks < end)]  #Possible strikes in range -- just pick biggest
            prominences = peak_prominences(tenor_probs, poss)[0]
            poss = np.array([val for _, val in sorted(zip(prominences,poss), reverse = True)]).astype('int')
            if len(poss) < 1:
                break
            teststrikes.append(poss[0])
            start = poss[0] + 1
            end = poss[0] + int(Paras.max_change_time/Paras.dt)  
        teststrikes = np.array(teststrikes)
        diffs = teststrikes[1:] - teststrikes[:-1]
        if max(diffs) - min(diffs) < int(0.5/Paras.dt):
            tenor_strikes = teststrikes
            print('Selected Tenor Strikes for Rounds', teststrikes*Paras.dt)
            break

    #Determine whether this is the start of the ringing or not... Actually, disregard the first change whatever. Usually going to be wrong...
        
    diff1s = tenor_strikes[1::2] - tenor_strikes[0:-1:2]
    diff2s = tenor_strikes[2::2] - tenor_strikes[1:-1:2]
    if np.mean(diff1s) < np.mean(diff2s):
        Paras.handstroke_first = True
    else:
        Paras.handstroke_first = False
        
    Paras.first_change_start = tenor_strikes[0]
    Paras.first_change_end = tenor_strikes[1]
    
    nrounds_test = len(tenor_strikes) - 1
    
    handstroke = Paras.handstroke_first
    
    init_aims = []; cadences = []

    for rounds in range(nrounds_test):
        #Interpolate the bells smoothly (assuming steady rounds)
        if handstroke:
            belltimes = np.linspace(tenor_strikes[rounds], tenor_strikes[rounds+1], Paras.nbells + 1)
        else:
            belltimes = np.linspace(tenor_strikes[rounds], tenor_strikes[rounds+1], Paras.nbells + 2)
            
        cadences.append(belltimes[1] - belltimes[0])
        belltimes = belltimes[-Paras.nbells:]
        
        handstroke = not(handstroke)
        
        init_aims.append(belltimes)
                
    plt.plot(tenor_probs)
    for r in range(len(init_aims)):
        plt.scatter(init_aims[r], np.zeros(Paras.nbells), c = 'red')
    plt.scatter(tenor_strikes, np.zeros(len(tenor_strikes)), c = 'black')
    plt.title('Initial rounds aims with tenor detection...')
    plt.xlim(np.min(tenor_strikes) - 50, np.max(tenor_strikes) + 50)
    plt.show()
    
    print('Attempting to find ', len(init_aims), ' rounds')
    
    cadence = np.mean(cadences)
    Data.cadence = cadence
    #Do this just like the final row finder! But have taims all nicely. Also use same confidences.

    #Obtain VERY smoothed probabilities, to compare peaks against
    
    #Use these guesses to find the ACTUAL peaks which should be nearby...
    init_aims = np.array(init_aims)
    
    strikes = np.zeros(init_aims.T.shape)
    strike_certs = np.zeros(strikes.shape)
    
    Paras.nrounds_max = len(init_aims)

    probs_raw = Data.strike_probabilities[:]
    probs_raw = gaussian_filter1d(probs_raw, Paras.strike_smoothing, axis = 1)

    tcut = 1 #Be INCREDIBLY fussy with these picks
    
    for bell in range(Paras.nbells):
        #Find all peaks in the probabilities for this individual bell
        probs_adjust = probs_raw[bell,:]**3/(np.sum(probs_raw[:,:], axis = 0) + 1e-6)**2  #Adjust for when the rounds is a bit shit
        
        peaks, _ = find_peaks(probs_adjust) 
        sigs = peak_prominences(probs_adjust, peaks)[0]
        sigs = sigs/np.max(sigs)

        for ri in range(Paras.nrounds_max): 
            #Actually find the things. These should give reasonable options
            aim = init_aims[ri, bell]
            
            poss = peaks[(peaks > aim - 1.0/Paras.dt)*(peaks < aim + 1.0/Paras.dt)]
            yvalues = sigs[(peaks > aim - 1.0/Paras.dt)*(peaks < aim + 1.0/Paras.dt)]

            scores = []
            for k in range(len(poss)):  #Many options...
                tvalue = 1.0/(abs(poss[k] - aim)/tcut + 1)**Paras.strike_alpha
                yvalue = yvalues[k]
                scores.append(tvalue*yvalue**Paras.strike_gamma_init)
                
            kbest = scores.index(max(scores))
            
            strikes[bell, ri] = poss[kbest]
            strike_certs[bell,ri] = scores[kbest]**2/np.sum(scores)**2

    strikes = np.array(strikes)
    strike_certs = np.array(strike_certs)    

    for bell in range(Paras.nbells):
                
        plt.plot(probs_raw[bell])
        
        plt.scatter(init_aims[:,bell], np.zeros(len(init_aims)), c= 'red', label = 'Predicted linear position')
        plt.scatter(strikes[bell,:], -0.1*max(probs_raw[bell])*np.ones(len(init_aims)), c= 'green', label = 'Probable pick', s = 50*strike_certs[bell,:])
        plt.legend()
        plt.xlim(np.min(strikes) - 100,np.max(strikes) + 100)
        plt.title(bell)
        plt.show()
    
    #Determine how many rounds there actually are? Nah, it's probably fine...
        
    return strikes, strike_certs
    
    

def find_strike_probabilities(Paras, Data, Audio, init = False):
    #Find times of each bell striking, with some confidence
        
           
    #Make sure that this transform is sorted in EXACTLY the same way that it's done initially.
    #No tbefores etc., just the derivatives.
    
    allprobs = np.zeros((Paras.nbells, Paras.nt))
             
    difflogs = []; all_diffpeaks = []; all_sigs = []
    #Produce logs of each FREQUENCY, so don't need to loop
    for fi, freq_test in enumerate(Data.test_frequencies):
        
        diff_slice = Data.transform_derivative[ :, freq_test - Paras.frequency_range : freq_test + Paras.frequency_range + 1]
        diff_slice[diff_slice < 0.0] = 0.0
        diffsum = np.sum(diff_slice**2, axis = 1)
        
        diffsum = gaussian_filter1d(diffsum, 5)

        diffpeaks, _ = find_peaks(diffsum)
        
        prominences = peak_prominences(diffsum, diffpeaks)[0]
        
        diffsum_smooth = gaussian_filter1d(diffsum, int(Paras.smooth_time/Paras.dt))
        
        if init:
            threshold = np.percentile(diffsum,10)   
            diffpeaks = diffpeaks[prominences > diffsum_smooth[diffpeaks]]  #This is just for plotting...
        
                
        else:
            threshold = np.percentile(diffsum,90)  #CAN change this
            sigs = prominences[prominences > diffsum_smooth[diffpeaks]]

            diffpeaks = diffpeaks[prominences > diffsum_smooth[diffpeaks]]
            sigs = sigs/diffsum_smooth[diffpeaks]

        difflogs.append(diffsum)
        all_diffpeaks.append(diffpeaks)
        
        if not init:
            all_sigs.append(sigs)
        
        if False:  #Plot for an individual frequency
            plt.plot(Data.ts,difflogs[-1])
            plt.plot(Data.ts,diffsum_smooth)
                
            for diffpeak in all_diffpeaks[-1]:
                plt.scatter(Data.ts[diffpeak],0.0, color = 'black')
                            
            plt.title((freq_test, np.sum(diffsum)/np.max(diffsum),len(diffpeaks)))
            plt.xlim(0,20)
            plt.show()
             
    if init:
        
        #The probabilities for each frequency correspnd exactly to those for each bell -- lovely
        for bell in range(Paras.nbells):  
            allprobs[bell] = difflogs[bell]/max(difflogs[bell])
    
            plt.plot(Data.ts, allprobs[bell]/max(allprobs[bell]), label = bell + 1)
            
        plt.legend()
        plt.xlim(0.0,15.0)
        plt.title('Initial probabilities of bell strikes')
        plt.show()
                
        return allprobs

    else:
        
        overall_bell_probs = np.zeros((nbells, len(diffsum)))
    
        difflogs = np.array(difflogs)
            
        doplot = False
        
        for bell in range(nbells):  
            final_poss = []; final_sigs = []; final_probs = []
            for fi, freq_test_int in enumerate(final_freq_ints):
                sigs = all_sigs[freq_test_int]/np.max(all_sigs[freq_test_int])
                if np.max(sigs) > 0.1 and best_probs[fi,bell] > 0.1: #Maybe this is harsh but it should work...         
                
                    peaks = all_diffpeaks[freq_test_int]
    
                    final_poss = final_poss + peaks.tolist()
                    final_sigs = final_sigs + sigs.tolist()
                    for k in range(len(sigs)):
                        final_probs = final_probs + [best_probs[fi,bell]]
                                                   
            if False:
                print('Finals:')
                print(final_poss)
                print(final_sigs)
                print(final_probs)
            final_poss = np.array(final_poss)
            final_sigs = np.array(final_sigs)
            final_probs = np.array(final_probs)/np.max(final_probs)
            
            alpha = 2
            beta = 0.5
            gamma = 2
            tcut = int(0.05/dt)
            
            overall_probs = np.zeros(len(diffsum))
                         
            t_ints = np.arange(len(diffsum))
            #Want number of significant peaks near the time really
                #Calculate probability at each time
            
            tvalues = 1.0/(np.abs(final_poss[:,np.newaxis] - t_ints[np.newaxis,:])/tcut + 1)**alpha
            allvalues = tvalues*final_sigs[:,np.newaxis]**beta*final_probs[:,np.newaxis]**gamma
            absvalues = np.sum([tvalues > 0.5], axis = 1)
            
            absvalues = absvalues/np.max(absvalues)
            
            allvalues = allvalues*absvalues**2.0
            
            overall_probs =  np.sum(allvalues, axis = 0)
                
            overall_probs_smooth = gaussian_filter1d(overall_probs, int(3.5/dt), axis = 0)
                
            overall_bell_probs[bell] = overall_probs/overall_probs_smooth
            
            overall_bell_probs[bell] = overall_bell_probs[bell]/np.max(overall_bell_probs)
               
        
            if doplot:
                plt.plot(ts, overall_bell_probs[bell], label = bell + 1)
            
        if doplot:
            plt.title('Probability of strike at each time')
            plt.legend()
            plt.xlim(0,20)
            plt.show()
             

        return overall_bell_probs

