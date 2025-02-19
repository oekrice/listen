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

def initial_analysis(fs,norm, dt, cut_length, nominal_freqs):
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
    freq_range = 3 #Either way in integer lumps
    

    nominal_ranges = np.zeros((nbells,2),dtype = 'int')  #max and min integers to get the nominals
    for i in range(nbells):
        for ui, u in enumerate([-1,1]):
            nominal_ranges[i, ui] = int(freq_ints[i] + u*freq_range)  #Can check plot to make sure these are within. Should be for meanwood
            
            
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
        if k%6 == 0:
              plt.plot([0,2000], [peaks[k]*dt-tbefore,peaks[k]*dt-tbefore], c = 'green')
              plt.plot([0,2000], [peaks[k]*dt+tafter,peaks[k]*dt+tafter], c = 'green')
         
    plt.gca().invert_yaxis()
    plt.show()

    #__________________________________________________
    
    min_freq_int= int(freq_ints[-1]*0.9)  
    max_freq_int = int(freq_ints[0]*4)
    
    ntests = 20   #Possible prominences to test
    
    bell_frequencies = []; first_strikes = []
    
    
    nominal_min = freq_ints[-1] - 10; nominal_max = freq_ints[0] + 10
    #Run through and find the frequencies most prominent at these times? Must be a few of them. Doesn't line up well with nominals...
    
    #Run through FREQUENCIES and see which match up with increases near the bell time?
    plt.pcolormesh(diffs[:,:max_freq_int].T)
    plt.show()
    
    cs = ['blue', 'red', 'green', 'yellow', 'brown', 'pink']
    all_freqs = []; maxs = []
    
    good_freqs = []; freq_peak_array = []; sig_peak_array = []
    
    for freq_test in range(min_freq_int, max_freq_int, 1):
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
        threshold = np.percentile(diffsum,90)
        
        prominences = sorted(prominences, reverse = True)
        
        diffpeaks = diffpeaks[prominences > threshold]
        prominences = np.array(prominences)[prominences > threshold]
        
        sigpeaks = (prominences - threshold)/(max(prominences) - threshold)
        
        #Number of prominences over a theshold below the max
        
        if len(diffpeaks) in range(2, nrounds*3 + 3):
        #if True:
            sig_peak_array.append(sigpeaks.tolist())

            freq_peak_array.append(diffpeaks.tolist())
            good_freqs.append(freq_test)
            
            if freq_test == 196:
                    
                plt.plot(ts,diffsum/max(diffsum))
            
                for k in range(nbells*nrounds):
                    plt.scatter(ts[peaks[k]],-0.1, color = cs[k%nbells])
                    plt.scatter(ts[peaks[k]],-0.1, color = cs[k%nbells])
                    
                for diffpeak in diffpeaks:
                    plt.scatter(ts[diffpeak],1.0, color = 'black')
                
                plt.plot([0.0,ts[-1]],threshold*np.ones(2)/max(diffsum))
                    
                all_freqs.append(freq_test)
        
                plt.title((freq_test, np.sum(diffsum)/np.max(diffsum),len(diffpeaks)))
                plt.show()
                        
    #print(freq_peak_array)
    
    init_strikes = np.zeros((nbells, nrounds))
    
    for bell in range(nbells):
        init_strikes[bell,:] = peaks[bell::nbells]
    
    print('Init strikes', init_strikes)

    allprobs = []; best_freqs = []
    #Run through frequencies and see which are closest -- inverse square with time? 
    for fi, freq_test in enumerate(good_freqs):
        tcut = int(0.1/dt)
        probs = np.zeros(nbells)

        npeaks = len(freq_peak_array[fi])
        for bell in range(nbells):
            for peak_test in range(npeaks):
                mindist = np.min(np.abs(freq_peak_array[fi][peak_test] - init_strikes[bell]))
                prop = mindist/tcut 
                probs[bell] += sig_peak_array[fi][peak_test]*1.0/(prop + 1)**2
            probs[bell] = probs[bell]/sum(sig_peak_array[fi])
            
        if max(probs) > 0.2:
            allprobs.append(probs)
            best_freqs.append(freq_test)

    for fi, freq in enumerate(best_freqs):
        print(freq, allprobs[fi], np.where(allprobs[fi] == max(allprobs[fi]))[0][0])

    #Plot probabilities somehow
    for fi, freq in enumerate(best_freqs):
        plt.scatter(freq*np.ones(nbells), np.arange(nbells)+1, s = 100*allprobs[fi]**2)
    plt.show()

    return first_strikes, best_freqs, allprobs
    
def find_strike_times(fs,norm, dt, cut_length, best_freqs, allprobs, first_strikes, nominal_freqs, doplots = False):
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
        
        if count%50 == -1:
            print('Analysing, t = ', t)
            
        trans = transform(fs, norm[cut_start:cut_end])
            
        count += 1
        
        ts.append(t)        
        allfreqs.append(trans)
        
        t = t + dt
        
    #Make sure that this transform is sorted in EXACTLY the same way that it's done initially.
    #No tbefores etc., just the derivatives.
    
    stop
    nominal_ints = np.array((cut_end - cut_start)*1.0*nominal_freqs/fs).astype('int') + 1   #Integer values for the correct frequencies. One hopes.

    allfreqs = np.array(allfreqs)    
    
    #Run through bells and see what happens
    tbefore = 0.1  #These could theoretically be optimised
    tafter = 0.1

    error_range = 2.0*2.2/nbells #Allowable variation in speed (dictaed by ringing things)
    fd = int(0.1/dt)   #Allowable variation in frequency time (delay for hum note etc.)
    
    nrounds = 1000
    
    all_strikes = []; all_confidences = []; all_louds = []
    for bell in range(nbells):
        if bell > -1:
            doplots = True
        else:
            doplots = False
        
        bell_strikes = []; bell_confidences = []; bell_louds = []
        
        freqs = bell_frequencies[bell]
        freq_ints =  np.array(trans_length*np.array(freqs)/fs).astype('int')
        
        if doplots:
            fig, axs = plt.subplots(len(freqs), figsize = (10,10))

        all_logdiffs = []
        all_logs = []
        
        for fi, freq_test in enumerate(freq_ints):
            if doplots:
                if len(freq_ints) < 2:
                    ax = axs
                else:
                    ax = axs[fi]
                
            log = np.sum(allfreqs[:,freq_test-1:freq_test+1], axis = 1)
            log = gaussian_filter1d(log, sigma = 0.05/dt)
            logdiffs = np.zeros(len(log))
            for i in range(len(logdiffs)):
                logdiffs[i] = log[int(i + tafter*dt)]  -  log[int(i - tbefore*dt)]     
                
            #Find prominences of logdiffs in this range

            #Cut out negatives
            logdiffs[logdiffs < 0.0] = 0.0
            #logdiffdiffs = logdiffs[1:] - logdiffs[:-1]
            if doplots:
                
                ax.plot(ts,log/np.max(log))
                ax.plot(ts,logdiffs/np.max(logdiffs))
                ax.set_xlim(0.0,20.0)
                ax.set_title(freq_test)
            all_logdiffs.append(logdiffs)
            all_logs.append(log)
            
        for ri in range(nrounds):
            all_poss = []

            if ri == 0:  #Use 'first strike' time to give range
                mint = int((first_strikes[bell] - error_range)/dt)
                maxt = int((first_strikes[bell] + error_range)/dt)
    
            else:   #Use previous strike to inform this
                mint = int((bell_strikes[-1] + 2.2 - error_range)/dt)
                maxt = int((bell_strikes[-1] + 2.2 + error_range)/dt)
                
                #mint = int((bell_strikes[-1] + 0.5)/dt)
                #maxt = int((bell_strikes[-1] + 4.0)/dt)
                
                if maxt > len(logdiffs):
                    break

            for fi, freq_test in enumerate(freq_ints):
                logdiffs = all_logdiffs[fi]
                   
                #Time to log OVERALL is when the second derivative of the frequency closest to the time increases fastest
                #This gives the STRIKE time, but not when the bell is loudest (best for finding frequencies)
                
                poss_times, _ = find_peaks(logdiffs[mint:maxt], prominence = 0.2*np.max(logdiffs[mint:maxt]))
                                    
                all_poss.append(poss_times.tolist())
                if doplots:
                    
                    if len(freq_ints) < 2:
                        ax = axs
                    else:
                        ax = axs[fi]

                    ax.scatter(mint*dt, 0.0, c = 'blue')
                    ax.scatter(maxt*dt, 0.0, c = 'yellow')
                    
                bestn = 0; besttime = 0
                best_nominal_distance = 1e6
            

            for test in range(maxt-mint):
                #Find peaks which lie in this area, with confidence?
                
                #What counts as the strike time is nuanced here
                #Perhaps even could do an interpolation?
                xs = []; ys = []; zs = []   #Zs is the peak of the log (not its derivative)
                n = 0; sumtime = 0
                for fi, times in enumerate(all_poss):  #These are the times for each frequency
                    times = np.array(times)
                    if any(test-fd < num < test+fd for num in times):
                        xs.append(freq_ints[fi]); ys.append(times[(times > test-fd)*(times < test+fd)][0])
                        sumtime += times[(times > test-fd)*(times < test+fd)][0]
                        loud_peaks, _ = find_peaks(all_logs[fi][ys[-1]-int(0.1/dt):ys[-1] + int(0.5/dt)])
                        if len(loud_peaks) > 0:
                            zs.append(ys[-1] + int(0.1/dt) + loud_peaks[0])
                        n += 1
                                         
                if n > bestn:
                    bestxs = xs.copy()
                    bestys = ys.copy()
                    #Search for zs
                                            
                    bestzs = zs.copy()
                    bestn = n
                    bestloud = np.sum(zs)/len(zs)
                                        
                    if np.max(bestys) - np.min(bestys) > 0.0:
                        res = linregress(bestxs, bestys)
                        besttime = res.intercept + res.slope*nominal_ints[bell]
                    else:
                        besttime = bestys[0]
                      
            if bestn/len(all_poss) < 0.5:  #Don't take this one, as it's probably wrong...
                bell_confidences.append(0.0)
                bell_strikes.append(bell_strikes[-1] + 2.2)
                bell_louds.append(bell_louds[-1] + 2.2)
                
            else:
                bell_strikes.append(dt*(besttime + mint))
                bell_confidences.append(bestn/len(all_poss))
                bell_louds.append(dt*(bestloud + mint))

        
        if doplots:

            for fi in range(len(freq_ints)):
                
                if len(freq_ints) < 2:
                    ax = axs
                else:
                    ax = axs[fi]

                ax.scatter(bell_strikes, np.zeros(len(bell_strikes)), c= 'green')
                ax.scatter(bell_louds, np.zeros(len(bell_louds)), c= 'orange')
    
            plt.suptitle('Bell %d, %d' % (bell, nominal_ints[bell]))
            plt.tight_layout()
            plt.show()
        
        all_strikes.append(bell_strikes)
        all_confidences.append(bell_confidences)
        all_louds.append(bell_louds)
        
    #Trim so there's the right amount of rounds
    min_length = 1e6
    for bell in range(nbells):
        min_length = min(min_length, len(all_strikes[bell]))
    
    for bell in range(nbells):
        all_strikes[bell] = all_strikes[bell][:min_length]
        all_confidences[bell] = all_confidences[bell][:min_length]
        all_louds[bell] = all_louds[bell][:min_length]

    all_strikes = np.array(all_strikes) 
    all_confidences = np.array(all_confidences)
    all_louds = np.array(all_louds)
   
    return all_strikes, all_louds, all_confidences

def plot_strikes(all_strikes, all_louds, all_confidences,nrows = -1):
    #Plots the things
    fig = plt.figure(figsize = (10,7))
    nbells = len(all_strikes)
    if nrows < 0:
        nrows = len(all_strikes[0])
    yvalues = np.arange(nbells) + 1
    
    for bell in range(nbells):
        plt.scatter(all_strikes[bell], yvalues[bell]*np.ones(len(all_strikes[bell])),s=all_confidences[bell]*100)
    
    for row in range(nrows):
        plt.plot(all_strikes[:,row],yvalues)
        order = np.array([val for _, val in sorted(zip(all_strikes[:,row], yvalues), reverse = False)])
        print('Strikes', row, order, sorted(all_strikes[:,row]))
        order = np.array([val for _, val in sorted(zip(all_louds[:,row], yvalues), reverse = False)])
        #print('Louds', row, order, sorted(all_louds[:,row]))


    plt.xlim(0.0,30.0)
    plt.gca().invert_yaxis()
    plt.show(fig)
    
def reinforce_frequencies(fs, norm, dt, cut_length, all_strikes, all_confidences, nominal_freqs, nchanges = 2):
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
    
    if nchanges < 0:
        nrows = len(all_strikes[0])
    else:
        nrows = nchanges
    
    min_freq_int = int(freq_ints[-1]*0.75)
    max_freq_int = int(freq_ints[0]*4)
    
    ntests = 20
    
    bell_frequencies = []; first_strikes = []
    
    tbefore = 0.1  #These could theoretically be optimised
    tafter = 0.1
        
    max_freqs = 5
    
    nominal_min = freq_ints[-1] - 10; nominal_max = freq_ints[0] + 10
    fig, axs = plt.subplots(nbells) #For frequency plots

    #Run through and find the frequencies most prominent at these times? Must be a few of them. Doesn't line up well with nominals...
    for bell in range(nbells):
        ax = axs[bell]
        freq_picks = []
        ax.set_ylim(-1,nrows)

        for row in range(nrows):
                        
            start = int(all_strikes[bell,row]/dt-tbefore/dt); end = int(all_strikes[bell,row]/dt + tafter/dt)
                      
            diff = allfreqs[end,:] - allfreqs[start,:] 
            #Cut out negatives
            diff[diff < 0.0] = 0.0
            xs = np.linspace(0.0, 1.0, len(diff))

            diff = diff*(xs)**1.75
            
            #plt.plot(diff[:max_freq_int])

            peak_freqs, _ = find_peaks(diff)

            peak_freqs = peak_freqs[peak_freqs > min_freq_int]
            peak_freqs = peak_freqs[peak_freqs < max_freq_int]
            
            prominences = peak_prominences(diff, peak_freqs)[0]

            peak_freqs = np.array([val for _, val in sorted(zip(prominences, peak_freqs), reverse = True)]).astype('int')
            prominences = sorted(prominences, reverse = True)

            #Check how many are resonable
            
            reason = np.sum([prominences > 0.2*max(prominences)])
            sorted_tests = sorted(peak_freqs[:min(ntests, reason)])
            freq_picks.append(sorted_tests)
            
            #print(peaks[:ntests])
            #ax.scatter(peak_freqs,row*np.ones(len(peak_freqs)), s = 10*np.array(prominences)**2/np.max(prominences)**2)
            ax.scatter(peak_freqs,row*np.ones(len(peak_freqs)), s = 100*np.array(prominences)/np.max(prominences))

            #Log (as a LIST) the frequencies which have consistently increased a lot here (at all rounds)
            #Needs to be consistent across everything though -- important
        
    
        fudge = 3 #Leeway either side
        confirmed_picks = []; ns = []
        nout_max = 0.0#int(nrows//5)            
            
        for freq_test in range(nominal_min, max_freq_int):
            #Find peaks which lie in this area, with confidence?
            
            #What counts as the strike time is nuanced here
            #Perhaps even could do an interpolation?
            n = 0; xs = []
            fine = True
            if freq_test >= nominal_min and freq_test <= nominal_max:
                if abs(freq_test - freq_ints[bell]) >= fudge:
                    continue
            
            for fi, freq_round in enumerate(freq_picks):  #These are the times for each frequency
                go = True
                for fi2, f_test2 in enumerate(freq_round):
                    if abs(f_test2 - freq_test) < fudge and go:
                        n += 1
                        xs.append(f_test2)
                        go = False
                    
            if n > len(freq_picks) - nout_max - 1:
                avg_test = np.sum(xs)/n
                if avg_test*fs/(cut_end-cut_start) not in confirmed_picks and len(confirmed_picks) < max_freqs:
                    
                    if len(confirmed_picks) > 0:
                        if abs(avg_test*fs/(cut_end-cut_start) - confirmed_picks[-1]) > fudge*2*fs/(cut_end-cut_start):
                            confirmed_picks.append(avg_test*fs/(cut_end-cut_start))
                            
                            #print(xs, confirmed_picks[-1], avg_test*fs/(cut_end-cut_start))
                    else:
                        confirmed_picks.append(avg_test*fs/(cut_end-cut_start))
                          
            
                
        #print(bell, freq_ints[bell], np.array(confirmed_picks)/fs*(cut_end-cut_start))
        ax.scatter(np.array(confirmed_picks)/fs*(cut_end - cut_start), (nrows-0.5)*np.ones(len(confirmed_picks)), s = 10, c = 'black', zorder = 10)

        bell_frequencies.append(confirmed_picks)   
        
    '''
    #Run through and remove those which might clash with other bells. 
    for bell1 in range(nbells):
        for f1, freq1 in enumerate(bell_frequencies[bell1]):
            flag = False
            for bell2 in range(nbells):
                if bell2 != bell1:
                    for f2, freq2 in enumerate(bell_frequencies[bell2]):
                        if abs(freq1 - freq2) < 10.0*fs/(cut_end-cut_start) and freq1 > 0.0 and freq2 > 0.0:
                            bell_frequencies[bell2][f2] = -1.
                            flag = True
            if flag:
                bell_frequencies[bell1][f1] = -1.
                
    for bell1 in range(nbells):
        fi = 0
        while fi < len(bell_frequencies[bell1]):
            if bell_frequencies[bell1][fi] < 0:
                bell_frequencies[bell1].pop(fi)
            else:
                fi += 1
    '''
    plt.tight_layout()
    plt.show()
    plt.close()

    return bell_frequencies
    
    
    

cut_length= 0.1 #Time for each cut
#freqs_ref = np.array([1899,1692,1582,1411,1252,1179,1046,930,828,780,693,617])
#nominal_freqs = np.array([1031,918,857,757,676]) #MEANWOOD
nominal_freqs = np.array([1439.,1289.5,1148.5,1075.,962.,861.])  #ST NICS

#fs, data = wavfile.read('audio/meanwood_all.wav')
fs, data = wavfile.read('audio/stedman_nics.wav')

print('Audio length', len(data)/fs)
tmax = 10.5
tmin = 0.0#1.5
cutmin = int(tmin*fs)
cutmax = int(tmax*fs)

import1 = np.array(data)[:int(60*fs),0]

ts = np.linspace(0.0, len(import1)/fs, len(import1))

dt = 0.01  #Time between analyses

audio_length = len(import1)


norm = normalise(16, import1)

dt = 0.01
cut_length = 0.1

first_strikes, best_freqs, allprobs = initial_analysis(fs, norm[cutmin:cutmax], dt, cut_length, nominal_freqs)

all_strikes, all_louds, all_confidences = find_strike_times(fs, norm[:], dt, cut_length, best_freqs, allprobs, first_strikes, nominal_freqs)

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
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
