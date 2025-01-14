#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 15:39:03 2025

@author: trcn27
"""


def individual_bells(freqs, cut_length, freqs_ref):
    #Takes the individual bell recordings and finds the respective profiles
    fs, data = wavfile.read('audio/bells_individual.wav')
    #fs, data = wavfile.read('stockton_roundslots.wav')

    import1 = np.array(data)[:,0]

    ts = np.linspace(0.0, len(import1)/fs, len(import1))
    
    dt = 0.005  #Time between analyses
    
    audio_length = len(import1)
    
    norm = normalise(16, import1)
    cut_start = 0; cut_end = int(cut_length*fs)
    

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
        if count%1000 == 1:
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
        

    meshfreqs = np.array(meshfreqs)
        
    alllog = np.array(alllog)
    
    #Get new frequency spectrum from this log, somehow...     
    #Strike time definitely isn't loudes time (in this case at least). Very fast uptick over one or two time periods should make it obvious
    tbefore = 0.1
    nprev = int(tbefore/dt)
    tlog = np.array(tlog)

    #Need range where we know the bell actually strikes to get the data. 
    bell_ranges = np.zeros((nbells, 3))   #Specify this manually for now
    #Time ranges in integers and number of strikes in the range
    bell_ranges[2] = np.array([5.0,20.0,5.0])
    bell_ranges[3] = np.array([25.0,35.0,3.0])
    bell_ranges[6] = np.array([50.0,80.0,6.0])

    all_new_freqs = np.zeros((nbells, freq_length))
    
    for bell in range(nbells):
        logs_smooth = gaussian_filter1d(alllog[bell],2)

        upness = np.zeros(len(logs_smooth))
        for n in range(0, len(logs_smooth)):
            if n < nprev:
                upness[n] = 0.
            else:
                upness[n] = logs_smooth[n]/(np.mean(logs_smooth[n - nprev:n-nprev//2]))

        #Find 'frequencies' for each one based on these peaks. 
        min_int = int(bell_ranges[bell][0]/dt)
        max_int = int(bell_ranges[bell][1]/dt)
        
        peaks, _ = find_peaks(upness[min_int:max_int])
        prominences = peak_prominences(upness[min_int:max_int], peaks)[0]

        widths, heights, leftips, rightips = peak_widths(upness[min_int:max_int], peaks, rel_height=0.9)
        
        #Sort based on heights?
        strike_times = np.array([val for _, val in sorted(zip(prominences, peaks), reverse = True)]).astype('int')
        leftips = np.array([val for _, val in sorted(zip(prominences, leftips), reverse = True)]).astype('int')
        widths = np.array([val for _, val in sorted(zip(prominences, widths), reverse = True)]).astype('int')

        prominences = sorted(prominences, reverse = True)
        #Filter out based on the given times and the number of blows
        strike_times = strike_times[:int(bell_ranges[bell][2])] + min_int
        leftips = leftips[:int(bell_ranges[bell][2])] + min_int

        if len(strike_times) > 0:

            print(bell, strike_times, leftips[:len(strike_times)], widths[:len(strike_times)])
            
            #plot_log(tlog, logs_smooth, title = bell, strikes = strike_times, mags = prominences)

            #plot_log(tlog, alllog[bell], title = bell, strikes = strike_times, mags = prominences)
            plot_log(tlog, upness, title = bell, strikes = strike_times, mags = prominences)
            #plot_log(tlog, upness, title = bell, strikes = leftips, mags = prominences)
            
            pass                
    
        #Establish 'new' frequency profile based on these times
        
        
        new_freqs = np.zeros((freq_length))
        
        freq_tests = []
        for i, strike in enumerate(strike_times):
            
            #base = np.mean(meshfreqs[strike - nprev:strike-nprev//2], axis = 0)
            #peak = meshfreqs[strike]
            #peak = np.mean(meshfreqs[strike:strike+nprev//2], axis = 0)
            
            base = meshfreqs[leftips[i]]
            peak = meshfreqs[strike]
            #Could do an attempt to check if ALL strikes are above a certain defined amount? Would require percentiles or the like
            new_freqs += np.maximum(peak-base,0.)
            freq_tests.append(peak-base)
            
            if bell in []:
                if i < len(strike_times) - 1:
                    plot_freq(peak-base, fs, freqs_ref, title = (bell, strike), end = False)
                else:
                    plot_freq(peak-base, fs, freqs_ref, title = (bell, strike), end = True)
                    
        #Run through the frequency tests and determine the consistent ones -- there is quite some variation
        freq_tests = np.array(freq_tests)
        threshold = 0
        for ti, test in enumerate(freq_tests):
            threshold += np.percentile(np.maximum(test[:int(freq_ints[-1]*0.85)],0),75)
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
                if np.min(freq_tests[:,n]) > threshold:
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
        
def test_rounds(norm, fs, freqs, freqs_ref, dt, cut_length):
    
    #Finds strike times in rounds and perhaps improves upon the recordings...
    
    #fs, data = wavfile.read('bells_individual.wav')

    ts = np.linspace(0.0, len(norm)/fs, len(norm))
        
    cut_start = 0; cut_end = int(cut_length*fs)
    

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
        if count%1000 == 1:
            print('Analysing, t = ', cut_start/fs)

        trans = transform(fs, norm[cut_start:cut_end])
        
        t_centre = (ts[cut_start] + ts[cut_end])/2
        
        cut_start = cut_start + int(dt*fs)
        cut_end = cut_start + int(cut_length*fs)
        tlog.append(t_centre)
        
        meshfreqs.append(trans)
        
        for bell in range(nbells):

            conv = trans*all_freqs[bell]
            

            if np.sum(trans) < 1e-6:
                alllog[bell].append(0.0)
            else:
                alllog[bell].append(np.sum(conv)/np.sum(trans))

            #alllog[bell].append(sum(trans[freq_ints[bell]-1:freq_ints[bell]+2]))
                
        #plot_freq(trans, fs, freqs, title = t_centre)
        count += 1
        
    meshfreqs = np.array(meshfreqs)
    
    
    alllog = np.array(alllog)
    
    #Get new frequency spectrum from this log, somehow...     
    #Strike time definitely isn't loudes time (in this case at least). Very fast uptick over one or two time periods should make it obvious
    tbefore = 0.2
    nprev = int(tbefore/dt)
    tlog = np.array(tlog)

    #Need range where we know the bell actually strikes to get the data. 
    bell_ranges = np.zeros((nbells, 3))   #Specify this manually for now
    nstrikes = int(tlog[-1] // 2)   #Lower bound on the number of strikes
    #Time ranges in integers and number of strikes in the range

    all_new_freqs = np.zeros((nbells, freq_length))
    
    for bell in range(nbells):
        logs_smooth = gaussian_filter1d(alllog[bell],5)

        upness = np.zeros(len(logs_smooth))
        for n in range(0, len(logs_smooth)):
            if n < nprev*2:
                upness[n] = 0.
            elif np.min(logs_smooth[n - nprev*2:n-nprev//2]) < 1e-6:
                upness[n] = 0.
            else:
                upness[n] = logs_smooth[n]/(np.mean(logs_smooth[n - nprev:n-nprev//2]))

        #Find 'frequencies' for each one based on these peaks. 
        
        if True:
            peaks, _ = find_peaks(upness)
            prominences = peak_prominences(upness, peaks)[0]
        else:
            peaks, _ = find_peaks(logs_smooth)
            prominences = peak_prominences(logs_smooth, peaks)[0]

        
        widths, heights, leftips, rightips = peak_widths(upness, peaks, rel_height=0.9)
        
        strike_times = np.array([val for _, val in sorted(zip(prominences, peaks), reverse = True)]).astype('int')
        leftips = np.array([val for _, val in sorted(zip(prominences, leftips), reverse = True)]).astype('int')
        widths = np.array([val for _, val in sorted(zip(prominences, widths), reverse = True)]).astype('int')
        
        prominences = sorted(prominences, reverse = True)
        #Filter out based on the given times and the number of blows
        strike_times = strike_times[:nstrikes] 
        leftips = leftips[:nstrikes] 


        #Filter out based on the given times and the number of blows
        #strike_times = strike_times[:nstrikes]
        
    
        new_freqs = np.zeros((freq_length))

        freq_tests = []
        
        #Filter the strikes...
        #Assume the first is correct
        strikets = np.array(sorted(tlog[strike_times]))
        
        realstrikes = []
        for i, strike in enumerate(sorted(strike_times)):
            
            if i > 0:
                if tlog[strike] - tlog[realstrikes[-1]] > 2.9 or tlog[strike] - tlog[realstrikes[-1]] < 2.1:
                    #This one probably isn't fine... Ignore and move on.
                    continue
                
            base = 0.0#np.mean(meshfreqs[strike - nprev:strike-nprev//2], axis = 0)
            #peak = meshfreqs[strike]
            peak = np.mean(meshfreqs[strike:strike+nprev//2], axis = 0)
            
            #Could do an attempt to check if ALL strikes are above a certain defined amount? Would require percentiles or the like
            new_freqs += np.maximum(peak-base,0.)
            freq_tests.append(peak-base)
            
            if bell in []:
                if i < len(strike_times) - 1:
                    plot_freq(peak-base, fs, freqs_ref, title = (bell, strike), end = False)
                else:
                    plot_freq(peak-base, fs, freqs_ref, title = (bell, strike), end = True)
                    
            realstrikes.append(strike)
            
        if bell in [2,3,6]:
            #plot_log(tlog, logs_smooth, title = (bell, 'smooth'), strikes = strike_times, mags = prominences)
            #plot_log(tlog, alllog[bell], title = bell, strikes = strike_times, mags = prominences)
            plot_log(tlog, upness, title = (bell, 'up'), strikes = realstrikes, mags = 10*np.ones(len(realstrikes)))
            pass                

        print(bell, realstrikes)
        #Run through the frequency tests and determine the consistent ones -- there is quite some variation
        freq_tests = np.array(freq_tests)
        threshold = 0
        for ti, test in enumerate(freq_tests):
            threshold += np.percentile(np.maximum(test[:int(freq_ints[-1]*0.85)],0),60)
            if True:
                if ti < len(freq_tests) - 1:
                    plot_freq(test, fs, freqs_ref, title = ('a', bell, strike), end = False)
                else:
                    plot_freq(test, fs, freqs_ref, title = ('a', bell, strike), end = True)
                    
        if len(freq_tests) > 0:
            
            threshold = threshold/len(freq_tests)
            print('Threshold', bell, threshold)

            new_freqs = np.zeros((freq_length))
            for n in range(len(freq_tests[0])):
                if np.min(freq_tests[:,n]) > threshold:
                    new_freqs[n] = np.sum(freq_tests[:,n])/len(freq_tests)
                else:
                    new_freqs[n] = 0.0
            #plot_freq(test, fs, new_freqs, title = ('b', bell, strike), end = False)
            
            new_freqs = np.maximum(new_freqs, 0.)
            new_freqs[:int(freq_ints[-1]*0.85)] = 0.0
            #new_freqs[int(freq_ints[0]*1.2):] = 0.0

            new_freqs = new_freqs/np.sum(new_freqs)
            
            plot_freq(new_freqs, fs, freqs_ref, title = bell)
        
        all_new_freqs[bell] = new_freqs
        
        
    return all_new_freqs

