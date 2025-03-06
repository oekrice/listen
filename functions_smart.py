# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 17:30:49 2025

@author: eleph
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_prominences
import matplotlib.pyplot as plt

cmap = plt.cm.jet

def normalise(nbits, raw_input):
    #Normalises the string to the number of bits
    return raw_input/(2**(nbits-1))

def find_nominal_frequencies(Paras, Data, loudest_bell_from_back = 0):
    #Attempts to find the nominal frequencies from the audio alone...
    #Looks for loudest strikes across all frequencies which are roughly consistent and goes with that
    
    freq_int_max = int(5000*Paras.fcut_length)
    
    loudness = Data.transform[:,:freq_int_max]
    loudness = loudness
    
    loudness = gaussian_filter1d(loudness, int(0.1/Paras.dt),axis = 0)
    loudsum = np.sqrt(np.sum(loudness, axis = 1))

    loudsmooth = gaussian_filter1d(loudsum, int(2.0/Paras.dt), axis = 0)
    loudsmooth[0] = 0.0; loudsmooth[-1] = 0.0 #For checking peaks
    
    
    threshold = np.max(loudsmooth)*0.8
    #Use this to determine the start time of the ringing -- time afte
    peaks, _= find_peaks(loudsmooth, width = int(10.0/Paras.dt))  #Prolonged peak in noise - probably ringing
    
    #I can't find an inbuilt finctino to do this, bafflingly
    peak = sorted(peaks)[-1]
    rlim = peak; llim = peak
    while rlim < len(loudsmooth):
        if loudsmooth[rlim] > threshold:
            rlim = rlim + 1
        else:
            break
        
    while llim > 0:
        if loudsmooth[llim] > threshold:
            llim = llim - 1
        else:
            break
    start_time = max(0,llim - int(5.0/Paras.dt))
    end_time = min(len(loudness)-1, rlim + int(5.0/Paras.dt))
    print('Ringing range', start_time, end_time)
    
    loudpeaks, _ = find_peaks(loudsum)
    loudpeaks = loudpeaks[(loudpeaks > start_time)*(loudpeaks < end_time)]
    loudproms = peak_prominences(loudsum, loudpeaks)[0]
    loudpeaks = loudpeaks[loudproms > 0.8*np.max(loudproms)]   #Probable tenor strikes --look for frequencies around here using the established methods...
    
    tcut = int(0.2/Paras.dt*Paras.freq_tcut) #Peak error diminisher

    allvalues = np.zeros((freq_int_max, len(loudpeaks)))
    print('Aligning frequencies with volume peaks...')
    for si in range(len(loudpeaks)):
        #Run through each row
        strike = loudpeaks[si]
        #print('Examining row %d \r' % si)
        for fi, freq_test in enumerate(np.arange(0, freq_int_max)):
            #fig = plt.figure()
            diff_slice = Data.transform_derivative[:,freq_test]
            diff_slice[diff_slice < 0.0] = 0.0
            diffsum = diff_slice**2
            
            diffsum = gaussian_filter1d(diffsum, Paras.freq_smoothing)
                
            peaks, _ = find_peaks(diffsum)
            
            prominences = peak_prominences(diffsum, peaks)[0]
            
            sigs = prominences/np.max(prominences)   #Significance of the peaks relative to the background flow
                    
            #Two options here -- the first works well on 12, the second on 6. Oh well...
            best_value = 0.0; min_tvalue = 1e6
            for pi, peak_test in enumerate(peaks):
                tvalue = 1.0/(abs(peak_test - strike)/tcut + 1)**Paras.strike_alpha
                best_value = max(best_value, sigs[pi]**Paras.strike_gamma_init*tvalue)
                min_tvalue = min(min_tvalue, tvalue)
            allvalues[fi,si] = best_value*min_tvalue**2
                
    allprobs = np.mean(allvalues, axis = 1)
    allprobs = gaussian_filter1d(allprobs, 2)

    peakfreqs, _ = find_peaks(allprobs)
    freqproms = peak_prominences(allprobs, peakfreqs)[0]
    
    peakfreqs = np.array([val for _, val in sorted(zip(freqproms, peakfreqs), reverse = True)]).astype('int')

    print('Loudest bell has significant frequencies at', peakfreqs/Paras.fcut_length)
    print('Assuming possition n -', loudest_bell_from_back)
    
    bp = loudest_bell_from_back
    temperament = 2.0**(1./12.)
    scale = [2,2,1,2,2,2,1,2,2,1,2,2,2,1,2,2,1,2,2,2] #etc (diatonic)
    
    base = peakfreqs[0]/Paras.fcut_length
    
    nominals = np.zeros(Paras.nbells)
    
    nominals[bp] = base
    for i in range(bp-1, -1, -1):  #Do heavier bells
        nominals[i] = nominals[i + 1]/(temperament**scale[i])
    for i in range(bp + 1, Paras.nbells, 1):
        nominals[i] = nominals[i - 1]*(temperament**scale[i-1])
        
    nominals = nominals[::-1]
    
    print('Assumed nominal frequencies:')
    print(nominals)
    print('___________________________')
    
    return nominals
    

def find_strike_times_rounds(Paras, Data, Audio, final = False, doplots = 0):
    #Go through the rounds in turn instead of doing it bellwise
    #Allows for nicer plotting and stops mistakely hearing louder bells. Hopefully.
    
    #This should use the routine from the initial rounds spotting?    
    tenor_probs = Data.strike_probabilities[-1]

    tenor_probs_smooth = gaussian_filter1d(tenor_probs, int(Paras.smooth_time/Paras.dt))

    tenor_peaks, _ = find_peaks(tenor_probs) 
    if not final:
        tenor_peaks = tenor_peaks[tenor_peaks < Paras.reinforce_tmax/Paras.dt]
    tenor_peaks = tenor_peaks[tenor_peaks > Paras.first_change_start - 20]
    
    prominences = peak_prominences(tenor_probs, tenor_peaks)[0]

    #Test the first few tenor peaks to see if the following diffs are fine...    
    tenor_probs_smooth = gaussian_filter1d(tenor_probs, int(Paras.smooth_time/Paras.dt))
    tenor_peaks = tenor_peaks[prominences > tenor_probs_smooth[tenor_peaks]]
    
    #Determine whether this is the start of the ringing or not... Actually, disregard the first change whatever. Usually going to be wrong...
    
    allstrikes = []; allconfs = []; allcadences = []
            
    start = 0; end = 0
    
    tcut = Paras.rounds_tcut*int(Paras.cadence)

    strike_probs = Data.strike_probabilities

    #Obtain adjusted probs
    strike_probs_adjust = np.zeros(strike_probs.shape)
    strike_probs_adjust = strike_probs[:, :]**(Paras.probs_adjust_factor + 1)/(np.sum(strike_probs[:,:], axis = 0) + 1e-6)**Paras.probs_adjust_factor
    
    strike_probs_adjust = gaussian_filter1d(strike_probs_adjust, Paras.rounds_probs_smooth, axis = 1)

    for bell in range(Paras.nbells):
        plt.plot(Data.ts[:len(strike_probs[bell])], strike_probs[bell], label = bell + 1, c = cmap(np.linspace(0,1,Paras.nbells)[bell]))
                
    plt.title('Probability of strike at each time')
    plt.legend()
    plt.xlim(0,10)
    plt.tight_layout()
    plt.close()

    plot_max = 30   #Do some plotting
    fig, axs = plt.subplots(3,4)
    tplots = np.arange(len(strike_probs[-1]))*Paras.dt
    for bell in range(Paras.nbells):
        ax = axs[bell//4, bell%4]
        ax.plot(tplots, strike_probs_adjust[bell,:])
        ax.set_title(bell+1)
        ax.set_xlim(0,plot_max)
    plt.tight_layout()
    plt.close()
        
    allpeaks = []; allbigs = []; allsigs = []
    for bell in range(Paras.nbells):
        
        probs = strike_probs_adjust[bell]  

        probs_smooth = 1.0*gaussian_filter1d(probs, int(Paras.smooth_time/Paras.dt))

        peaks, _ = find_peaks(probs)
        
        peaks = peaks[peaks > Data.first_change_limit[bell]]
        
        prominences = peak_prominences(probs, peaks)[0]
        
        bigpeaks = peaks[prominences > 0.5*probs_smooth[peaks]]  #For getting first strikes, need to mbe more significant
        peaks = peaks[prominences > 0.1*probs_smooth[peaks]]

        sigs = peak_prominences(probs, peaks)[0]/probs_smooth[peaks]
        
        sigs = sigs/np.max(sigs)
        
        allpeaks.append(peaks); allbigs.append(bigpeaks); allsigs.append(sigs)

    #Find all peaks to begin with
    #Run through each set of rounds 
        
    handstroke = Data.handstroke_first
    next_end = 0
    
    count = 0
    unsurecount = 0
    
    if len(Paras.allstrikes) == 0:
        taims = np.zeros(Paras.nbells)
    else:
        change_start = np.mean(Data.last_change) - Data.cadence_ref*((Paras.nbells - 1)/2)
        change_end = np.mean(Data.last_change) + Data.cadence_ref*((Paras.nbells - 1)/2)
        
        rats = (Data.last_change - change_start)/(change_end - change_start)
        if handstroke:
            taims  = np.array(Data.last_change) + int(Paras.nbells*Data.cadence_ref)
            next_start = change_start + int(Paras.nbells*Data.cadence_ref)
            next_end = change_end + int(Paras.nbells*Data.cadence_ref)
        else:
            taims  = np.array(Data.last_change) + int((Paras.nbells + 1)*Data.cadence_ref)
            next_start = change_start + int((Paras.nbells+1)*Data.cadence_ref)
            next_end = change_end + int((Paras.nbells+1)*Data.cadence_ref)

        taims = next_start + (next_end - next_start)*rats
                                   
        start = next_start - 3.0*int(Data.cadence_ref)
        end  =  next_end   + 3.0*int(Data.cadence_ref)

    while next_end < np.max(peaks) - int(Paras.max_change_time/Paras.dt):
        
        #Paras.local_tmin = 0
        plotflag = False
        strikes = np.zeros(Paras.nbells)
        confs = np.zeros(Paras.nbells)
        certs = np.zeros(Paras.nbells) #To know when to stop
        
        count += 1
        if len(Paras.allstrikes) == 0 and len(allstrikes) == 0:  #Establish first strike overall.
            #IMPROVE ON THIS - DETERMINE FROM THE INIT AUDIO BIT
            for bell in range(Paras.nbells): #This is a bit shit -- improve it?
                taim = Paras.first_change_start + Paras.cadence*bell
                
                start_bell = taim - int(3.5*Paras.cadence)  #Aim within the change
                end_bell = taim + int(3.5*Paras.cadence)

                
                poss = allbigs[bell][(allbigs[bell] > start_bell)*(allbigs[bell] < end_bell)]
                
                if len(poss) < 1:
                    
                    print(allbigs[bell], start_bell, end_bell)
                    plt.plot(probs)
                    plt.show()
                    raise Exception('Cannot find strike for bell', bell+1, 'in rounds. If the initial rounds was choppy try changing the start time.')
                
                strikes[bell] = poss[0]
                
                if final:
                    confs[bell] = 1.0
                else:
                    confs[bell] = 0.0

        else:  #Find options in the correct range
            failcount = 0; 
            for bell in range(Paras.nbells):
                peaks = allpeaks[bell]
                sigs = allsigs[bell]
                peaks_range = peaks[(peaks > start)*(peaks < end)]
                sigs_range = sigs[(peaks > start)*(peaks < end)]
                
                start_bell = taims[bell] - int(3.5*Paras.cadence)  #Aim within the change
                end_bell = taims[bell] + int(3.5*Paras.cadence)
                #Check physically possible...
                if len(allstrikes) == 0:
                    start_bell = max(start_bell, Data.last_change[bell] + int(3.0*Paras.cadence))
                else:
                    start_bell = max(start_bell, allstrikes[-1][bell] + int(3.0*Paras.cadence))
                    
                sigs_range = sigs_range[(peaks_range > start_bell)*(peaks_range < end_bell)]
                peaks_range = peaks_range[(peaks_range > start_bell)*(peaks_range < end_bell)]

                if len(peaks_range) == 1:   #Only one time that it could reasonably be
                    strikes[bell] = peaks_range[0]
                    tvalue = 1.0/(abs(peaks_range[0] - taims[bell])/tcut + 1)**Paras.strike_alpha
                    if final:
                        confs[bell]  = 1.0
                    else:
                        confs[bell] = 1.0  #Timing doesn't really matter, but prominence does -- don't want ambiguity
                    certs[bell] = tvalue*sigs_range[0]/np.max(sigs)
                    
                elif len(peaks_range) > 1:
                                          
                    scores = []
                    for k in range(len(peaks_range)):  #Many options...
                        if abs(peaks_range[k] - taims[bell]) < int(Paras.rounds_leeway*Paras.cadence):
                            tvalue = 1.0
                        else:
                            tvalue = 1.0/(abs(abs(peaks_range[k] - taims[bell]) - int(Paras.rounds_leeway*Paras.cadence))/tcut + 1)**(Paras.strike_alpha)
                            
                        if final:
                            yvalue = sigs_range[k]
                        else:
                            yvalue = sigs_range[k]
                            
                        scores.append(tvalue*sigs_range[k]/np.max(sigs))
                                                
                        
                    kbest = scores.index(max(scores))
                    
                    strikes[bell] = peaks_range[kbest]
                    if final:
                        confs[bell] = (sigs_range[kbest]/np.sum(sigs_range))**2
                    else:
                        confs[bell] = (sigs_range[kbest]/np.sum(sigs_range))**2
                        
                    certs[bell] = scores[kbest]

                    if confs[bell] < 0.5:
                        unsurecount += 1
                        if doplots > 0:
                            plotflag = True
                            print('Bell', bell + 1, 'unsure but not too bad...')
                        
                else:
                    #print('No peaks found in sensible range')
                    if doplots > 0:
                        plotflag = True
                    failcount += 1
                    #Pick best peak in the change? Seems to work when things are terrible
                    
                    peaks = allpeaks[bell]
                    sigs = allsigs[bell]
                    peaks_range = peaks[(peaks > start)*(peaks < end)]
                    sigs_range = sigs[(peaks > start)*(peaks < end)]
                    
                    start_bell = max(start_bell, allstrikes[-1][bell] + int(3.0*Paras.cadence))
                    end_bell = end
                    
                    sigs_range = sigs_range[(peaks_range > start_bell)*(peaks_range < end_bell)]
                    peaks_range = peaks_range[(peaks_range > start_bell)*(peaks_range < end_bell)]


                    scores = []
                    for k in range(len(peaks_range)):  #Many options...
                        tvalue = 1.0/(abs(peaks_range[k] - taims[bell])/tcut + 1)**Paras.strike_alpha
                        yvalue = sigs_range[k]/np.max(sigs_range)
                        scores.append(tvalue*yvalue**2.0)
                        
                    if len(scores) > 0:
                        kbest = scores.index(max(scores))
                        
                        strikes[bell] = peaks_range[kbest]
                        confs[bell] = 0.0
                        certs[bell] = 0.0

                        if doplots > 0:
                            plotflag = True
                            print('Bell', bell + 1, 'Not found near to its past position... Will either guess and move on or stop')
                    else:
                        #Pick average point in the change

                        strikes[bell] = int(0.5*(start + end))
                        confs[bell] = 0.0
                        certs[bell] = 0.0
 
            if failcount > 1 or np.median(certs) < 0.01:
                #Nothing has been found - stop!!
                print('Confidence in the change not good enough to continue...')
                if plotflag > 1 or plotflag:
                    plotstart = int(min(strikes)); plotend = int(max(strikes))
                    ts = np.arange(plotstart - int(1.0/Paras.dt),plotend + int(1.0/Paras.dt))*Paras.dt + Paras.local_tmin
    
                    for bell in range(Paras.nbells):
                        plt.plot(ts, strike_probs[bell,plotstart - int(1.0/Paras.dt):plotend + int(1.0/Paras.dt)], c = cmap(bell/(Paras.nbells-1)), linestyle = 'dotted')
                        plt.plot(ts, strike_probs_adjust[bell,plotstart - int(1.0/Paras.dt):plotend + int(1.0/Paras.dt)], label = bell + 1, c = cmap(bell/(Paras.nbells-1)))
                    plt.scatter(start*Paras.dt + Paras.local_tmin, - 0.1, c = 'green')
                    plt.scatter(end*Paras.dt + Paras.local_tmin,  - 0.1, c = 'red')
                    plt.scatter(taims*Paras.dt + Paras.local_tmin,  - 0.2*np.ones(Paras.nbells), c = cmap(np.linspace(0,1,Paras.nbells)), marker = 's')
                    plt.scatter(strikes*Paras.dt + Paras.local_tmin,  - 0.3*np.ones(Paras.nbells), c = cmap(np.linspace(0,1,Paras.nbells)), marker = '*')
                    plt.legend()
                    plt.title(np.median(certs))
                    plt.close()

                Paras.ringing_finished = True
                if len(allstrikes) == 0:
                    return [], []
         
                else:
                    return np.array(allstrikes).T, np.array(allconfs).T
            
            
        allstrikes.append(strikes)
        allconfs.append(confs)
                
        if len(allstrikes) == 0:
            Paras.ringing_finished = True

            return [], []
 
        if doplots > 1:
            plotflag = True
        
        if plotflag:  #Plot the probs and things
            plotstart = int(min(strikes)); plotend = int(max(strikes))
            ts = np.arange(plotstart - int(1.0/Paras.dt),plotend + int(1.0/Paras.dt))*Paras.dt + Paras.local_tmin
            
            for bell in range(Paras.nbells):
                plt.plot(ts, strike_probs[bell,plotstart - int(1.0/Paras.dt):plotend + int(1.0/Paras.dt)], c = cmap(bell/(Paras.nbells-1)), linestyle = 'dotted')
                plt.plot(ts, strike_probs_adjust[bell,plotstart - int(1.0/Paras.dt):plotend + int(1.0/Paras.dt)], label = bell + 1, c = cmap(bell/(Paras.nbells-1)))
            plt.scatter(start*Paras.dt + Paras.local_tmin, - 0.1, c = 'green')
            plt.scatter(end*Paras.dt + Paras.local_tmin,  - 0.1, c = 'red')
            plt.scatter(taims*Paras.dt + Paras.local_tmin,  - 0.2*np.ones(Paras.nbells), c = cmap(np.linspace(0,1,Paras.nbells)), marker = 's')
            plt.scatter(strikes*Paras.dt + Paras.local_tmin,  - 0.3*np.ones(Paras.nbells), c = cmap(np.linspace(0,1,Paras.nbells)), marker = '*')
            plt.legend()
            plt.title(np.median(certs))
            plt.show()
            
        #Determine likely location of the next change END
        #Need to be resilient to method mistakes etc... 
        #Log the current avg. bell cadences
        allcadences.append((max(strikes) - min(strikes))/(Paras.nbells - 1))     

        nrows_count = int(min(len(allcadences), 20))
        Data.cadence_ref = np.mean(allcadences[-nrows_count:])
        
        change_start = np.mean(strikes) - Data.cadence_ref*((Paras.nbells - 1)/2)
        change_end = np.mean(strikes) + Data.cadence_ref*((Paras.nbells - 1)/2)
        
        rats = (strikes - change_start)/(change_end - change_start)
                
        if not handstroke:
            taims  = np.array(allstrikes[-1]) + int(Paras.nbells*Data.cadence_ref)
            next_start = change_start + int(Paras.nbells*Data.cadence_ref)
            next_end = change_end + int(Paras.nbells*Data.cadence_ref)
        else:
            taims  = np.array(allstrikes[-1]) + int((Paras.nbells + 1)*Data.cadence_ref)
            next_start = change_start + int((Paras.nbells+1)*Data.cadence_ref)
            next_end = change_end + int((Paras.nbells+1)*Data.cadence_ref)

        taims = next_start + (next_end - next_start)*rats
                   
        handstroke = not(handstroke)
                
        start = next_start - 1.5*int(Data.cadence_ref)
        end  =  next_end   + 3.5*int(Data.cadence_ref)

    if len(allconfs) > 0:
        Data.freq_data = np.array([Paras.dt, Paras.fcut_length, np.mean(allconfs[1:]), np.min(allconfs[1:])])

        #Want to prioritise rows which are nicely spaced -- minimum distance to a strike either side
        if not final:
            print('Average confidence for reinforcement on peaks alone: %.1f' % (100*np.mean(allconfs[1:])), '%')
        else:
            print('Average confidence in this range: %.1f' % (100*np.mean(allconfs)), '%')
            print('Minimum confidence in this range: %.1f' % (100*np.min(allconfs)), '%')


    if len(allstrikes) == 0:
        Paras.ringing_finished = True

        return [], []

    spacings = 1e6*np.ones((len(allstrikes), Paras.nbells, 2))
    yvalues = np.arange(Paras.nbells)
        
    for ri, row in enumerate(allstrikes):
        #Sort out ends
        order = np.array([val for _, val in sorted(zip(row, yvalues), reverse = False)])
        for si in range(len(row)):
            if si == 0:
                if ri == 0:
                    spacings[ri,order[si],0] =  Paras.cadence*2
                else:
                    spacings[ri,order[si],0] =  row[order[si]] - np.max(allstrikes[ri-1])
            else:
                spacings[ri,order[si],0] = row[order[si]] - row[order[si-1]]
            
            if si == len(row)- 1:    
                if ri == len(allstrikes) - 1:
                    spacings[ri,order[si],1] =  Paras.cadence*2
                else:
                    spacings[ri,order[si],1] =  np.min(allstrikes[ri+1])  - row[order[si]]  
            else:
                spacings[ri,order[si],1] = row[order[si+1]] - row[order[si]]
           
    allconfs = allconfs*(np.min(spacings, axis = 2)/ np.max(spacings))
            
        
    if not final:
        print('Average confidence for reinforcement, striking adjusted: %.1f' % (100*np.mean(allconfs[1:])), '%')
        print('Number of unsure changes:', unsurecount)

        
    return np.array(allstrikes).T, np.array(allconfs).T
        

def do_frequency_analysis(Paras, Data, Audio):
    #Now takes existing strikes data to do this (to make reinforcing easier)
    #__________________________________________________
    #Takes strike times and reinforces the frequencies from this. Needs nothing else, so works with the rounds too
     
    tcut = int(Data.cadence*Paras.freq_tcut) #Peak error diminisher
    tcut_big = int(Data.cadence*2.5)
    
    freq_tests = np.arange(0, len(Data.transform[0])//4)
    nstrikes = len(Data.strikes[0])
    allprobs = np.zeros((len(freq_tests), Paras.nbells))
    allvalues = np.zeros((len(freq_tests), len(Data.strikes[0]), Paras.nbells))
        
    bellsums = np.zeros(Paras.nbells)
    for bell in range(Paras.nbells):
        bellsums[bell] = np.sum(Data.strike_certs[bell,:]**Paras.beta)
        Data.strike_certs[bell,:] = Data.strike_certs[bell,:]/bellsums[bell]
        
    #Try going through in turn for each set of rounds? Should reduce the order of this somewhat
    for si in range(nstrikes):
        #Run through each row
        strikes = Data.strikes[:,si] #Strikes on this row.
        certs = Data.strike_certs[:, si]
        tmin = int(np.min(strikes) - Paras.cadence*(Paras.nbells - 2)); tmax = int(np.max(strikes) + Paras.cadence*(Paras.nbells - 2))
        #print('Examining row %d \r' % si)
        for fi, freq_test in enumerate(freq_tests):
            #fig = plt.figure()
            diff_slice = Data.transform_derivative[tmin:tmax,freq_test]
            diff_slice[diff_slice < 0.0] = 0.0
            diffsum = diff_slice**2
            
            diffsum = gaussian_filter1d(diffsum, Paras.freq_smoothing)
                
            peaks, _ = find_peaks(diffsum)
            
            prominences = peak_prominences(diffsum, peaks)[0]
            
            sigs = prominences/np.max(prominences)   #Significance of the peaks relative to the background flow
            
            peaks = peaks + tmin
        
            #Two options here -- the first works well on 12, the second on 6. Oh well...
            if True:
                #For each frequency, check consistency against confident strikes (threshold can be really high for that -- 99%?)
                for bell in range(Paras.nbells):
                    best_value = 0.0; min_tvalue = 1e6
                    pvalue = certs[bell]**Paras.beta
                    for pi, peak_test in enumerate(peaks):
                        tvalue = 1.0/(abs(peak_test - strikes[bell])/tcut + 1)**Paras.strike_alpha
                        best_value = max(best_value, sigs[pi]**Paras.strike_gamma_init*tvalue*pvalue)
                        min_tvalue = min(min_tvalue, tvalue)
                    allvalues[fi,si,bell] = best_value*min_tvalue**2
                
            else:
                for bell in range(Paras.nbells):
                    scores = []; tvalues = []
                    pvalue = certs[bell]**Paras.beta
                    for pi, peak_test in enumerate(peaks):
                        tvalue = (abs(peak_test - strikes[bell])/tcut_big)
                        scores.append(1.0/(tvalue + 1)**Paras.strike_alpha)
    
                        #scores.append(1.0 - sigs[pi]**Paras.strike_gamma_init*tvalue)
                        tvalues.append(tvalue)
                        
                    minscore = min(scores) #Score of the WORST PEAK
                    ind = scores.index(max(scores))  #Index of the BEST PEAK
                    
                    allvalues[fi,si,bell] = minscore*sigs[ind]**Paras.strike_gamma_init*pvalue
            

    allprobs[:,:] = np.mean(allvalues, axis = 1)
    

    #INSTEAD:: LOOK for frequenc
    
    #for bell in range(Paras.nbells):  #Normalise with the mean value perhaps
    #    allprobs[:,bell] = allprobs[:,bell]/np.mean(allprobs[:,bell])

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
    if Paras.nbells > 6:
        ncols = 3; nrows = 4
    else:
        ncols = 2; nrows = 3
    
    fig, axs = plt.subplots(nrows,ncols, figsize = (15,10))

    for bell in range(Paras.nbells):
        allprobs[:,bell] = allprobs[:,bell]/np.max(allprobs[:,bell])
        
    npeaks = Paras.n_frequency_picks
    final_freqs = []   #Pick out the frequencies to test
    best_probs = []   #Pick out the probabilities for each bell for each of these frequencies
    #Filter allprobs nicely
    #Get rid of narrow peaks etc.
    for bell in range(Paras.nbells):
        ax = axs[bell%nrows, bell//nrows]
        
        probs_raw = allprobs[:,bell]
        probs_clean = np.zeros(len(probs_raw))   #Gets rid of the horrid random peaks
        for fi in range(1,len(probs_raw)-1):
            #probs_clean[fi] = np.min(probs_raw[fi-1:fi+2])
            probs_clean[fi] = probs_raw[fi]
            
        probs_clean = gaussian_filter1d(probs_clean, Paras.freq_filter) #Stops peaks wiggling around. Can cause oscillation in ability.
        
        ax.plot(freq_tests, probs_clean, label = bell, zorder = 10)
        #ax.plot(best_freqs/cut_length, probs_clean_smooth, label = bell, zorder = 5)
        
        peaks, _ = find_peaks(probs_clean)
        
        peaks = peaks[peaks > 50]#Data.nominals[bell]*1.1]  #Use nominal frequencies here?
        
        prominences = peak_prominences(probs_clean, peaks)[0]
        
        peaks = np.array([val for _, val in sorted(zip(prominences, peaks), reverse = True)]).astype('int')
           
        peaks = peaks[:npeaks]
        prominences = peak_prominences(probs_clean, peaks)[0]

        peaks = peaks[prominences > 0.25*np.max(prominences)]
        
        ax.scatter(freq_tests[peaks], np.zeros(len(peaks)), c = 'red')

        final_freqs = final_freqs + freq_tests[peaks].tolist()
                   
    #Determine probabilities for each of the bells, in case of overlap
    #Only keep definite ones?
    
    #Remove repeated indices
    final_freqs = list(set(final_freqs))
    final_freqs = sorted(final_freqs)
    final_freqs = np.array(final_freqs)

    #Sort by height and filter out those which are too nearby -- don't count for the final number
    for freq in final_freqs:
        bellprobs = np.zeros(Paras.nbells)
        for bell in range(Paras.nbells):

            freq_range = 2  #Put quite big perhaps to stop bell confusion. Doesn't like it, unfortunately.

            #Put some blurring in here to account for the wider peaks
            top = np.sum(allprobs[freq-freq_range:freq+freq_range + 1, bell])
            bottom = np.sum(allprobs[freq-freq_range:freq+freq_range + 1, :])
                            
            bellprobs[bell] = (top/bottom)**2.0
        
        best_probs.append(bellprobs)
        
    best_probs = np.array(best_probs)

    nfinals = Paras.n_frequency_picks
    

    for bell in range(Paras.nbells):
        rat = 0.1*np.max(allprobs[:,bell])

        best_probs[:,bell] = best_probs[:,bell]/np.max(best_probs[:,bell])

        ax = axs[bell%nrows, bell//nrows]
        ax.scatter(np.array(final_freqs), -1*rat*np.ones(len(final_freqs)), c = 'green', s = 50*best_probs[:,bell])

        freqs_arrange = np.array([val for _, val in sorted(zip(best_probs[:,bell], final_freqs), reverse = True)]).astype('int')
        
            
        for fi in freqs_arrange[:nfinals*2]:
            #Check for close by alternatives to the biggest peaks and remove them
            others = final_freqs[(abs(final_freqs - fi) < 20) * (abs( final_freqs - fi) > 0)]
            if best_probs[np.where(final_freqs == fi)[0], bell] > 0.0:
                for other in others:
                    ind = np.where(final_freqs == other)[0]
    
                    best_probs[ind, bell] = 0.0
                
    #print('Confident frequency picks for bell', bell+1, ': ', np.sum(best_probs[:,bell] > 0.1), np.array(final_freq_ints)[best_probs[:,bell] > 0.1])

    for bell in range(Paras.nbells):
        #Filter out so there are only a few peaks on each one (will be quicker)
        threshold = sorted(best_probs[:,bell], reverse = True)[nfinals + 1]
        threshold = max(threshold, 5e-2)
        best_probs[:,bell] = best_probs[:,bell]*[best_probs[:,bell] > threshold]
    
    #Then finally run through and remove any picks that are generally useless
    frequencies = []; frequency_probabilities = []
    for fi, freq in enumerate(final_freqs):
        if np.max(best_probs[fi, :]) > 0.05:
            frequencies.append(freq)
            frequency_probabilities.append(best_probs[fi,:])
            
    frequencies = np.array(frequencies)
    frequency_probabilities = np.array(frequency_probabilities)
    
    
    for bell in range(Paras.nbells):
        ax = axs[bell%nrows, bell//nrows]
        ax.plot(freq_tests, allprobs[:,bell], label = bell)
        ax.scatter(np.array(frequencies), -2*rat*np.ones(len(frequencies)), c = 'blue', s = 50*frequency_probabilities[:,bell])

        ax.set_title('Bell %d' % (bell + 1))
        #if bell != Paras.nbells-1:
        #    ax.set_xticks([])

    plt.suptitle('Frequency Analysis After filtering')
    plt.tight_layout()

    plt.show()
 
    
    
    return frequencies, frequency_probabilities
       
def find_first_strikes(Paras, Data, Audio):
    
    #Takes normalised wave vector, and does some fourier things
        
    tenor_probs = Data.strike_probabilities[-1]
    tenor_peaks, _ = find_peaks(tenor_probs) 
    tenor_peaks = tenor_peaks[tenor_peaks < Paras.rounds_tmax/Paras.dt]
    prominences = peak_prominences(tenor_probs, tenor_peaks)[0]
    
    #Test the first few tenor peaks to see if the following diffs are fine...    
    tenor_big_peaks = np.array(tenor_peaks[prominences > 0.25])  
    tenor_peaks = np.array(tenor_peaks[prominences > 0.01]) 
    
    plt.scatter(Data.ts[tenor_peaks] + Paras.local_tmin, np.zeros(len(tenor_peaks)), c= 'orange')
    plt.scatter(Data.ts[tenor_big_peaks] + Paras.local_tmin, np.zeros(len(tenor_big_peaks)), c= 'red')

    plt.plot(Data.ts + Paras.local_tmin, tenor_probs)
    if len(tenor_big_peaks) > 10:
        plt.xlim(0.0,Data.ts[tenor_big_peaks[10]] + Paras.local_tmin)
    else:
        plt.xlim(0.0,Data.ts[np.max(tenor_big_peaks)] + Paras.local_tmin)
    plt.show()
    
                        
    if len(tenor_big_peaks) < 4:
        raise Exception('Reliable tenor strikes not found within the required time... Try cutting out start silence?')

    tenor_strikes = []; best_length = 0; go = True
    for first_test in range(4):
        if not go:
            break
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
        diff2s = teststrikes[2:] - teststrikes[:-2]
        for tests in range(2, len(diff2s)):
            if max(diff2s[:4]) - min(diff2s[:4]) < int(0.5/Paras.dt):
                if tests + 2 > best_length:
                    tenor_strikes = teststrikes[:tests+2]
                    if best_length >= Paras.nrounds_max:
                        go = False
                        break
                    
    if len(tenor_strikes) < 4:
        print(tenor_big_peaks, tenor_peaks)
        raise Exception('Reliable tenor strikes not found within the required time... Try cutting out start silence?')
        
    print('Tenor strikes in rounds (check these are reasonable): ', np.array(tenor_strikes)*Paras.dt)
    
    
    diff1s = tenor_strikes[1::2] - tenor_strikes[0:-1:2]
    diff2s = tenor_strikes[2::2] - tenor_strikes[1:-1:2]
    if np.mean(diff1s) < np.mean(diff2s):
        Paras.handstroke_first = True
    else:
        Paras.handstroke_first = False
        
    Data.handstroke_first = Paras.handstroke_first
    
    Paras.first_change_start = tenor_strikes[0]
    Paras.first_change_end = tenor_strikes[1]
    
    Paras.first_change_limit = tenor_strikes[0]*np.ones(Paras.nbells) + 10
    Paras.reinforce_tmax = Paras.reinforce_tmax + tenor_strikes[0]
    nrounds_test = len(tenor_strikes) - 1
    
    handstroke = Data.handstroke_first
    
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
                
        if rounds == 0:
            Data.first_change_limit = belltimes - 20  #Time after which each bell needs to strike
            
    plt.plot(tenor_probs)
    for r in range(len(init_aims)):
        plt.scatter(init_aims[r], np.zeros(Paras.nbells), c = 'red')
    plt.scatter(tenor_strikes, np.zeros(len(tenor_strikes)), c = 'black')
    plt.title('Initial rounds aims with tenor detection...')
    plt.xlim(np.min(tenor_strikes) - 50, np.max(tenor_strikes) + 50)
    plt.show()
    
    print('Attempting to find ', len(init_aims), ' rows for rounds...')
    
    cadence = np.mean(cadences)
    Data.cadence = cadence
    Paras.cadence = cadence
    #Do this just like the final row finder! But have taims all nicely. Also use same confidences.

    #Obtain VERY smoothed probabilities, to compare peaks against
    
    #Use these guesses to find the ACTUAL peaks which should be nearby...
    init_aims = np.array(init_aims)
    
    strikes = np.zeros(init_aims.T.shape)
    strike_certs = np.zeros(strikes.shape)
    
    Paras.nrounds_max = len(init_aims)

    probs_raw = Data.strike_probabilities[:]
    probs_raw = gaussian_filter1d(probs_raw, Paras.strike_smoothing, axis = 1)

    tcut = 1 #Be INCREDIBLY fussy with these picks or the wrong ones will get nicked
    
    for bell in range(Paras.nbells):
        #Find all peaks in the probabilities for this individual bell
        probs_adjust = probs_raw[bell,:]**(Paras.probs_adjust_factor + 1)/(np.sum(probs_raw[:,:], axis = 0) + 1e-6)**Paras.probs_adjust_factor
        #Adjust for when the rounds is a bit shit
        
        peaks, _ = find_peaks(probs_adjust) 
        sigs = peak_prominences(probs_adjust, peaks)[0]
        sigs = sigs/np.max(sigs)

        for ri in range(Paras.nrounds_max): 
            #Actually find the things. These should give reasonable options
            aim = init_aims[ri, bell]
            
            poss = peaks[(peaks > aim - 0.5*cadence)*(peaks < aim + 0.5*cadence)]   #These should be accurate strikes
            yvalues = sigs[(peaks > aim - 0.5*cadence)*(peaks < aim + 0.5*cadence)]

            scores = []
            for k in range(len(poss)):  #Many options...
                tvalue = 1.0/(abs(poss[k] - aim)/tcut + 1)**Paras.strike_alpha
                yvalue = yvalues[k]
                scores.append(tvalue*yvalue**Paras.strike_gamma_init)
                
            if len(scores) > 0:
                
                kbest = scores.index(max(scores))
                
                strikes[bell, ri] = poss[kbest]
                strike_certs[bell,ri] = scores[k]

            else:
                strikes[bell, ri] = aim
                strike_certs[bell, ri] = 0.0
                
    strikes = np.array(strikes)
    strike_certs = np.array(strike_certs)    

    #Check this is indeed handstroke or not, in case of an oddstruck tenor
    diff1s = strikes[:,1::2] - strikes[:,0:-1:2]
    diff2s = strikes[:,2::2] - strikes[:,1:-1:2]
    
    if np.mean(diff1s) < np.mean(diff2s):
        Paras.handstroke_first = False
    else:
        Paras.handstroke_first = True
        
    Data.handstroke_first = Paras.handstroke_first

    for bell in range(Paras.nbells):
                
        plt.plot(probs_raw[bell])
        
        plt.scatter(init_aims[:,bell], np.zeros(len(init_aims)), c= 'red', label = 'Predicted linear position')
        plt.scatter(strikes[bell,:], -0.1*max(probs_raw[bell])*np.ones(len(init_aims)), c= 'green', label = 'Probable pick', s = 50*strike_certs[bell,:])
        plt.legend()
        plt.xlim(np.min(strikes[strikes > 0.0]) - 100,np.max(strikes) + 100)
        plt.title(bell)
        plt.show()
    
    #Determine how many rounds there actually are? Nah, it's probably fine...
    return strikes, strike_certs
    
def find_strike_probabilities(Paras, Data, Audio, init = False, final = False):
    #Find times of each bell striking, with some confidence
        
    #Make sure that this transform is sorted in EXACTLY the same way that it's done initially.
    #No tbefores etc., just the derivatives.
    #If reinforcing, don't need the whole domain
    
    nt_reinforce = Paras.nt
        
    allprobs = np.zeros((Paras.nbells, nt_reinforce))
             
    difflogs = []; all_diffpeaks = []; all_sigs = []
    
    plot_bell = 5
    #Produce logs of each FREQUENCY, so don't need to loop
    for fi, freq_test in enumerate(Data.test_frequencies):
        
        diff_slice = Data.transform_derivative[:nt_reinforce, freq_test - Paras.frequency_range : freq_test + Paras.frequency_range + 1]
        diff_slice[diff_slice < 0.0] = 0.0
        diffsum = np.sum(diff_slice**2, axis = 1)
        
        diffsum = gaussian_filter1d(diffsum, 5)

        diffpeaks, _ = find_peaks(diffsum)
        
        prominences = peak_prominences(diffsum, diffpeaks)[0]
        
        diffsum_smooth = gaussian_filter1d(diffsum, int(Paras.smooth_time/Paras.dt))
        
        if init:
            diffpeaks = diffpeaks[prominences > diffsum_smooth[diffpeaks]]  #This is just for plotting...
                
        else:
            sigs = prominences[prominences > diffsum_smooth[diffpeaks]]

            diffpeaks = diffpeaks[prominences > diffsum_smooth[diffpeaks]]
            sigs = sigs/diffsum_smooth[diffpeaks]

        difflogs.append(diffsum)
        all_diffpeaks.append(diffpeaks)
        
        if not init:
            all_sigs.append(sigs)
        
            if False:
                for di, diffpeak in enumerate(all_diffpeaks[-1]):
                    if Data.ts[diffpeak] < 15.0:
                        plt.scatter(Data.ts[diffpeak],freq_test, color = 'black', s = 10*sigs[di]*Data.frequency_profile[fi,plot_bell])
        
    if not init:                    
        plt.xlim(0,15)
        plt.close()
     
        #input()
    
    if init:
        #The probabilities for each frequency correspnd exactly to those for each bell -- lovely
        difflogs = np.array(difflogs)

        for bell in range(Paras.nbells):  
            allprobs[bell] = difflogs[bell]/max(difflogs[bell])
    
            plt.plot(Data.ts[:nt_reinforce], allprobs[bell]/max(allprobs[bell]), label = bell + 1)
            
        plt.legend()
        plt.xlim(0.0,15.0)
        plt.title('Initial probabilities of bell strikes')
        plt.show()
                
        return allprobs

    else:
        #There are multiple frequency picks to choose from here, so it's more complicated
        difflogs = np.array(difflogs)
            
        if final:
            doplot = False
        else:
            doplot = True
        for bell in range(Paras.nbells):  
            final_poss = []; final_sigs = []; final_probs = []; final_freqs = []
            for fi, freq_test in enumerate(Data.test_frequencies):
                if Data.frequency_profile[fi,bell] > 0.05:  #This is a valid frequency
                    sigs = all_sigs[fi]/np.max(all_sigs[fi])
                    
                    if np.max(sigs) > 0.1: #Maybe this is harsh but it should work...         
                    
                        peaks = all_diffpeaks[fi]
                        final_poss = final_poss + peaks.tolist()
                        final_sigs = final_sigs + sigs.tolist()
                        for k in range(len(sigs)):
                            final_probs = final_probs + [Data.frequency_profile[fi,bell]]
                            final_freqs.append(freq_test)
     
            final_poss = np.array(final_poss)
            final_sigs = np.array(final_sigs)
            final_probs = np.array(final_probs)/np.max(final_probs)
            final_freqs = np.array(final_freqs)

            tcut = int(Paras.prob_tcut/Paras.dt)
            
            overall_probs = np.zeros(len(diffsum))
                         
            #Need to split this up into time slices Ideally...
            t_ints = np.arange(len(diffsum))
            #Want number of significant peaks near the time really
                #Calculate probability at each time
            
            tvalues = 1.0/(np.abs(final_poss[:,np.newaxis] - t_ints[np.newaxis,:])/tcut + 1)**Paras.strike_alpha
            
            if Paras.frequency_skew < 0.5:
                fshift = np.zeros(len(final_freqs))
            else:
                fas = final_freqs/np.max(final_freqs)
                fshift = 1.0 - (1 - fas)**Paras.frequency_skew
            
            #fshift = np.ones(len(final_freqs))
                                    
            allvalues = tvalues*final_sigs[:,np.newaxis]**Paras.prob_beta*final_probs[:,np.newaxis]**Paras.strike_gamma*fshift[:,np.newaxis]

            allvalues = tvalues*final_sigs[:,np.newaxis]**Paras.prob_beta*final_probs[:,np.newaxis]**Paras.strike_gamma
                        
            absvalues = np.sum([tvalues > 0.5], axis = 1)
            
            absvalues = absvalues/np.max(absvalues)
            
            allvalues = allvalues*absvalues**Paras.near_freqs
            
            overall_probs =  np.sum(allvalues, axis = 0)
                
            overall_probs_smooth = gaussian_filter1d(overall_probs, int(Paras.smooth_time/Paras.dt), axis = 0)
                
            allprobs[bell] = overall_probs/(overall_probs_smooth + 1e-6)
            
            allprobs[bell] = allprobs[bell]/np.max(allprobs)
               
        
            if doplot:
                plt.plot(Data.ts[:nt_reinforce], allprobs[bell], label = bell + 1, c = cmap(np.linspace(0,1,Paras.nbells)[bell]))
            
        if doplot:
            try:
                plt.title('Probability of strike at each time')
                plt.legend()
                plt.xlim(Paras.first_change_start*Paras.dt, Paras.first_change_start*Paras.dt + 8.0)
                plt.tight_layout()
                plt.show()
            except:
                 pass

        return allprobs