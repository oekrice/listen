# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 19:18:11 2025

@author: eleph
"""

import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.io import wavfile
import numpy as np
from scipy.ndimage import gaussian_filter1d
import time
import os

from functions_smart import normalise, find_ringing_times, find_strike_probabilities, find_first_strikes, do_frequency_analysis, find_strike_times_rounds

import pandas as pd


plt.style.use('default')
cmap = plt.cm.jet
    

def plot_strikes(Paras):
    #Plots the things
    fig = plt.figure(figsize = (10,7))
    
    nrows = len(Paras.allstrikes[0])
    yvalues = np.arange(Paras.nbells) + 1
    
    #for bell in range(nbells):
    #    plt.scatter(all_strikes[bell], yvalues[bell]*np.ones(len(all_strikes[bell])),s=all_confidences[bell]*100)
    
    for row in range(nrows):
        plt.plot(Paras.allstrikes[:,row],yvalues)
        order = np.array([val for _, val in sorted(zip(Paras.allstrikes[:,row], yvalues), reverse = False)])
        print('Strikes', row, order, np.array(sorted(Paras.allstrikes[:,row]))*Paras.dt)#, confs)
        #print(all_strikes[:,row])
        #print('Louds', row, order, sorted(all_louds[:,row]))

    plt.xlim(0.0,30.0)
    plt.gca().invert_yaxis()
    plt.close(fig)

def save_strikes(Paras, tower):
    #Saves as a pandas thingummy like the strikeometer does
    allstrikes = []
    allbells = []
    yvalues = np.arange(Paras.nbells) + 1

    if not Paras.handstroke_first:
        for row in range(len(Paras.allstrikes[0])):
            order = np.array([val for _, val in sorted(zip(Paras.allstrikes[:,row], yvalues), reverse = False)])
            allstrikes = allstrikes + sorted((Paras.allstrikes[:,row]).tolist())
            allbells = allbells + order.tolist()
    else:
        for row in range(1, len(Paras.allstrikes[0])):
            order = np.array([val for _, val in sorted(zip(Paras.allstrikes[:,row], yvalues), reverse = False)])
            allstrikes = allstrikes + sorted((Paras.allstrikes[:,row]).tolist())
            allbells = allbells + order.tolist()
        
    allstrikes = 1000*np.array(allstrikes)*Paras.dt
    allbells = np.array(allbells)
    
    data = pd.DataFrame({'Bell No': allbells, 'Actual Time': allstrikes})
    data.to_csv('%s.csv' % (Paras.output_folder +  Paras.fname))  
    data.to_csv('%s.csv' % (Paras.output_folder + 'current_run'))  
    return
    
class audio_data():
    #Does the initial audio normalisation things
    def __init__(self, audio_filename, audio_folder):
        
        #Deal with file extensions automatically here
        fname_full = audio_filename
        if audio_filename[-4:] != '.wav':
            #Wav file is not imported. Check for an alternative...   
            for fname_check in os.listdir(audio_folder):   #Check for suitable .wav file
                if fname_check[:len(audio_filename)-4] == audio_filename[:len(audio_filename)-4]:
                    if fname_check[-4:] == '.wav':
                        fname_full = fname_check
                        break

        if fname_full[-4:] != '.wav':
            #Convert this to a wav
            print('Converting file to .wav format...')
            os.system('ffmpeg -loglevel quiet -i %s %s.wav' % (audio_folder + fname_full, audio_folder + fname_full[:-4]))
            print('File converted.')
        fname = audio_folder + fname_full[:-4] + '.wav'

        self.fs, self.data = wavfile.read(fname)
        if len(self.data.shape) > 1:  #Is stereo
            import_wave = np.array(self.data)[:,0]
        else:  #Isn't
            import_wave = np.array(self.data)[:]
            
        self.signal = normalise(16, import_wave)
        

class parameters():
    #Contains information like number of bells, max times etc. 
    #Also all variables that can theoretically be easily changed
    def __init__(self, Audio, nominal_freqs, overall_tmin, overall_tmax, overall_tcut, nbells):
                
        self.dt = 0.01
        self.fcut_length = 0.125  #Length of each transform slice (in seconds)
        
        self.transform_smoothing = 0.05 #Transform smoothing for the initial derivatives of the transform (in seconds)
        self.frequency_range = 3    #Range over which to include frequencies in a sweep (as in, 300 will count between 300-range:frequency+range+1 etc.)
        self.derivative_smoothing = 5  #Smoothing for the derivative (in INTEGER time lumps -- could change if necessary...)
        self.smooth_time = 2.0    #Smoothing over which to apply change-long changes (in seconds)
        self.max_change_time = 3.5 #How long could a single change reasonably be
        self.nrounds_min = 8 #How many rounds do you need (8 = 4 whole pulls, seems reasonable...)
        self.nrounds_max = 30 #How many rounds maximum
        self.nreinforce_rows = 4
        
        self.strike_smoothing = 1 #How much to smooth the input probability function
        self.strike_tcut = 1.0 #How many times the average cadence to cut off
        self.strike_alpha = 2  #How much to care about timing
        self.strike_gamma = 1  #How much to care about prominence
        self.strike_gamma_init = 1.5  #How much to care about prominence for the initial rounds
        
        self.freq_tcut = 0.2 #How many times the average cadence to cut off for FREQUENCIES (should be identical strikes really)
        self.freq_smoothing = 2 #How much to smooth the data when looking for frequencies (as an INTEGER)
        self.beta = 1   #How much to care whether strikes are certain when looking at frequencies
        self.freq_filter = 2#How much to filter the frequency profiles (in INT)
        self.n_frequency_picks = 10  #Number of frequencies to look for (per bell)
        
        self.rounds_probs_smooth = 2  
        self.rounds_tcut = 0.5 #How many times the average cadence to cut off find in rounds
        self.rounds_leeway = 1.5 #How far to allow a strike before it is more improbable

        self.rounds_tmax = 30.0
        self.reinforce_tmax = 60.0
        
        self.overall_tcut = overall_tcut  #How frequently (seconds) to do update rounds etc.
        self.probs_adjust_factor = 2.0   #Power of the bells-hitting-each-other factor. Less on higher numbers seems favourable.
        
        if overall_tmax > 0.0:
            Audio.signal = Audio.signal[int(overall_tmin*Audio.fs):int(overall_tmax*Audio.fs)]
        else:
            Audio.signal = Audio.signal[int(overall_tmin*Audio.fs):]
            
        self.overall_tmin = overall_tmin
        self.overall_tmax = overall_tmax
        if min(nominal_freqs) > 1.0:
            self.nbells = len(nominal_freqs)
        else:
            self.nbells = nbells
        self.fcut_int = 2*int(self.fcut_length*Audio.fs/2)  #Length of this cut (must be even for symmetry purposes)
        self.tmax =  len(Audio.signal)/Audio.fs
        
        self.prob_tcut = 0.1   #Time cutoff for all frequency identification
        self.prob_beta = 1.0  #How much to care about prominence looking at STRIKES
        self.near_freqs = 2  #How much to care about frequency peaks being nearby
        
        self.frequency_skew = 2.0   #How much to favour the high frequencies for timing reasons
        
        self.allstrikes = []
        
        if len(nominal_freqs) > 0:
            self.nominals = np.round(nominal_freqs*self.fcut_length).astype('int')
        else:
            self.nominals = []

class data():
    def __init__(self, Paras, Audio, tmin = -1, tmax = -1):
        #This is called at the start -- can make some things like blank arrays for the nominals and the like. Can also do the FTs here etc (just once)
        
        #Chnage the length of the audio as appropriate
        
        if tmin > 0.0:
            cut_min_int = int(tmin*Audio.fs)
        else:
            cut_min_int = 0
        if tmax > 0.0:
            cut_max_int = int(tmax*Audio.fs)
        else:
            cut_max_int = -1
        
        Audio.signal_trim = Audio.signal[cut_min_int:cut_max_int]
            
        self.nominals = Paras.nominals

        self.initial_profile = np.identity(Paras.nbells)     #Initial frequencies for the bells -- these are just the nominals
     
        self.ts, self.transform = self.do_fourier_transform(Paras, Audio)
     
        self.transform_derivative = self.find_transform_derivatives()
        
        print('__________________________________________________________________________________________')
        print('Calculating transform in range', cut_min_int/Audio.fs, 'to', cut_max_int/Audio.fs, 'seconds...')
        
        self.test_frequencies = self.nominals    #This is the case initially
        self.frequency_profile = np.identity(Paras.nbells)   #Each bell corresponds to its nominal frequency alone -- this will later be updated.
        

    def do_fourier_transform(self, Paras, Audio):
        
        full_transform = []; ts = []
        
        Paras.tmax = len(Audio.signal_trim)/Audio.fs
        
        t = Paras.fcut_length/2   #Initial time (halfway through each transform)
        
        while t < Paras.tmax - Paras.fcut_length/2:
            cut_start  = int(t*Audio.fs - Paras.fcut_int/2)
            cut_end    = int(t*Audio.fs + Paras.fcut_int/2)
            
            signal_cut = Audio.signal_trim[cut_start:cut_end]
            
            transform_raw = abs(fft(signal_cut)[:len(signal_cut)//2])
            transform = 0.5*transform_raw*Audio.fs/len(signal_cut)
                            
            ts.append(t)        
            full_transform.append(transform)
            
            t = t + Paras.dt
        
        ts = np.array(ts)
        full_transform = np.array(full_transform)    
                
        Paras.nt = len(ts)
        
        return ts, full_transform
    
    def find_transform_derivatives(self):
        allfreqs_smooth = gaussian_filter1d(self.transform, int(Paras.transform_smoothing/Paras.dt), axis = 0)
        diffs = np.zeros(allfreqs_smooth.shape)
        diffs[1:,:] = allfreqs_smooth[1:,:] - allfreqs_smooth[:-1,:] 
        
        diffs[diffs < 0.0] = 0.0
        return diffs
    
def do_reinforcement(Paras, Data, Audio):
        
    #Find the first strikes based on these probabilities. Hopefully some kind of nice pattern to the treble at least... 
    
    #Check if there is suitable existing frequency data for this tower and at these parameters. 
    #If this calculation comes out better, save it out. If not, don't.
    
    Paras.new_frequencies = False
    if Paras.overwrite_existing_freqs:
        if os.path.exists('%s%s_freq_quality.npy' % (Paras.frequency_folder, Paras.freqname)):
            os.system('rm -r %s%s_freq_quality.npy' % (Paras.frequency_folder, Paras.freqname))
            os.system('rm -r %s%s_freqprobs.npy' % (Paras.frequency_folder, Paras.freqname))
            os.system('rm -r %s%s_freqs.npy' % (Paras.frequency_folder, Paras.freqname))

    if Paras.use_existing_freqs:
        if os.path.exists('%s%s_freq_quality.npy' % (Paras.frequency_folder, Paras.freqname)):
            check_data = np.load('%s%s_freq_quality.npy' % (Paras.frequency_folder, Paras.freqname))
            if check_data[0] == Paras.dt and check_data[1] == Paras.fcut_length:
                print('__________________________________________')
                print('Suitable frequency file already exists:')
                print('Average confidence: %.1f' % (100*check_data[2]), '%')
                print('Minimum confidence: %.1f' % (100*check_data[3]), '%')
                Paras.n_reinforces = 0
                if check_data[2] < 0.9:
                    print('Imported frequencies not great... Proceeding anyway but consider making new ones')
                    time.sleep(2.0)
            
    for count in range(Paras.n_reinforces):
        
        #Find the probabilities that each frequency is useful. Also plots frequency profile of each bell, hopefully.
        print('__________________________________________________')
        print('Doing frequency analysis,  iteration number', count + 1, 'of', n_reinforces)
        
        Data.test_frequencies, Data.frequency_profile = do_frequency_analysis(Paras, Data, Audio)  
            
        #Save out frequency data only when finished reinforcing?
        
        print('Finding strike probabilities...')
        
        Data.strike_probabilities = find_strike_probabilities(Paras, Data, Audio, init = False, final = False)
                
        strikes, strike_certs = find_strike_times_rounds(Paras, Data, Audio, final = False, doplots = 1) #Finds strike times in integer space
    
        #Filter these strikes for the best rows, to then be used for reinforcement
        best_strikes = []; best_certs = []; allcerts = []; row_ids = []
        #Pick the ones that suit each bell in turn --but make sure to weight!
        for bell in range(Paras.nbells):
            threshold = 0.05   #Need changes to be at least this good... Need to improve on this really.
            allcerts = []; count = 0
            for row in range(len(strikes[0])):
                allcerts.append(strike_certs[bell,row])
            if len(allcerts) > Paras.nreinforce_rows:
                threshold = max(threshold, sorted(allcerts, reverse = True)[Paras.nreinforce_rows]) 
            for row in range(len(strikes[0])):
                if strike_certs[bell,row] > threshold and count < Paras.nreinforce_rows:
                    if row not in row_ids:
                        row_ids.append(row)
                        best_strikes.append(strikes[:,row])
                        best_certs.append(strike_certs[:,row])
                        count += 1
        print('Using', len(best_strikes), 'rows for next reinforcement')
        Data.strikes, Data.strike_certs = np.array(best_strikes).T, np.array(best_certs).T
        
        count += 1
        
        if len(Data.strikes) > 0 and len(Data.strike_certs) > 0:
            #Check if it's worth overwriting the old one? Do this at EVERY STEP, and save out to THIS filename.
            dosave = False
    
            if os.path.exists('%s%s_freq_quality.npy' % (Paras.frequency_folder, Paras.fname[:-4])):
                check_data = np.load('%s%s_freq_quality.npy' % (Paras.frequency_folder, Paras.fname[:-4]))
                if check_data[2] < Data.freq_data[2] and check_data[3] < Data.freq_data[3]:
                    #Worth overwriting any existing data
                    dosave = True
            else:
                dosave = True
                     
            if dosave:
                Paras.new_frequencies = True
                print('Best yet frequency data: saving it.')
                np.save('%s%s_freqs.npy' % (Paras.frequency_folder, Paras.fname[:-4]), Data.test_frequencies)
                np.save('%s%s_freqprobs.npy' % (Paras.frequency_folder, Paras.fname[:-4]), Data.frequency_profile)
                np.save('%s%s_freq_quality.npy' % (Paras.frequency_folder, Paras.fname[:-4]), Data.freq_data)
        else:
            raise Exception('Frequency reinforcement failed...')
    return
    
def find_final_strikes(Paras, Audio):
    
     #Create new data files in turn -- will be more effeicient ways but meh...
     tmin = 0.0
     tmax = tmin + Paras.overall_tcut + Paras.ringing_start*Paras.dt
     allstrikes = []; allcerts = []
     Paras.allcadences = []
     Paras.stop_flag = False
     Paras.local_tmin = Paras.overall_tmin
     Paras.local_tint = int(Paras.overall_tmin/Paras.dt)
     Paras.ringing_finished = False
     while not Paras.stop_flag and not Paras.ringing_finished:
         
         if tmax >= overall_tmax - 1.0:  #Last one
             Paras.stop_flag = True
             
         Paras.local_tmin = tmin + Paras.overall_tmin
         Paras.local_tint = int((tmin+Paras.overall_tmin)/Paras.dt) 

         Data = data(Paras, Audio, tmin = tmin, tmax = tmax) #This class contains all the important stuff, with outputs and things
         
         if Paras.new_frequencies:
             Data.test_frequencies = np.load('%s%s_freqs.npy' % (Paras.frequency_folder, Paras.fname[:-4]))
             Data.frequency_profile = np.load('%s%s_freqprobs.npy' % (Paras.frequency_folder, Paras.fname[:-4]))
         else:
             Data.test_frequencies = np.load('%s%s_freqs.npy' % (Paras.frequency_folder, Paras.freqname))
             Data.frequency_profile = np.load('%s%s_freqprobs.npy' % (Paras.frequency_folder, Paras.freqname))
            
         
         Data.strike_probabilities = find_strike_probabilities(Paras, Data, Audio, init = False, final = True)
                           
         if len(allstrikes) == 0:  #Look for changes after this time
             Data.first_change_limit = Paras.first_change_limit 
             Data.handstroke_first = Paras.handstroke_first
         else:
             if len(allstrikes)%2 == 0:
                 Data.handstroke_first = Paras.handstroke_first
             else:
                 Data.handstroke_first = not(Paras.handstroke_first)
             Data.first_change_limit = np.array(allstrikes[-1][:]) - int(tmin/Paras.dt) - 50   
             Data.last_change = np.array(allstrikes[-1]) - int(tmin/Paras.dt)
             Data.cadence_ref = Paras.cadence_ref

         Data.strikes, Data.strike_certs = find_strike_times_rounds(Paras, Data, Audio, final = True, doplots = 1) #Finds strike times in integer space
                   
         if len(Data.strikes) > 0:
             pass
         else:
             Paras.stop_flag = True
             print('No strike found for a bit so this is probably the end.')
            
         if len(Data.strikes) > 0:
             for row in range(0,len(Data.strikes[0])):
                 allstrikes.append((Data.strikes[:,row] + int(tmin/Paras.dt)).tolist())
                 allcerts.append(Data.strike_certs[:,row].tolist())
                 Paras.allcadences.append((np.max(allstrikes[-1]) - np.min(allstrikes[-1]))/(Paras.nbells-1))
         tmin = min(allstrikes[-1])*Paras.dt - 5.0
         tmax = min(tmin + Paras.overall_tcut, Paras.overall_tmax)
             
         #Update global class things
         Paras.first_change_limit = np.array(allstrikes[-1]) - int(tmin/Paras.dt) + 20
         nrows_count = int(min(len(Paras.allcadences), 20))
         Paras.cadence_ref = np.mean(Paras.allcadences[-nrows_count:])
         Paras.allstrikes = np.array(allstrikes)

         
     return np.array(allstrikes).T, np.array(allcerts).T
     
def establish_initial_rhythm(Paras):
    #Obtain various things about the ringing. What exactlythis does will depend on what's required from the situation
    #Hopefully remove a load of the bugs that seem to have appeared.
    
    #This function needs to establish rounds times and nominal data (if required)
    frequencies_fine = False
    if Paras.use_existing_freqs: 
        #Using existing frequency profile -- just use as is.
        if os.path.exists('%s%s_freq_quality.npy' % (Paras.frequency_folder, Paras.freqname)):
            check_data = np.load('%s%s_freq_quality.npy' % (Paras.frequency_folder, Paras.freqname))
            if check_data[0] == Paras.dt and check_data[1] == Paras.fcut_length:
                print('__________________________________________')
                print('Suitable frequency file already exists:')
                print('Average confidence: %.1f' % (100*check_data[2]), '%')
                print('Minimum confidence: %.1f' % (100*check_data[3]), '%')
                frequencies_fine = True
                if check_data[2] < 0.9:
                    print('Imported frequencies not great... Proceeding anyway but consider making new ones')
                    time.sleep(2.0)
                    
        #Find initial rounds using these informed frequencies rather than the nominals.
        Data = data(Paras, Audio, tmin = 0.0, tmax = Paras.reinforce_tmax) #This class contains all the important stuff, with outputs and things
        
        Paras.ringing_start, Paras.ringing_end = find_ringing_times(Paras, Data)
        Paras.reinforce_tmax = Paras.ringing_start*Paras.dt + Paras.reinforce_tmax
        Paras.rounds_tmax = Paras.ringing_start*Paras.dt  + Paras.rounds_tmax
        print('Ringing detected from approx. time %d seconds.' % (Paras.ringing_start*Paras.dt))

        #Find initial rounds using these informed frequencies rather than the nominals.
        Data = data(Paras, Audio, tmin = 0.0, tmax = Paras.reinforce_tmax) #This class contains all the important stuff, with outputs and things

        #Find strike probabilities from the nominals
        Data.strike_probabilities = find_strike_probabilities(Paras, Data, Audio, init = False, final = False)
        #Find the first strikes based on these probabilities. Hopefully some kind of nice pattern to the treble at least... 
        Paras.local_tmin = Paras.overall_tmin
        Paras.local_tint = int(Paras.overall_tmin/Paras.dt)
        Paras.stop_flag = False
        
        Paras.first_strikes, Paras.first_strike_certs = find_first_strikes(Paras, Data, Audio)
        Data.strikes, Data.strike_certs = Paras.first_strikes, Paras.first_strike_certs
            

        return Data

    if not frequencies_fine:
        if np.min(Paras.nominals) > 1.0:
            #Have nominal data here -- just do normal things
            Data = data(Paras, Audio, tmin = 0.0, tmax = Paras.reinforce_tmax) #This class contains all the important stuff, with outputs and things
            
            Paras.ringing_start, Paras.ringing_end = find_ringing_times(Paras, Data)
            Paras.reinforce_tmax = Paras.ringing_start*Paras.dt + Paras.reinforce_tmax
            Paras.rounds_tmax = Paras.ringing_start*Paras.dt  + Paras.rounds_tmax
            print('Ringing detected from approx. time %d seconds.' % (Paras.ringing_start*Paras.dt))

            Data = data(Paras, Audio, tmin = 0.0, tmax = Paras.reinforce_tmax) #This class contains all the important stuff, with outputs and things

            #Find strike probabilities from the nominals
            Data.strike_probabilities = find_strike_probabilities(Paras, Data, Audio, init = True, final = False)
            #Find the first strikes based on these probabilities. Hopefully some kind of nice pattern to the treble at least... 
            Paras.local_tmin = Paras.overall_tmin
            Paras.local_tint = int(Paras.overall_tmin/Paras.dt)
            Paras.stop_flag = False
    
            Paras.ringing_start, Paras.ringing_end = find_ringing_times(Paras, Data)
            
            Paras.first_strikes, Paras.first_strike_certs = find_first_strikes(Paras, Data, Audio)
            Data.strikes, Data.strike_certs = Paras.first_strikes, Paras.first_strike_certs
            
            return Data
        
        else:
            raise Exception('Need to provide nominal frequencies for now... Sorry.')
            '''
            Data = data(Paras, Audio, tmin = 0.0, tmax = Paras.reinforce_tmax) #This class contains all the important stuff, with outputs and things

            nominals = find_nominal_frequencies(Paras, Data, loudest_bell_from_back = 1)
            
            Paras.nominals = np.round(nominals*Paras.fcut_length).astype('int')
            
            Data = data(Paras, Audio, tmin = 0.0, tmax = Paras.reinforce_tmax) #This class contains all the important stuff, with outputs and things
            
            #Find strike probabilities from the nominals
            Data.strike_probabilities = find_strike_probabilities(Paras, Data, Audio, init = True, final = False)
            #Find the first strikes based on these probabilities. Hopefully some kind of nice pattern to the treble at least... 
            Paras.local_tmin = Paras.overall_tmin
            Paras.local_tint = int(Paras.overall_tmin/Paras.dt)
            Paras.stop_flag = False
    
            Paras.first_strikes, Paras.first_strike_certs = find_first_strikes(Paras, Data, Audio)
            Data.strikes, Data.strike_certs = Paras.first_strikes, Paras.first_strike_certs
        
            print(Paras.first_strikes, Paras.first_strike_certs)
            '''
            return Data

tower_number = 2

if tower_number == 0:
    fname = 'stedman_nics.wav'
    nominal_freqs = np.array([1439.,1289.5,1148.5,1075.,962.,861.])  #ST NICS
   
if tower_number == 1:  
    #fname = 'stockton_stedman.wav'
    #fname = 'stockton_finaltouch.wav'
    fname = 'stockton_all.wav'
    nominal_freqs = np.array([1892,1679,1582,1407,1252,1179,1046,930,828,780,693,617])

if tower_number == 2:    
    #fs, data = wavfile.read('audio/brancepeth.wav')
    fname = 'brancepeth_cambridge.wav'
    #fname = 'brancepeth_grandsire.wav'
    #fname = 'brancepeth.wav'
    #fname = 'brancepeth_firing.wav'
    nominal_freqs = np.array([1230,1099,977,924,821.5,733])

if tower_number == 3:
    fname = 'leeds2.wav'
    nominal_freqs = np.array([1554,1387,1307,1163,1037,976,872,776,692.5,653,581.5,518])

if tower_number == 4:
    fname = 'burley_cambridge.wav'
    nominal_freqs = np.array([1538,1372,1225,1158,1027,913])

if tower_number == 5:
    fname = 'ripon_touch4.m4a'
    nominal_freqs = np.array([1849,1648,1556,1385,1234.5,1162,1037,924,824,776.5,692,617.4])
    
if tower_number == 6:
    fname = 'york_1.m4a'
    nominal_freqs = np.array([1369,1218,1151,1023.5,912,861,767,684,608,575.5,512.5,456])
    
audio_folder = './audio/'
frequency_folder = './frequency_data/'
output_folder = './strike_times/'

use_existing_frequency_data = False   #If true, attempts to find existing frequency data that's fine. If not, does reinforcement.
existing_frequency_fname = fname[:-4]
overwrite_existing_frequency_data = True    #Replaces existing data even if the new one is worse.

nbells = len(nominal_freqs)

nominal_freqs = nominal_freqs[-nbells:]

tower_name = ''

#Input parameters which may need to be changed for given audio
overall_tmin = 0.0   #Can be smarter about this and get the amount of silence. Hopefully.
overall_tmax = 2000.0    #Max and min values for the audio signal (just trims overall and the data is then gone)

overall_tcut = 60.0

n_reinforces = 10  #Number of times the frequencies should be reinforced

#Import the data
Audio = audio_data(fname, audio_folder)

print('Imported audio length: %.2f seconds' % (len(Audio.signal)/Audio.fs))

overall_tmax = min(overall_tmax, len(Audio.signal)/Audio.fs)
#Establish parameters, some of which are hard coded into the class
Paras = parameters(Audio, nominal_freqs, overall_tmin, overall_tmax, overall_tcut, nbells = nbells)
Paras.fname = fname; Paras.frequency_folder = frequency_folder; Paras.freqname = existing_frequency_fname; Paras.output_folder = output_folder
Paras.use_existing_freqs = use_existing_frequency_data; Paras.overwrite_existing_freqs = overwrite_existing_frequency_data
Paras.n_reinforces = n_reinforces

print('Trimmed audio length: %.2f seconds' % (len(Audio.signal)/Audio.fs))
#print('Running assuming', Paras.nbells, 'bells')
Data = establish_initial_rhythm(Paras)

do_reinforcement(Paras, Data, Audio)

print('Frequencies imported or determined -- finding strike times throughout...')

Paras.allstrikes, Paras.allcerts = find_final_strikes(Paras, Audio)
    
plot_strikes(Paras)
save_strikes(Paras, fname[-4:])


    
    
    
    
    
    
    
    
