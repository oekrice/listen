import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.ndimage import gaussian_filter1d
from scipy import stats
from scipy.optimize import minimize_scalar, curve_fit

def find_ideal_hgap(cut_init, gap_init, row, nbells):
    #Finds ideal handstroke gap for two rows (given in cut_init)
    def find_r(gap):
        cut = np.array(cut_init)
        cut[nbells:] = cut_init[nbells:] - gap
        _, _, r, _, _ = stats.linregress(np.arange(nbells*2), cut)
        return (1-r)

    res = minimize_scalar(find_r, bounds=(10, 400), method='bounded')

    return res.x

#I think this is the easiest way to do it, and quickest. Then go through again with a priori data

def find_all_gaps(alltimes, nbells, nrows):
    all_gaps = np.zeros(nrows) #Handstroke gaps BEFORE each row. Backstrokes will just be zero
    gap_init = (alltimes[nbells*2-1] - alltimes[0])/(nbells*2)
    all_gaps[0] = gap_init

    for row in range(1,nrows-1,2): #Starting the cut on each backstroke
        start = row*nbells
        end   = (row+2)*nbells
        cut = np.array(alltimes[start:end])   #Contains 24 blows with a backstroke change and handstroke change
        #Find ideal handstroke gap
        gap = find_ideal_hgap(cut, gap_init, row, nbells)

        cut[nbells:] = cut[nbells:] - gap
        all_gaps[row+1] = gap

    print('Ideal handstroke gaps found...')
    return all_gaps

def find_predicted_gaps(all_ideal_gaps, nbells, nrows, ngaps):
    #ONLY using gaps from previous rows, predict the 'best' amount of handstroke gap.
    #Perhaps fit a quadratic to the previous few, if possible
    def f(x,a,b,c):
        return a*x**2 + b*x + c

    print('Finding best possible handstroke gaps in the moment')
    all_gaps = np.zeros(nrows)
    minfits = 6
    nfits = ngaps #Number of handstroke rows to establish the rhythm from
    for row in range(0,nrows,2):
        if row == 0:
            all_gaps[row] = all_ideal_gaps[0]
        elif row < minfits*2:  #Take the mean of these
            all_gaps[row] = np.mean(all_ideal_gaps[:row])*2
        else:
            alldata = all_ideal_gaps[max(row-2*nfits,0):row:2]
            popt, pcov = curve_fit(f, np.arange(len(alldata)), alldata)
            all_gaps[row] = f(len(alldata), *popt)

            #plt.plot(all_ideal_gaps[max(row-nfits,0):row:2])
            #plt.show()

    return all_gaps


def find_ideal_times(alltimes, nbells, ncount = 24, ngaps = 6, reference_data = []):

    alltimes = np.array(alltimes)

    nrows = int(len(alltimes)/nbells)

    print('Number of rows', len(alltimes)/nbells)

    all_ideal_gaps = find_all_gaps(alltimes, nbells, nrows)

    all_predicted_gaps = find_predicted_gaps(all_ideal_gaps, nbells, nrows, ngaps = ngaps)

    print('Handstroke gaps determined...')
    #Actually finds the ideal strike time of each bell, based on the predicted handstroke gaps and the preceding strikes (up to a point to be determined.)
    #Bell number is irrelevant
    def f1(x,a,b):
        return a*x + b

    def f2(x,a,b,c):
        return a*x**2 + b*x + c

    def adjust_times(data, row_number, n_adjust,position):
        #Remove the effect of the handstroke gaps -- keep the individual times the same but rewrite the preceding few as necessary, essentially assuming everything before comes earlier
        first_change = max(0,row_number - n_adjust)
        count = 0
        for row_change in range(row_number,first_change,-1):
            #Figure out which ones to retrofit
            limit = count*nbells + position
            #print(row_change, first_change, position, all_predicted_gaps[row_change], limit)
            count += 1
            #print(data)
            #print(all_predicted_gaps[row_change])
            #print(data[1:] - data[:-1])
            #print('a',data)

            if limit == 0:
                data[:] = data[:] + all_predicted_gaps[row_change]
            elif limit < len(data):
                data[:-limit] = data[:-limit] + all_predicted_gaps[row_change]
            #print('b',data)
        return data

    nback = ncount  #Influenced by the preceding change in its entirety. CAN CHANGE THIS.
    all_ideals = np.zeros(nrows*nbells)
    n_adjust = int(nback/nbells)

    if len(alltimes) != len(all_ideals):
        raise Exception('Not a complete number of changes -- aborting')

    print('Finding individual strikes')

    for fname in os.listdir('./plots'):
        os.system('rm ./plots/' + fname)
    for strike in range(len(all_ideals)):
    #for strike in range(0,1000):
        row_position = strike%(nbells)  #Row position up to nbells
        row_number = strike//nbells
        if strike == 0 or strike == 1:  #First strikes are naturally perfect
            all_ideals[strike] = alltimes[strike]

        elif strike < nback:  #Not enough preceding data, assume a linear interpolation from preceding
            data = np.array(alltimes[:strike])
            data = adjust_times(data, row_number, n_adjust,row_position)  #Adjust to take into account handstroke gaps
            basis = np.arange(len(data)) - len(data) + row_position

            popt, pcov = curve_fit(f1, basis, data)
            all_ideals[strike] = f1(row_position, *popt)


            if False:

                fig, ax = plt.subplots()

                basis_extend = np.arange(basis[0], basis[-1] + 5)
                plot_xs = np.linspace(basis[0],basis_extend[-1],100)
                plot_ys = f1(plot_xs, *popt)
                #plt.plot(plot_xs,plot_ys,zorder=1)
                plt.plot(basis, data,zorder=10, c= 'black', label = 'Actual Times')
                plt.scatter(basis, data, c = 'black',zorder=10)
                ax.set_xticks(basis_extend)
                labs = basis_extend%nbells + 1
                ax.set_xticklabels(labs)

                if len(reference_data) > 0:
                    models = ['Contest Model','HawkEar Model', 'RWP Model']
                    for model in models:
                        model_data = np.array(reference_data[model])[strike-nback:strike]
                        model_data = adjust_times(model_data, row_number, n_adjust,row_position)
                        plt.scatter(basis[-1] + 1, np.array(reference_data[model])[strike], label = model)
                        plt.plot(basis, model_data,zorder=1)

                    model_data = np.array(all_ideals)[strike-nback:strike]
                    model_data = adjust_times(model_data, row_number, n_adjust,row_position)

                    plt.scatter(basis[-1] + 1, all_ideals[strike], label = 'My Model')
                    plt.plot(basis, model_data,zorder=1)

                ax.set_xlim(basis[-12], basis[-1] + 3)
                ax.set_ylim(all_ideals[strike]-2500,all_ideals[strike] + 1000 )
                plt.legend()
                #plt.scatter(len(data), all_ideals[strike])
                plt.savefig('./plots/%04d' % strike)
                plt.close()

        else:  #Is enough, try a quadratic one
            data = np.array(alltimes[strike-nback:strike])
            data = adjust_times(data, row_number, n_adjust,row_position)  #Adjust to take into account handstroke gaps
            basis = np.arange(len(data)) - len(data) + row_position
            popt, pcov = curve_fit(f2, basis, data)
            all_ideals[strike] = f2(row_position, *popt)
            if False:
                fig, ax = plt.subplots()

                basis_extend = np.arange(basis[0], basis[-1] + 5)
                plot_xs = np.linspace(basis[0],basis_extend[-1],100)
                plot_ys = f2(plot_xs, *popt)
                #plt.plot(plot_xs,plot_ys,zorder=1)
                plt.plot(basis, data,zorder=10, c= 'black', label = 'Actual Times')
                plt.scatter(basis, data, c = 'black',zorder=10)
                ax.set_xticks(basis_extend)
                labs = basis_extend%nbells + 1
                ax.set_xticklabels(labs)

                if len(reference_data) > 0:
                    if False:
                        models = ['Contest Model','HawkEar Model', 'RWP Model']
                        for model in models:
                            model_data = np.array(reference_data[model])[strike-nback:strike]
                            model_data = adjust_times(model_data, row_number, n_adjust,row_position)
                            plt.scatter(basis[-1] + 1, np.array(reference_data[model])[strike], label = model)
                            plt.plot(basis, model_data,zorder=1)

                    model_data = np.array(all_ideals)[strike-nback:strike]
                    model_data = adjust_times(model_data, row_number, n_adjust,row_position)

                    plt.scatter(basis[-1] + 1, all_ideals[strike], label = 'My Model')
                    plt.plot(basis, model_data,zorder=1)


                ax.set_xlim(basis[-12], basis[-1] + 3)
                centre = np.mean(data[-10:])

                ax.set_ylim(centre-2500,centre + 2500 )
                plt.legend()
                #plt.scatter(len(data), all_ideals[strike])
                plt.savefig('./plots/%04d' % strike)
                plt.close()

    #plt.plot(alltimes)
    #errors = (alltimes - all_ideals)
    #plt.plot(errors[:])
    #plt.show()
    print('Ideal times determined')
    return all_ideals

#all_ideal_times = find_ideal_times(alltimes)

