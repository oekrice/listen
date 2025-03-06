#Plotting strikeometer things nicely

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from strike_model import find_ideal_times
from scipy import interpolate

touch_number = 0
plt.style.use('default')
cmap = plt.cm.nipy_spectral
cmap = plt.cm.inferno

#tower_name = 'Brancepeth'
#tower_name = 'Stockton'
#tower_name = 'Nics'
tower_name = 'leeds'
#tower_name = 'burley'

data_filename = ('%s%d.csv' % (tower_name, touch_number))  #Could automate this if necessary
data_filename = ('%s%d.csv' % (tower_name, touch_number))  #Could automate this if necessary

data_filename = ('./strike_times/current_run.csv')  #Could automate this if necessary

model = 'My Model'
nbins = 50
max_error_plot = 150 #in ms
bar_width = 0.3

data = pd.read_csv(data_filename)
nbells = int(np.max(data['Bell No']))

alldiags = np.zeros((3,3,nbells))   #Type, stroke, bell

titles = ['All blows', 'Handstrokes', 'Backstrokes']
#cs = ['greenyellow', 'chartreuse', 'lawngreen']

#Bodge to fix the dodgy bell data. The three is logged two changes too early.

count_test = 48
gap_test = 20
optimise = False

if optimise:
    print('Optimising parameters')
    best_var = 1e6
    
    for count_test in range(nbells*2, nbells*16, 4):    
    #for count_test in range(nbells*2):  #can use this to minimise std error
        #Vary the parameters to minimise the variance
        ideal_times = find_ideal_times(data['Actual Time'], nbells, ncount = count_test, ngaps = gap_test, reference_data = data)
        data['My Model'] = ideal_times
        allerrors = np.array(data['Actual Time'] - data[model])
        
        diffs = np.array(data['Actual Time'])[1:] - np.array(data['Actual Time'])[:-1]
        cadence = np.mean(diffs)*(2*nbells)/(2*nbells + 1)
        
        allerrors = allerrors*[np.abs(allerrors) < cadence*0.75]
        variance = (np.sum(allerrors**2)/len(allerrors))/1e4
        #print('Variance with significant errors removed:', count_test, gap_test, variance)
        if variance < best_var:
            best_var = variance
            best_count = count_test
    count_test = best_count
    best_var = 1e6

    for gap_test in range(20,80,4):
        ideal_times = find_ideal_times(data['Actual Time'], nbells, ncount = count_test, ngaps = gap_test, reference_data = data)
        data['My Model'] = ideal_times
        allerrors = np.array(data['Actual Time'] - data[model])
        
        diffs = np.array(data['Actual Time'])[1:] - np.array(data['Actual Time'])[:-1]
        cadence = np.mean(diffs)*(2*nbells)/(2*nbells + 1)
        
        allerrors = allerrors*[np.abs(allerrors) < cadence*0.75]
        variance = (np.sum(allerrors**2)/len(allerrors))/1e4
        #print('Variance with significant errors removed:', count_test, gap_test, variance)
        if variance < best_var:
            best_var = variance
            best_gap = gap_test
    
    gap_test = best_gap
    print('Best count', best_count, 'Best Gap', best_gap, 'Variance', best_var)
    
ideal_times = find_ideal_times(data['Actual Time'], nbells, ncount = count_test, ngaps = gap_test, reference_data = data)
data['My Model'] = ideal_times

data.to_csv('%s.csv' % tower_name)  

nstrikes = len(data['Actual Time'])
nrows = int(nstrikes//nbells)


toprint = []
orders = []; starts = []; ends = []
for row in range(nrows):
    yvalues = np.arange(nbells) + 1
    actual = np.array(data['Actual Time'][row*nbells:(row+1)*nbells])
    target = np.array(data['My Model'][row*nbells:(row+1)*nbells])
    bells =   np.array(data['Bell No'][row*nbells:(row+1)*nbells])  
    order = np.array([val for _, val in sorted(zip(actual, yvalues), reverse = False)])
    targets = np.array([val for _, val in sorted(zip(actual, actual-target), reverse = False)])
    toprint.append(actual-target)
    orders.append(bells)
    starts.append(np.min(target))
    ends.append(np.max(target))
    #starts.append(np.min(actual))
    #ends.append(np.max(actual))
#An attempt to plot the method?
if True:
    rows_per_plot = 6*int(nrows//24)
    nplotsk = nrows//rows_per_plot + 1
    rows_per_plot = int(nrows/nplotsk) + 2
    fig,axs = plt.subplots(1,nplotsk, figsize = (10,10))
    for plot in range(nplotsk):
        if nplotsk > 1:
            ax = axs[plot]
        else:
            ax = axs
        for bell in range(1,nbells+1):#nbells):
            points = []
            belldata = data.loc[data['Bell No'] == bell]
            errors = np.array(belldata['Actual Time'] - belldata[model])
            targets = np.array(belldata['My Model'])
            for row in range(len(belldata)):
                #Find linear position... Linear interpolate?
                target_row = np.array(data['My Model'][row*nbells:(row+1)*nbells])
                ys = np.arange(1,nbells+1)
                f = interpolate.interp1d(target_row, ys, fill_value = "extrapolate")
                rat = f(np.array(belldata['Actual Time'])[row])

                #rat = (np.array(belldata['Actual Time'])[row] - starts[row])/(ends[row] - starts[row])
                points.append(rat)
            ax.plot(points, np.arange(len(belldata)),label = bell, c = cmap(np.linspace(0,0.8,nbells)[bell-1]))
            ax.plot((bell)*np.ones(len(points)), np.arange(len(belldata)), c = 'black', linewidth = 0.5, linestyle = 'dotted', zorder = 0)
        for row in range(len(belldata)):
            ax.plot(np.arange(-1,nbells+3), row*np.ones(nbells+4), c = 'black', linewidth = 0.5, linestyle = 'dotted', zorder = 0)
        
        plt.gca().invert_yaxis()
        ax.set_ylim((plot+1)*rows_per_plot, plot*rows_per_plot)
        ax.set_xlim(-1,nbells+2)
        ax.set_xticks([])
        ax.set_aspect('equal')
        #if plot == nplotsk-1:
        #    plt.legend()
        #ax.set_yticks([])
    plt.tight_layout()
    plt.savefig('./plots/%dblueline.png' % touch_number)
    plt.show()

diffs = np.array(data['Actual Time'])[1:] - np.array(data['Actual Time'])[:-1]
cadence = np.mean(diffs)*(2*nbells)/(2*nbells + 1)


for plot_id in range(3):
    #Everything, then handstrokes, then backstrokes

    fig, axs = plt.subplots(3,4, figsize = (10,7))
    allerrors = []
    for bell in range(1,nbells+1):#nbells):
        #Extract data for this bell
        belldata = data.loc[data['Bell No'] == bell]

        errors = np.array(belldata['Actual Time'] - belldata[model])

        #Attempt to remove outliers (presumably method mistakes, hawkear being silly or other spannering)
        maxlim = cadence*0.75
        minlim = -cadence*0.75

        #Trim for the appropriate stroke
        if plot_id == 1:
            errors = errors[::2]
        if plot_id == 2:
            errors = errors[1::2]

        #Adjust stats to disregard these properly
        count = len(errors)
        count -= np.sum(errors > maxlim)
        count -= np.sum(errors < minlim)

        errors[errors > maxlim] = 0.0
        errors[errors < minlim] = 0.0


        #Diagnostics
        alldiags[0,plot_id,bell-1] = np.sum(errors)/count
        alldiags[1,plot_id,bell-1] = np.sqrt(np.sum((errors-np.sum(errors)/count)**2)/count)
        alldiags[2,plot_id,bell-1] = np.sqrt(np.sum(errors**2)/count)

        allerrors += np.sum(errors)/count
        ax = axs[(bell-1)//4, (bell-1)%4]

        ax.set_title('Bell %d' % bell)
        bin_bounds = np.linspace(-max_error_plot, max_error_plot, nbins+1)
        n, bins, _ = ax.hist(errors, bins = bin_bounds)

        curve = gaussian_filter1d(n, sigma = nbins/20)
        ax.plot(0.5*(bins[1:] + bins[:-1]),curve, c= 'black')
        ax.set_xlim(-max_error_plot, max_error_plot)
        ax.set_ylim(0,10)
        ax.plot([0,0],[0,max(n)], linewidth = 2)
        ax.set_yticks([])
    plt.suptitle(titles[plot_id])
    plt.tight_layout()
    #plt.savefig('./plots/%dhists_%d.png' % (touch_number, plot_id))
    plt.close()

fig, axs = plt.subplots(3, figsize = (12,7))

data_titles = ['Avg. Error', 'Std. Dev. from Average', 'Std. Dev. From Ideal']

x = np.arange(nbells)
for plot_id in range(3):
    ax = axs[plot_id]

    xmin = np.min(alldiags[plot_id,:,:])*0.9
    xmax = np.max(alldiags[plot_id,:,:])*1.1
    

    rects0 = ax.bar(x-bar_width*1,alldiags[plot_id,0,:],bar_width,label = titles[0], color='lightgray')
    ax.bar_label(rects0, padding = 3, fmt = '%d')

    rects1 = ax.bar(x-bar_width*0,alldiags[plot_id,1,:],bar_width,label = titles[1], color='red')
    ax.bar_label(rects1, padding = 3, fmt = '%d')

    rects2 = ax.bar(x+bar_width*1.0,alldiags[plot_id,2,:],bar_width,label = titles[2], color='blue')
    ax.bar_label(rects2, padding = 3, fmt = '%d')

    ax.set_xticks(np.arange(nbells), np.arange(1,nbells+1))
    ax.set_title(data_titles[plot_id])
    if plot_id > 0:
        ax.set_ylim(xmin, xmax)
    if plot_id == 0:
        ax.legend()

plt.tight_layout()
plt.savefig('./plots/%dbars.png' % touch_number)
#plt.show()

plt.close()

fig, axs = plt.subplots(3, figsize = (10,7))

for plot_id in range(3):
    #Everything, then handstrokes, then backstrokes

    ax = axs[plot_id]

    
    for bell in range(1,nbells+1):#nbells):
        #Extract data for this bell
        belldata = data.loc[data['Bell No'] == bell]

        errors = np.array(belldata['Actual Time'] - belldata[model])

        #Attempt to remove outliers (presumably method mistakes, hawkear being silly or other spannering)
        maxlim = np.percentile(errors,98)+50
        minlim = np.percentile(errors,2)-50

        #Trim for the appropriate stroke
        if plot_id == 1:
            errors = errors[::2]
        if plot_id == 2:
            errors = errors[1::2]

        #errors[errors > maxlim] = 0.0
        #errors[errors < minlim] = 0.0

        #Box plot data
        ax.boxplot(errors,positions = [bell], sym = 'x', widths = 0.35, zorder = 1)
    #ax.axhline(0.0, c = 'black', linestyle = 'dashed')

    ax.set_ylim(-150,150)
    ax.set_title(titles[plot_id])

plt.tight_layout()
plt.savefig('./plots/%dboxes.png' % touch_number)
#plt.show()
plt.close()

