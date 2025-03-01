#Plotting strikeometer things nicely

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from strike_model import find_ideal_times
from scipy import interpolate

plt.style.use('default')
cmap = plt.cm.rainbow

#tower_name = 'Brancepeth'
tower_name = 'Stockton'
#tower_name = 'Nics'

data_filename = ('%s.csv' % tower_name)  #Could automate this if necessary

#data_filename = ('ym.20250202-1451.4.vcga.bl.csv')  #Could automate this if necessary
#data_filename = ('ym.20250202-1435.3.eslh.bl.csv')  #Could automate this if necessary
#data_filename = ('bristol.20240323-1256.03C.ezos.bl.csv')
#data_filename = ('ym.20240921-1340.2.jziw.bl.csv')  #Could automate this if necessary

nbells = 12
model = 'My Model'
nbins = 50
max_error_plot = 150 #in ms
bar_width = 0.3

data = pd.read_csv(data_filename)

alldiags = np.zeros((3,3,nbells))   #Type, stroke, bell

titles = ['All blows', 'Handstrokes', 'Backstrokes']
#cs = ['greenyellow', 'chartreuse', 'lawngreen']

#Bodge to fix the dodgy bell data. The three is logged two changes too early.

count_test = nbells*4
gap_test = 40
#for count_test in range(nbells*2):  #can use this to minimise std error
#    for gap_test in range(16,17):
if model == 'My Model':
    ideal_times = find_ideal_times(data['Actual Time'], nbells, ncount = count_test, ngaps = gap_test, reference_data = data)
    data['My Model'] = ideal_times
    allerrors = np.array(data['Actual Time'] - data[model])
    std = np.sqrt(np.sum(allerrors**2)/len(allerrors))
    print('std', count_test, gap_test, std)

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
    nplotsk = nrows//36 + 1
    rows_per_plot = 2*int(nrows/nplotsk/2) + 2
    fig,axs = plt.subplots(1,nplotsk, figsize = (10,10))
    for plot in range(nplotsk):
        ax = axs[plot]
        
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
            ax.plot(points, np.arange(len(belldata)),label = bell)#, c = cmap(np.linspace(0,1,nbells)[bell-1]))
            ax.plot((bell)*np.ones(len(points)), np.arange(len(belldata)), c = 'black', linewidth = 0.5, linestyle = 'dotted', zorder = 0)
        for row in range(len(belldata)):
            ax.plot(np.arange(-1,nbells+3), row*np.ones(nbells+4), c = 'black', linewidth = 0.5, linestyle = 'dotted', zorder = 0)
        
        plt.gca().invert_yaxis()
        ax.set_ylim((plot+1)*rows_per_plot, plot*rows_per_plot)
        ax.set_xlim(-1,nbells+2)
        ax.set_xticks([])
        ax.set_aspect('equal')
        if plot == nplotsk-1:
            plt.legend()
        #ax.set_yticks([])
    plt.tight_layout()
    plt.show()


for plot_id in range(3):
    #Everything, then handstrokes, then backstrokes

    fig, axs = plt.subplots(3,4, figsize = (10,7))
    allerrors = []
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

    plt.suptitle(titles[plot_id])
    plt.tight_layout()
    plt.savefig('./data_plots/plot%d.png' % plot_id)
    plt.close()

fig, axs = plt.subplots(3, figsize = (12,7))

data_titles = ['Avg. Error', 'Std. Dev. from Average', 'Std. Dev. From Ideal']

x = np.arange(nbells)
for plot_id in range(3):
    ax = axs[plot_id]

    xmin = np.min(alldiags[plot_id,:,:])*0.9
    xmax = np.max(alldiags[plot_id,:,:])*1.1
    

    rects0 = ax.bar(x-bar_width*1,alldiags[plot_id,0,:],bar_width,label = titles[0])
    ax.bar_label(rects0, padding = 3, fmt = '%d')

    rects1 = ax.bar(x-bar_width*0,alldiags[plot_id,1,:],bar_width,label = titles[1])
    ax.bar_label(rects1, padding = 3, fmt = '%d')

    rects2 = ax.bar(x+bar_width*1.0,alldiags[plot_id,2,:],bar_width,label = titles[2])
    ax.bar_label(rects2, padding = 3, fmt = '%d')

    ax.set_xticks(np.arange(nbells), np.arange(1,nbells+1))
    ax.set_title(data_titles[plot_id])
    if plot_id > 0:
        ax.set_ylim(xmin, xmax)
    if plot_id == 0:
        ax.legend()

plt.tight_layout()
plt.savefig('./data_plots/data.png')
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
        ax.boxplot(errors,positions = [bell], sym = 'x', widths = 0.35)
    ax.set_ylim(-150,150)
    ax.set_title(titles[plot_id])

plt.tight_layout()
plt.savefig('./data_plots/boxplots.png')
#plt.show()
plt.close()

