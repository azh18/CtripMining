import matplotlib as mpl

mpl.use('agg')
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rc

rc('font', **{'family': 'serif', 'serif': ['Times New Roman'], 'size': 18})
rc('text', usetex=True)
rc('lines', **{'linewidth': 2.0, 'marker': 'o', 'markersize': 0})


# labelFontSize=18
# tickFontSize=18
def plotScatterWithDensity(x, y, fileName=None):
    # # the random data
    # x = np.random.randn(1000)
    # y = np.random.randn(1000)
    # print(x)
    # print(y)

    fig, axScatter = plt.subplots(figsize=(8, 8))
    # the scatter plot:
    axScatter.scatter(x, y, color='cornflowerblue')
    axScatter.set_aspect(1.)
    plt.xlabel(r'Seat preference')
    plt.ylabel(r'Travel time preference')
    # plt.xticks(fontsize=tickFontSize)
    # plt.yticks(fontsize=tickFontSize)


    # create new axes on the right and on the top of the current axes # The first argument of the new_vertical(new_horizontal) method is  # the height (width) of the axes to be created in inches.
    divider = make_axes_locatable(axScatter)
    axHistx = divider.append_axes("top", 1.2, pad=0.3, sharex=axScatter)
    axHisty = divider.append_axes("right", 1.2, pad=0.3, sharey=axScatter)

    # make some labels invisible
    plt.setp(axHistx.get_xticklabels() + axHisty.get_yticklabels(), visible=False)

    # now determine nice limits by hand:
    binwidth = 0.05
    xymax = np.max([np.max(np.fabs(x)), np.max(np.fabs(y))])
    lim = (int(xymax / binwidth) + 1) * binwidth

    bins = np.arange(0, lim + binwidth, binwidth)
    axHistx.hist(x, bins=bins)
    axHisty.hist(y, bins=bins, orientation='horizontal')

    # the xaxis of axHistx and yaxis of axHisty are shared with axScatter,# thus there is no need to manually adjust the xlim and ylim of these# axis.

    # axHistx.axis["bottom"].major_ticklabels.set_visible(False)
    for tl in axHistx.get_xticklabels():
        tl.set_visible(False)
        axHistx.set_yticks([0, 100, 200, 300, 400])
    # axHistx.tick_params(labelsize=tickFontSize)

    # axHisty.axis["left"].major_ticklabels.set_visible(False)
    for tl in axHisty.get_yticklabels():
        tl.set_visible(False)
        axHisty.set_xticks([0, 100, 200])
    # axHisty.tick_params(labelsize=tickFontSize)

    plt.draw()
    if (fileName != None):
        plt.savefig(fileName)
    # plt.show()


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as path


def plotHistgram(data, nBins=50, fileName=None):
    fig, ax = plt.subplots()
    ax.set_position([0.2, 0.2, 0.7, 0.7])
    n, bins = np.histogram(data, nBins)

    # get the corners of the rectangles for the histogram
    left = np.array(bins[:-1])
    right = np.array(bins[1:])
    bottom = np.zeros(len(left))
    top = bottom + n

    # we need a (numrects x numsides x 2) numpy array for the path helper
    # function to build a compound path
    XY = np.array([[left, left, right, right], [bottom, top, top, bottom]]).T

    # get the Path object
    barpath = path.Path.make_compound_path_from_polys(XY)

    # make a patch out of it
    patch = patches.PathPatch(barpath, facecolor='blue', edgecolor='gray', alpha=0.8)
    ax.add_patch(patch)

    # update the view limits
    ax.set_xlim(left[0], right[-1])
    ax.set_ylim(bottom.min(), top.max())

    plt.xlabel(r'Distance')
    plt.ylabel(r'Histgram')
    if (fileName != None):
        plt.savefig(fileName)


from myplotlib import plotBars


def plotDetailedInfo(data1, data2):
    # legends=['Passenger A','Passenger B','Passenger C']
    # xlabel_str='Takeoff hours'
    # ylabel_str='Proporation of flights at different hours'
    # xticks_str=[str(e) for e in range(6,24)]
    # plotBars(xlabel_str,ylabel_str,3,data1,legends, xticks_str,None,'Proporation_hours.png')

    legends = ['Passenger A', 'Passenger B', 'Passenger C']
    xlabel_str = 'Takeoff days'
    ylabel_str = 'Proporation of flights in different days'
    xticks_str = ['Workday', 'Weekend', 'Holiday']
    plotBars(xlabel_str, ylabel_str, 3, data2, legends, xticks_str, None, 'Proporation_days.png')
