import matplotlib as mpl

mpl.use('agg')
from matplotlib import rc
import matplotlib.pylab as plt

rc('font', **{'family': 'serif', 'serif': ['Times New Roman'], 'size': 18})
rc('text', usetex=True)
rc('lines', **{'linewidth': 2.0, 'marker': 'o', 'markersize': 0})

WORK_PATH = 'plots'

# plot style settings
# FONT_SIZE=18
# LINE_WIDTH=2
LINE_STYLES = ['-', '-', '-', '-']
COLORS = ['k', 'b', 'y', 'r']
MARKERS = ['o', '^', 'v''*']

import numpy as np
from os.path import join


def plotLines(xlabel_str, ylabel_str, nlines, data, legends, axis_range=None, fig_name='lineplot.pdf'):
    fig = plt.figure(figsize=(8, 6))
    for k in range(0, nlines):
        (x, y) = data[k]
        plt.plot(x, y, color=COLORS[k], linestyle=LINE_STYLES[k], label=legends[k], marker=MARKERS[k])
    if axis_range is not None:
        # plt.axis([29,151,0,9000])
        plt.axis(axis_range)
    plt.xlabel(xlabel_str)
    plt.ylabel(ylabel_str)
    plt.savefig(join(WORK_PATH, fig_name))
    plt.close(fig)


BAR_FACE_COLORS = ['maroon', 'navy', 'darkolivegreen', 'dimgray']


# BAR_FACE_COLORS=['r','b','y','g']
def plotBars(xlabel_str, ylabel_str, nlines, data, legends, xticks_str=None, axis_range=None, fig_name='barplot.pdf',
             add_text=True, text_is_int=True):
    x = data[0]
    N = len(data[1])
    ind = np.arange(0, 3 * N, 3) + 0.5  # the x locations for the groups
    print(ind)
    bar_width = 3 / nlines  # the width of the bars
    print(data[1])
    print(data[2])
    print(nlines)

    # fig=plt.figure(figsize=(26,13))
    fig = plt.figure(figsize=(8, 6))
    ax = plt.subplot(111)
    rects = []
    for k in range(0, nlines):
        rect = ax.bar(ind + (k - (nlines - 1.) / 2) * bar_width, data[k + 1], width=bar_width, color=BAR_FACE_COLORS[k],
                      align='center', label=legends[k])
        rects.append(rect)
    ax.set_xlabel(xlabel_str)
    ax.set_ylabel(ylabel_str)
    ax.set_xlim(xmin=-1)
    ax.set_xticks(ind)
    if xticks_str is not None:
        ax.set_xticklabels(xticks_str, fontsize=12)
    elif x is not None:
        ax.set_xticklabels([str(e) for e in x])
    if legends is not None:
        ax.legend(rects, legends, loc=1)

    def autolabel(rects):  # attach some text labels
        for rect in rects:
            height = rect.get_height()
            if text_is_int is True:
                t = '%d' % int(height)
            else:  # text is a float number
                t = '%s' % str(round(height, 3))
            ax.text(rect.get_x() + rect.get_width() / 2., 1.02 * height, t, ha='center', va='bottom', fontsize=14)

    if add_text is True:
        for rect in rects:
            autolabel(rect)

    plt.savefig(join(WORK_PATH, fig_name))
    plt.close(fig)


# #legend location
# String	Number
# upper right	1
# upper left	2
# lower left	3
# lower right	4
# right	5
# center left	6
# center right	7
# lower center	8
# upper center	9
# center	10

from scipy.stats.kde import gaussian_kde


def plotPDF(data_samples, xlabel_str, ylabel_str, fig_name='pdf.pdf', xais_range=None, N=1000):
    # this create the kernel, given an array it will estimate the probability over that values
    kde = gaussian_kde(data_samples)
    # these are the values over wich your kernel will be evaluated
    dist_space = np.linspace(min(data_samples), max(data_samples), N)
    # plot the results
    fig = plt.figure(figsize=(8, 6))
    x = dist_space
    y = kde(dist_space)
    plt.plot(x, y, markersize=0)  # pdf line with no markers
    if xais_range is not None:
        plt.xlim(xais_range[0], xais_range[1])
    # plot a vertical line and a honrizontal line
    i = np.argmax(y)
    xline = x[i]
    yline = y[i]
    plt.axvline(x=xline, linewidth=1, markersize=0, color='b', linestyle='--')
    plt.axhline(y=yline, linewidth=1, markersize=0, color='b', linestyle='--')
    # plt.xticks(list(plt.xticks()[0]) + [xline])
    # plt.yticks(list(plt.yticks()[0]) + [yline])
    plt.xticks(replaceNearestElement(plt.xticks()[0], xline))
    plt.yticks(replaceNearestElement(plt.yticks()[0], yline))

    plt.xlabel(xlabel_str)
    plt.ylabel(ylabel_str)
    plt.savefig(join(WORK_PATH,fig_name))
    # plt.savefig(fig_name)
    plt.close(fig)


def replaceNearestElement(data_array, v):
    i = np.argmin(abs(data_array - v))
    data_array[i] = v
    return data_array


def plotCDF(data_samples, xlabel_str, ylabel_str, fig_name='cdf.pdf', xais_range=None, N=1000):
    # this create the kernel, given an array it will estimate the probability over that values
    kde = gaussian_kde(data_samples)
    # these are the values over wich your kernel will be evaluated
    dist_space = np.linspace(min(data_samples), max(data_samples), N)
    cdf = np.cumsum(kde(dist_space))
    cdf /= cdf[-1]  # normalized
    print(min(data_samples), max(data_samples))
    # plot the results
    fig = plt.figure(figsize=(8, 6))
    plt.plot(dist_space, cdf, markersize=0)  # pdf line with no markers
    plt.xlabel(xlabel_str)
    plt.ylabel(ylabel_str)
    if xais_range is not None:
        plt.xlim(xais_range[0], xais_range[1])
    # plt.savefig(join(WORK_PATH,fig_name))
    plt.savefig(fig_name)
    plt.close(fig)


def plotFreq(data_samples, xlabel_str, ylabel_str, fig_name='cdf.pdf', xais_range=None, yaxis_range=None,
             xtick_gap=None, ytick_gap=None, shrinkScale=None, N=1000, yaxis_log=False):
    # this create the kernel, given an array it will estimate the probability over that values
    # kde = gaussian_kde( data_samples )
    # these are the values over wich your kernel will be evaluated
    # dist_space = np.linspace( min(data_samples), max(data_samples), N )
    # cdf=np.cumsum(kde(dist_space))
    # cdf/=cdf[-1] #normalized
    # print(min(data_samples),max(data_samples))
    # plot the results
    data_samples = np.array(data_samples)
    sample_bins = np.arange(np.amin(data_samples) - 0.5, np.amax(data_samples) + 0.5, 1)
    data_hist, data_bin = np.histogram(data_samples, sample_bins)
    # shrinkscale: make the y num become small, for example: 50000->500
    if shrinkScale is not None:
        data_hist /= shrinkScale
    fig = plt.figure(figsize=(8, 6))
    # plt.plot(dist_space, cdf,markersize=0)  #pdf line with no markers
    if yaxis_log:
        plt.semilogy(data_bin[0:len(data_bin) - 1] + 0.5, data_hist, lw=2)
    else:
        plt.plot(data_bin[0:len(data_bin) - 1] + 0.5, data_hist, lw=2, color='b')
    plt.xlabel(xlabel_str)
    plt.ylabel(ylabel_str)
    if xais_range is not None:
        plt.xlim(xais_range[0], xais_range[1])
    if yaxis_range is not None:
        plt.ylim(yaxis_range[0], yaxis_range[1])
    if xtick_gap is not None:
        x_tick = np.arange(xais_range[0], xais_range[1] + xtick_gap, xtick_gap)
        plt.xticks(x_tick)
    if ytick_gap is not None:
        y_tick = np.arange(yaxis_range[0], yaxis_range[1] + ytick_gap, ytick_gap)
        plt.yticks(y_tick)
    plt.savefig(join(WORK_PATH, fig_name))
    # plt.savefig(fig_name)
    plt.close(fig)


def plotGivenFreq(x, y, xlabel_str, ylabel_str, fig_name='cdf.pdf', xais_range=None, yaxis_range=None,
             xtick_gap=None, ytick_gap=None, shrinkScale=None, N=1000, yaxis_log=False):
    # this create the kernel, given an array it will estimate the probability over that values
    # kde = gaussian_kde( data_samples )
    # these are the values over wich your kernel will be evaluated
    # dist_space = np.linspace( min(data_samples), max(data_samples), N )
    # cdf=np.cumsum(kde(dist_space))
    # cdf/=cdf[-1] #normalized
    # print(min(data_samples),max(data_samples))
    # plot the results
    # shrinkscale: make the y num become small, for example: 50000->500
    if shrinkScale is not None:
        y /= shrinkScale
    fig = plt.figure(figsize=(8, 6))
    # plt.plot(dist_space, cdf,markersize=0)  #pdf line with no markers
    if yaxis_log:
        plt.semilogy(x, y, lw=2)
    else:
        plt.plot(x, y, lw=2)
    plt.xlabel(xlabel_str)
    plt.ylabel(ylabel_str)
    if xais_range is not None:
        plt.xlim(xais_range[0], xais_range[1])
    if yaxis_range is not None:
        plt.ylim(yaxis_range[0], yaxis_range[1])
    if xtick_gap is not None:
        x_tick = np.arange(xais_range[0], xais_range[1] + xtick_gap, xtick_gap)
        plt.xticks(x_tick)
    if ytick_gap is not None:
        y_tick = np.arange(yaxis_range[0], yaxis_range[1] + ytick_gap, ytick_gap)
        plt.yticks(y_tick)
    plt.savefig(join(WORK_PATH, fig_name))
    # plt.savefig(fig_name)
    plt.close(fig)


if __name__ == '__main__':
    # # Example data
    # t = np.arange(0.0, 1.0 + 0.01, 0.01)
    # s1 = np.cos(4 * np.pi * t) + 2
    # s2 = np.cos(8 * np.pi * t) + 5
    # xlabel_str=r'\textbf{time} (s)'
    # ylabel_str='Number of passengers'
    # plotLines(xlabel_str,ylabel_str,2,((t,s1),(t,s2)),('line1','line2'))

    # #Example data
    # x = [1, 2, 3, 10,100,1000]
    # y1 = [4, 9, 2, 29,10.5,31]
    # y2=[1,2,3,4,2,13]
    # y3=[11,12,13,31,32,23]
    # y4=[11,52,43,31,32,45]
    # legends=['y1','y2','y3','y4']
    # xlabel_str=r'\textbf{time} (s)'
    # ylabel_str='Number of passengers'
    # xticks_str=('G1', 'G2', 'G3', 'G4', 'G5','G6')
    # plotBars(xlabel_str,ylabel_str,4,[x,y1,y2,y3,y4],legends, xticks_str)

    # Example data
    data = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 5]
    plotPDF(data, r'$\alpha$', r'Empirical PDF')
