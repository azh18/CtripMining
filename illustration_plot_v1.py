import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap
from os.path import join

workpath = 'plots'


def plotCityWithHotDensity(posList, densList, figureName='city'):
    fig = plt.figure()  # create new figure
    # setup Lambert Conformal basemap.
    m = Basemap(width=5200000, height=4000000, projection='lcc',
                resolution='h', lat_1=10, lat_2=48, lat_0=37, lon_0=105)
    # read shapefile
    m.readshapefile(shapefile=join(workpath, 'CHN_adm/CHN_adm1'), name='counties', color='k')

    # lons=[116.46]
    # lats=[39.92]
    # x,y=m(lons,lats)
    # m.plot(x,y,'bo',markersize=5)
    # plt.show()

    markersizeDict = [2, 4, 6, 8, 10]
    # colorDict=['bo','bo','bo','bo']
    # colorDict=[(1,1,1),(0,0,1),(0,1,1),(1,1,1)]
    colorDict = ['blue', 'cyan', 'green', 'yellow', 'red']
    for i in range(0, len(posList)):
        p = posList[i]
        dens = densList[i]
        level = discretizeDensity(dens)

        lons = [p[0]]
        lats = [p[1]]
        x, y = m(lons, lats)
        m.plot(x, y, markerfacecolor=colorDict[level], marker='o', markersize=markersizeDict[level])
    # plt.savefig('{0}.png'.format(figureName))
    fig.savefig(join(workpath, '{0}.png'.format(figureName)))


def discretizeDensity(dens):
    # num of dens: 5
    level = 0
    if dens < 10:
        level = 0
    elif dens < 30:
        level = 1
    elif dens < 50:
        level = 2
    elif dens < 80:
        level = 3
    else:
        level = 4
    return level


def generateColorWithDensity(densList):
    minVal, maxVal = 1, 200
    colorList = []
    for d in densList:
        if d > maxVal:
            d = maxVal
        val = 1.0 * (d - minVal) / (maxVal - minVal)
        colorList.append((val, val, val))
    return colorList


# FONT_SIZE=18
# LINE_WIDTH=2
from myplotlib import plotLines


def plotSparsity():
    import usergeneration
    numsDict = usergeneration.loadNums()
    numsList = []
    X = np.arange(30, 151, 1)
    for i in X:
        if i in numsDict:
            # numsList.append(len(numsDict[i]))
            numsList.append(numsDict[i])
        else:
            numsList.append(0)
    # fig=plt.figure()
    # plt.plot(range(30,501),numsList, color='blue', linewidth=LINE_WIDTH,linestyle='-',label='')
    # plt.legend(fontsize=FONT_SIZE)
    # plt.axis([29,151,0,9000])
    # plt.xlabel('Number of records',fontsize=FONT_SIZE)
    # plt.xticks(fontsize=FONT_SIZE)
    # plt.ylabel('Number of passengers',fontsize=FONT_SIZE)
    # plt.yticks(fontsize=FONT_SIZE)
    # plt.savefig(join(workpath,'sparsity.png'))
    plotLines('Number of flights', 'Number of passengers', 1, [(X, numsList)], [''], None, 'sparsity.png')


def run():
    # load position(longitude,latitude)
    cityPosDict = {}
    with open(join(workpath, 'city_details_all.csv'), 'rb') as fh:
        lines = fh.readlines()
        for row in lines:
            [cid, lon, lat] = row.strip().split(',')
            if len(lon) > 0 and len(lat) > 0:
                cityPosDict[int(cid)] = (float(lon), float(lat))

    # load density(travel times)
    userInfoDict = {}
    with open(join(workpath, 'city_travel_times.txt'), 'rb') as fh:
        lines = fh.readlines()
        for row in lines:
            [uid, dens] = row.strip().split(':')
            data = dens.replace('[', '').replace(']', '').replace('(', '').replace(')', '').split(',')
            posList = []
            densList = []
            for i in range(0, len(data) / 2):
                cid = int(data[2 * i])
                dens = int(data[2 * i + 1])
                if cid in cityPosDict:
                    posList.append(cityPosDict[cid])
                    densList.append(dens)
                    userInfoDict[uid] = (posList, densList)

    # plot
    for u in userInfoDict.keys():
        v = userInfoDict[u]
        plotCityWithHotDensity(v[0], v[1], u)


if __name__ == '__main__':
    plotSparsity()
