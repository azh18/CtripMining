# Observation, used for feature extraction
# Randomly select 10,000 passengers to perform all observations 
import pickle
def dumpSamplePassengersInfo():
	passengers = executeSQL('select passengerid from passenger_flight_counts where count>=20 ORDER BY RAND() limit 10000;')
	passengers = [p[0] for p in passengers]
	randomSelectedPassengerInfoDict = {}
	for passengerid in passengers:
		passengerInfo = getAllRecordsofAPassengerFromDB(passengerid)
		randomSelectedPassengerInfoDict[passengerid] = passengerInfo
	dumpFile = open('observe/choosedPassengersInfo_10000.json','w')
	pickle.dump(randomSelectedPassengerInfoDict,dumpFile)
	dumpFile.close()


###################################################################################
#Reservation factors
###################################################################################
import time
import datetime
import numpy as np
# import 

from commonoperation import getAllRecordsofAPassengerFromDB
from commonoperation import computeTimeDiff
from commonoperation import executeSQL


def extractReservationFactors():
	# passengers=executeSQL('select passengerid from passenger_flight_counts where count>=20 ORDER BY RAND() limit 10000;')
	# passengers=[p[0] for p in passengers]
	sampleData = pickle.load(open('observe/choosedPassengersInfo_10000.json'))
	passengers = sampleData.keys()
	alpha_samples=[]	#alpha=takeoff time - order time
	beta_samples=[] 	#beta=takeoff time - last takeoff time
	companion_sample = [] # number of companion for each passenger
	cnt = 0
	for passengerid in passengers:
		print('id:%d: %s'%(cnt,passengerid))
		#records=getAllRecordsofAPassengerFromDB(passengerid) #get records from DB, of this single user
		cnt += 1
		records = sampleData[passengerid]
		for i in range(0,len(records)):
			r=records[i]
			#advanced days
			ordertime=r[11]
			takeofftime=r[12]
			alpha_samples.append(computeTimeDiff(takeofftime,ordertime))
			#time gap between two trips
			if i>0:
				# last_trip_time=time.strptime(records[i-1][12],'%Y-%m-%d %H:%M:%S.%f')
				last_trip_time=records[i-1][12]
				beta_samples.append(computeTimeDiff(takeofftime,last_trip_time))
			# companion num searching
			companion_instance = executeSQL('select count from freq_orderid where orderid=\'%s\''%r[1])
			companion_sample.append(companion_instance[0][0])


	alpha_samples=list(filter(lambda x:x>=0,alpha_samples))
	with open('observe/alpha_samples.txt','w') as fh:
		fh.write('{0}\r\n'.format('\t'.join(str(e) for e in alpha_samples)))
	with open('observe/beta_samples.txt','w') as fh:
		fh.write('{0}\r\n'.format('\t'.join(str(e) for e in beta_samples)))
	with open('observe/companion_samples.txt','w') as fh:
		fh.write('{0}\r\n'.format('\t'.join(str(e) for e in companion_sample)))
	return (alpha_samples,beta_samples,companion_sample)

from myplotlib import plotPDF
from myplotlib import plotCDF
from myplotlib import plotFreq
def plotReservationFactors():
	# alpha_samples,beta_samples=extractReservationFactors()
	with open('observe/alpha_samples.txt','r') as fh:
		# fh.write('{0}\r\n'.format('\t'.join(str(e) for e in alpha_samples)))
		alpha_samples=fh.readline().replace('\r\n','').split('\t')
	with open('observe/beta_samples.txt','r') as fh:
		# fh.write('{0}\r\n'.format('\t'.join(str(e) for e in alpha_samples)))
		beta_samples=fh.readline().replace('\r\n','').split('\t')
	with open('observe/companion_samples.txt') as fh:
		companion_samples=fh.readline().replace('\r\n','').split('\t')

	alpha_samples=[float(e) for e in alpha_samples]
	beta_samples=[float(e) for e in beta_samples]
	companion_samples = [int(e) for e in companion_samples]

	# beta_samples=[float(e) for e in beta_samples]
	plotPDF(alpha_samples,r'$\alpha$ (days)',r'Empirical PDF','alpha_pdf.png',(0,32))
	plotCDF(alpha_samples,r'$\alpha$ (days)',r'Empirical CDF','alpha_cdf.png')
	plotPDF(beta_samples,r'$\beta$ (days)',r'Empirical PDF','beta_pdf.png',(0,101))
	plotCDF(beta_samples,r'$\beta$ (days)',r'Empirical CDF','beta_cdf.png')
	# plotPDF(companion_samples,r'$\sigma$',r'Empirical PDF','companion_pdf.png')
	plotFreq(companion_samples,r'$\sigma$',r'Number of passengers ($\times 1000$)','companion_pdf.png',xais_range=(2,15),
			 yaxis_range=(0,60),xtick_gap=1,ytick_gap=10,shrinkScale=1000)

###################################################################################
#Flight factors
# no need to extract the date and time, because no modification is needed
###################################################################################
import datapreparation
def extractFlightFactors():
	female_age_samples=[]
	male_age_samples=[]
	discount_samples=[]
	airline_samples = {}
	sampleData = pickle.load(open('observe/choosedPassengersInfo_10000.json'))
	airlineInfo = pickle.load(open('data/airlineIden.json'))
	print "load finished."
	for passengerid in sampleData.keys():
		print "process:%s" % passengerid
		data = sampleData[passengerid]
		age = int(data[0][-3])
		gender = data[0][-2]
		if gender=='F':
			female_age_samples.append(age)
		elif gender=='M':
			male_age_samples.append(age)
		for record in data:
			discount_samples.append(float(record[5]))
			airlineName = record[6]
			# airlineName = airlineInfo[airlineName] # transfer name into Iden
			if airline_samples.has_key(airlineName):
				airline_samples[airlineName] += 1
			else:
				airline_samples[airlineName] = 0
	pickle.dump(airline_samples,open('observe/airline_name_samples.json','w'))

	with open('observe/discount_samples.txt','w') as fh:
		fh.write('{0}\r\n'.format('\t'.join(str(e) for e in discount_samples)))
	with open('observe/female_age_samples.txt', 'w') as fh:
		fh.write('{0}\r\n'.format('\t'.join(str(e) for e in female_age_samples)))
	with open('observe/male_age_samples.txt', 'w') as fh:
		fh.write('{0}\r\n'.format('\t'.join(str(e) for e in male_age_samples)))
	# with open('observe/airline_name_samples.txt','w') as fh:
	# 	fh.write('{0}\r\n'.format('\t'.join([str(airline_samples[k]) for k in airline_samples])))
	return (discount_samples, female_age_samples, male_age_samples, airline_samples)


def plotFlightFactors():
	with open('observe/discount_samples.txt','r') as fh:
		discount_samples=fh.readline().replace('\r\n','').split('\t')
	discount_samples=[float(e) for e in discount_samples]
	# beta_samples=[float(e) for e in beta_samples]
	plotPDF(discount_samples,r'Price discount $\rho$',r'Empirical PDF','discount_pdf.png',(0,1))
	plotCDF(discount_samples,r'Price discount $\rho$',r'Empirical CDF','discount_cdf.png',(0,1))

	# draw airline freq
	airline_samples = pickle.load(open('observe/airline_name_samples.json'))
	# airline_samples = [int(e) for e in airline_samples]
	# airline_hist_bin = np.arange(-0.5,len(airline_samples)-0.5,1)
	# airline_hist, airline_hist_bin = np.histogram(airline_samples, airline_hist_bin, range=(-0.5,len(airline_samples)-0.5))
	airline_samples = sorted(airline_samples.iteritems(),key=lambda d:d[1], reverse=True) # sort from big to small
	airline_hist = []
	airline_bin = []
	displaycnt = 0
	display_num = 20
	for item in airline_samples:
		airline_hist.append(item[1])
		airline_bin.append(item[0])
		displaycnt += 1
		if displaycnt >= display_num:
			break
	airline_hist = np.array(airline_hist)

	xticks_str = []
	# airlineNames = pickle.load(open('data/airlineIden.json'))
	for nm in airline_bin:
		xticks_str.append(nm)
	plotBars('Airline', r'Number($\times$ 100)', 1, (None, airline_hist/100), xticks_str=xticks_str,
			 fig_name='airline_hist.png')

###################################################################################
#Passenger factors
###################################################################################
# from myplotlib import plotBars
import basicmining
def extractPassengerFactors():
	hometown_samples = []
	passengersData = pickle.load(open('observe/choosedPassengersInfo_10000.json'))
	for passengerid in passengersData.keys():
		hometown_samples.append(basicmining.getHomeCity(passengersData[passengerid]))
	hometown_histogram = {}
	for hometown in hometown_samples:
		if hometown_histogram.has_key(hometown):
			hometown_histogram[hometown] += 1
		else:
			hometown_histogram[hometown] = 1
	pickle.dump(hometown_samples, open('observe/hometown_samples.json','w'))
	pickle.dump(hometown_histogram, open('observe/hometown_histogram.json','w'))



def plotPassengerFactors():
	with open('observe/female_age_samples.txt','r') as fh:
		female_age_samples=fh.readline().replace('\r\n','').split('\t')
	female_age_samples=[float(e) for e in female_age_samples]
	# female_age_samples=female_age_samples[0:100]

	with open('observe/male_age_samples.txt','r') as fh:
		male_age_samples=fh.readline().replace('\r\n','').split('\t')
	male_age_samples=[float(e) for e in male_age_samples]
	# male_age_samples=male_age_samples[0:100]

	bins=[0,18,25,35,50,100]	#bins for ages
	female_age_hist,bins=np.histogram(female_age_samples,bins=bins)
	female_age_prop_hist=female_age_hist*1./sum(female_age_hist)
	male_age_hist,bins=np.histogram(male_age_samples,bins=bins)
	male_age_prop_hist=male_age_hist*1./sum(male_age_hist)

	xticks_str=[]
	for i in range(0,len(bins)-2):
		xticks_str.append('[{0},{1})'.format(bins[i],bins[i+1]))
	xticks_str.append('[{0},+)'.format(bins[-2]))
	plotBars('Age','Number',2,(None,female_age_hist,male_age_hist),('Female','Male'),xticks_str,fig_name='age_num_hist.png')
	plotBars('Age','Proportion',2,(None,female_age_prop_hist,male_age_prop_hist),('Female','Male'),xticks_str,fig_name='age_prop_hist.png',text_is_int=False)
	# hometown city
	hometown_samples = pickle.load(open('observe/hometown_samples.json'))
	hometown_samples = np.array(hometown_samples)
	hometown_histogram = pickle.load(open('observe/hometown_histogram.json'))
	hometown_histogram = sorted(hometown_histogram.iteritems(),key = lambda d:d[1],reverse=True)
	hometown_bin = []
	hometown_hist = []
	displaycnt = 0
	display_num = 20
	for item in hometown_histogram:
		hometown_bin.append(item[0])
		hometown_hist.append(item[1])
		displaycnt += 1
		if displaycnt >= display_num:
			break
	hometown_hist = np.array(hometown_hist)

	# hometown_hist, hometown_bins = np.histogram(hometown_samples,np.sort(np.unique(hometown_samples)))
	# print hometown_hist
	# print hometown_bins
	xticks_str=[]
	plotBars('City', 'Number', 1, (None, hometown_hist), xticks_str = hometown_bin, fig_name='city_hist.png')


###################################################################################
# Relation between factors
###################################################################################

# age, gender, hometown, income v.s. pricekpi, seat, beta
###################################################################################



def findRelationAndPlot(samples, figname='noName.png',):
	hometown_samples = []
	passengersData = pickle.load(open('observe/choosedPassengersInfo_10000.json'))
	for passengerid in passengersData.keys():
		hometown_samples.append(basicmining.getHomeCity(passengersData[passengerid]))
	hometown_histogram = {}
	for hometown in hometown_samples:
		if hometown_histogram.has_key(hometown):
			hometown_histogram[hometown] += 1
		else:
			hometown_histogram[hometown] = 1
	pickle.dump(hometown_samples, open('observe/hometown_samples.json', 'w'))
	pickle.dump(hometown_histogram, open('observe/hometown_histogram.json', 'w'))

from os.path import join
import matplotlib as mpl
mpl.use('agg')
from matplotlib import rc
import matplotlib.pylab as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
rc('font', **{'family': 'serif', 'serif': ['Times New Roman'],'size':18})
rc('text', usetex=True)
rc('lines', **{'linewidth':2.0,'marker':'o','markersize':0})

WORK_PATH='plots'
BAR_FACE_COLORS=['white','darkgray','maroon','navy','darkolivegreen','dimgray']
PATTERNS=['-','x','/']
# BAR_FACE_COLORS=['r','b','y','g']

def drawRelationScatter(xlabel_str,ylabel_str, data,xais_range=None,yaxis_range=None,
			 xtick_gap=None, ytick_gap=None, shrinkScale=None,N=1000, yaxis_log=False, figname='noname.png'):
	fig, axScatter = plt.subplots(figsize=(8, 8))
	# the scatter plot:
	x = data[0]
	y = data[1]
	axScatter.scatter(x, y, color='cornflowerblue')
	axScatter.set_aspect(1.)
	# plt.xlabel(r'Seat preference')
	# plt.ylabel(r'Travel time preference')
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

	# plt.show()
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
		y_tick = np.arange(yaxis_range[0], yaxis_range[1] + xtick_gap, ytick_gap)
		plt.yticks(y_tick)
	if (figname != None):
		plt.savefig(join(WORK_PATH, figname))
	# plt.savefig(fig_name)
	plt.close(fig)

def plotBars(xlabel_str,ylabel_str,nlines,data,legends=None,xticks_str=None,axis_range=None,
			 fig_name='barplot.png', add_text=False, text_is_int=True):
	x=data[0]
	N=len(data[1])
	ind = np.arange(0,3*N,3)+0.5  # the x locations for the groups
	print(ind)
	bar_width = 3/nlines*0.7 	# the width of the bars
	print(data[1])
	# print(data[2])
	print(nlines)

	fig=plt.figure(figsize=(8,6))
	# fig=plt.figure()
	ax = plt.subplot(111)
	rects=[]
	for k in range(0,nlines):
		if legends != None:
			rect=ax.bar(ind+(k-(nlines-1.)/2)*bar_width, data[k+1],width=bar_width,color=BAR_FACE_COLORS[k],hatch=PATTERNS[k], align='center',label=legends[k])
		else:
			# rect = ax.bar(ind + (k - (nlines - 1.) / 2) * bar_width, data[k + 1], width=bar_width,
			# 			  color=BAR_FACE_COLORS[k+1], hatch=PATTERNS[k], align='center')
			rect = ax.bar(ind + (k - (nlines - 1.) / 2) * bar_width, data[k + 1], width=bar_width,
						  color=BAR_FACE_COLORS[k+2], align='center')
		rects.append(rect)

	ax.set_xlim(xmin=-1)
	ax.set_xticks(ind)
	if xticks_str is not None:
		ax.set_xticklabels(xticks_str,fontsize=12)
	elif x is not None:
		ax.set_xticklabels([str(e) for e in x])
	if legends is not None:
		ax.legend(rects,legends,loc='upper left')

	def autolabel(rects): # attach some text labels
		for rect in rects:
			height = rect.get_height()
			if text_is_int is True:
				t='%d'%int(height)
			else: # text is a float number
				t='%s'%str(round(height,3))
			ax.text(rect.get_x()+rect.get_width()/2., 1.02*height, t, ha='center', va='bottom',fontsize=14) 
	ax.set_xlabel(xlabel_str)
	ax.set_ylabel(ylabel_str)
	if add_text is True:
		for rect in rects:
			autolabel(rect)
	plt.savefig(join(WORK_PATH,fig_name))
	plt.close(fig)

###################################################################################
#Other observations
###################################################################################
import operator
def plotVisitingNumOfCity():
	#read selected cities
	cityNameDict={}
	cityFile='observe/selected_cities.csv'
	with open(cityFile,'rb') as fh:
		lines=fh.readlines();
	for r in lines:
		fields=r.strip().split(',')
		i=int(fields[0])
		cityNameDict[i]=fields[4]
	cityNameDict=dict(sorted(cityNameDict.items(),key=operator.itemgetter(0)))

	#read data
	dataFile='observe/TravelersOfEachCity.csv'
	departureDict={}
	arrivalDict={}
	with open(dataFile,'rb') as fh:
		fh.readline() 
		lines=fh.readlines()
	for r in lines:
		fields=r.strip().split(',')
		city=int(fields[0])
		if city in cityNameDict:
			departureDict[city]=float(fields[1])
			arrivalDict[city]=float(fields[2])
	departureDict=dict(sorted(departureDict.items(),key=operator.itemgetter(0)))
	arrivalDict=dict(sorted(arrivalDict.items(),key=operator.itemgetter(0)))
	departureMeans=departureDict.values()
	arrivalMeans=arrivalDict.values()

	departureMeans= np.array(departureMeans)*1./sum(departureMeans)
	arrivalMeans= np.array(arrivalMeans)*1./sum(arrivalMeans)
	xticks_str=cityNameDict.values()
	plotBars('city','Number of passengers',2,(None,departureMeans,arrivalMeans),('Departure','Arrival'),xticks_str,fig_name='city_num_hist.png',text_is_int=False)


if __name__ == '__main__':
	# dumpSamplePassengersInfo() # have finished
	# extractFlightFactors()

	# plotFlightFactors()
	# extractPassengerFactors()
	# plotPassengerFactors()
	# extractReservationFactors()
	plotReservationFactors()
	# extractFactors()
	# plotFlightFactors()
	# plotPassengerFactors()
	# plotVisitingNumOfCity()