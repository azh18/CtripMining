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
	for passengerid in passengers:
		print(passengerid)
		#records=getAllRecordsofAPassengerFromDB(passengerid) #get records from DB, of this single user
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
	plotPDF(alpha_samples,r'$\alpha$ (days)',r'Empirical PDF','observe/alpha_pdf.png',(0,32))
	plotCDF(alpha_samples,r'$\alpha$ (days)',r'Empirical CDF','observe/alpha_cdf.png')
	plotPDF(beta_samples,r'$\beta$ (days)',r'Empirical PDF','observe/beta_pdf.png',(0,101))
	plotCDF(beta_samples,r'$\beta$ (days)',r'Empirical CDF','observe/beta_cdf.png')
	plotPDF(companion_samples,r'$\sigma$',r'Empirical PDF','observe/companion_pdf.png')

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
	for passengerid, data in sampleData:
		age = int(data[0][-3])
		gender = data[0][-2]
		if gender=='F':
			female_age_samples.append(age)
		elif gender=='M':
			male_age_samples.append(age)
		for record in data:
			discount_samples.append(float(record[6]))
			airlineName = record[7]
			airlineName = airlineInfo[airlineName] # transfer name into Iden
			if airline_samples.has_key(airlineName):
				airline_samples[airlineName] += 1
			else:
				airline_samples[airlineName] = 0

	with open('observe/discount_samples.txt','w') as fh:
		fh.write('{0}\r\n'.format('\t'.join(str(e) for e in discount_samples)))
	with open('observe/female_age_samples.txt', 'w') as fh:
		fh.write('{0}\r\n'.format('\t'.join(str(e) for e in female_age_samples)))
	with open('observe/male_age_samples.txt', 'w') as fh:
		fh.write('{0}\r\n'.format('\t'.join(str(e) for e in male_age_samples)))
	with open('observe/airline_name_samples.txt','w') as fh:
		fh.write('{0}\r\n'.format('\t'.join(str(e) for k,e in airline_samples)))
	return (discount_samples, female_age_samples, male_age_samples, airline_samples)


def plotFlightFactors():
	with open('observe/discount_samples.txt','r') as fh:
		discount_samples=fh.readline().replace('\r\n','').split('\t')
	discount_samples=[float(e) for e in discount_samples]
	# beta_samples=[float(e) for e in beta_samples]
	plotPDF(discount_samples,r'Price discount $\rho$',r'Empirical PDF','observe/discount_pdf.png',(0,1))
	plotCDF(discount_samples,r'Price discount $\rho$',r'Empirical CDF','observe/discount_cdf.png',(0,1))

	# draw airline freq
	with open('observe/airline_name_samples.txt','r') as fh:
		airline_samples = fh.readline().replace('\r\n', '').split('\t')
	airline_samples = [int(e) for e in airline_samples]
	airline_hist_bin = np.arange(-0.5,len(airline_samples)-0.5,1)
	# airline_hist, airline_hist_bin = np.histogram(airline_samples, airline_hist_bin, range=(-0.5,len(airline_samples)-0.5))
	airline_hist = (np.array(airline_samples), airline_hist_bin)

	xticks_str = []
	airlineNames = pickle.load(open('data/airlineIden.json'))
	for key in airlineNames.keys():
		xticks_str.append(key)
	plotBars('Airline', 'Number', 1, (None, airline_hist), xticks_str=xticks_str,
			 fig_name='observe/airline_hist.png')

###################################################################################
#Passenger factors
###################################################################################
# from myplotlib import plotBars
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
	plotBars('Age','Number',2,(None,female_age_hist,male_age_hist),('Female','Male'),xticks_str,fig_name='observe/age_num_hist.png')
	plotBars('Age','Proportion',2,(None,female_age_prop_hist,male_age_prop_hist),('Female','Male'),xticks_str,fig_name='observe/age_prop_hist.png',text_is_int=False)

from os.path import join
import matplotlib as mpl
mpl.use('agg')
from matplotlib import rc
import matplotlib.pylab as plt
rc('font', **{'family': 'serif', 'serif': ['Times New Roman'],'size':18})
rc('text', usetex=True)
rc('lines', **{'linewidth':2.0,'marker':'o','markersize':0})

WORK_PATH='plots'
BAR_FACE_COLORS=['white','darkgray','maroon','navy','darkolivegreen','dimgray']
PATTERNS=['-','x','/']
# BAR_FACE_COLORS=['r','b','y','g']
def plotBars(xlabel_str,ylabel_str,nlines,data,legends=None,xticks_str=None,axis_range=None,fig_name='observe/barplot.png', add_text=True, text_is_int=True):
	x=data[0]
	N=len(data[1])
	ind = np.arange(0,3*N,3)+0.5  # the x locations for the groups
	print(ind)
	bar_width = 3/nlines 	# the width of the bars
	print(data[1])
	print(data[2])
	print(nlines)

	# fig=plt.figure(figsize=(26,13))
	fig=plt.figure()
	ax = plt.subplot(111)
	rects=[]
	for k in range(0,nlines):
		if legends != None:
			rect=ax.bar(ind+(k-(nlines-1.)/2)*bar_width, data[k+1],width=bar_width,color=BAR_FACE_COLORS[k], hatch=PATTERNS[k], align='center',label=legends[k])
		else:
			rect = ax.bar(ind + (k - (nlines - 1.) / 2) * bar_width, data[k + 1], width=bar_width,
						  color=BAR_FACE_COLORS[k], hatch=PATTERNS[k], align='center', label=0)
		rects.append(rect)
	ax.set_xlabel(xlabel_str)
	ax.set_ylabel(ylabel_str)
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
	plotBars('city','Number of passengers',2,(None,departureMeans,arrivalMeans),('Departure','Arrival'),xticks_str,fig_name='observe/city_num_hist.png',text_is_int=False)


if __name__ == '__main__':
	dumpSamplePassengersInfo()
	# extractReservationFactors()
	# plotReservationFactors()
	# extractFactors()
	# plotFlightFactors()
	# plotPassengerFactors()
	# plotVisitingNumOfCity()