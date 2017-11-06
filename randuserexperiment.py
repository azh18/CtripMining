#randomly select 3 passengers and retrieve their records. To illustrate passengers are different.

# Selected passengers 
# passengersid,	num of flights in two years (2013-2014)
# MjIwMjg0MTk4NDA1MjcyNjE3	20
# MTEwMTA1MTk2NTA0MDM1MDNY	40
# NTEzMjI5MTk2ODAxMjUwMDEy	60

import time
import datetime
import numpy as np

from commonoperation import getAllRecordsofAPassengerFromDB
from commonoperation import computeTimeDiff
def extractStat(passengerid):
	records=getAllRecordsofAPassengerFromDB(passengerid) #get records from DB
	
	freq=len(records)*1./2  # 2 years
	avg_advanced_days=0
	avg_discount=0
	avg_travel_day_gaps=0

	for i in range(0,len(records)):
		r=records[i]
		#price discount
		avg_discount+=float(r[5])
		#advanced days
		# ordertime=time.strptime(r[11],'%Y-%m-%d %H:%M:%S.%f')
		# takeofftime=time.strptime(r[12],'%Y-%m-%d %H:%M:%S.%f')
		ordertime=r[11]
		takeofftime=r[12]
		avg_advanced_days+=computeTimeDiff(takeofftime,ordertime)
		#time gap between two trips
		if i>0:
			# last_trip_time=time.strptime(records[i-1][12],'%Y-%m-%d %H:%M:%S.%f')
			last_trip_time=records[i-1][12]
			avg_travel_day_gaps+=computeTimeDiff(takeofftime,last_trip_time)

	avg_discount/=1.*len(records)
	avg_advanced_days/=1.*len(records)
	avg_travel_day_gaps/=1.*len(records)-1
	return (freq,avg_advanced_days,avg_discount,avg_travel_day_gaps)

def comparePassengers():
	'''
	Compare 3 passengers, statistical information
	'''
	passengers=['MjIwMjg0MTk4NDA1MjcyNjE3','MTEwMTA1MTk2NTA0MDM1MDNY','NTEzMjI5MTk2ODAxMjUwMDEy']
	for p in passengers:
		print(p,extractStat(p))
# #output:
# ('MjIwMjg0MTk4NDA1MjcyNjE3', (10.0, 3.425070601851852, 0.844, 15.785818713450295))
# ('MTEwMTA1MTk2NTA0MDM1MDNY', (20.0, 4.5274916087962955, 0.8212499999999998, 8.697204415954417))
# ('NTEzMjI5MTk2ODAxMjUwMDEy', (30.0, 1.852125385802469, 0.7621666666666668, 11.691031073446329))

from commonoperation import getDayType
def extractDetailedInfo(passengerid):
	records=getAllRecordsofAPassengerFromDB(passengerid) #get records from DB
	hours_proportion=np.zeros(24)
	daytype_proportation=np.zeros(3)
	for r in records:
		h=r[12].hour
		hours_proportion[h]+=1
		d=getDayType(r[12])
		daytype_proportation[d-1]+=1

	hours_proportion/=sum(hours_proportion)
	daytype_proportation/=sum(daytype_proportation)
	hours_proportion=hours_proportion.tolist()
	daytype_proportation=daytype_proportation.tolist()
	return hours_proportion[6:],daytype_proportation

# from commonplot import plotDetailedInfo
def comparePassengers_v1():
	'''
	Compare 3 passengers, parameters when time changes
	'''
	passengers=['MjIwMjg0MTk4NDA1MjcyNjE3','MTEwMTA1MTk2NTA0MDM1MDNY','NTEzMjI5MTk2ODAxMjUwMDEy']
	data1=[None]
	data2=[None]
	for p in passengers:
		d1,d2=extractDetailedInfo(p)
		# print(d1,d2)
		data1.append(d1)
		data2.append(d2)
	plotDetailedInfo(data1,data2)
# # output
# ('MjIwMjg0MTk4NDA1MjcyNjE3', ([0.0, 0.0, 0.05, 0.0, 0.0, 0.05, 0.0, 0.0, 0.15, 0.2, 0.1, 0.15, 0.05, 0.05, 0.05, 0.15, 0.0, 0.0], [0.7, 0.25, 0.05]))
# ('MTEwMTA1MTk2NTA0MDM1MDNY', ([0.0, 0.025, 0.05, 0.05, 0.05, 0.175, 0.05, 0.125, 0.1, 0.025, 0.075, 0.025, 0.025, 0.125, 0.075, 0.025, 0.0, 0.0], [0.825, 0.175, 0.0]))
# ('NTEzMjI5MTk2ODAxMjUwMDEy', ([0.0, 0.1, 0.06666666666666667, 0.11666666666666667, 0.0, 0.0, 0.06666666666666667, 0.05, 0.05, 0.16666666666666666, 0.16666666666666666, 0.08333333333333333, 0.03333333333333333, 0.016666666666666666, 0.05, 0.03333333333333333, 0.0, 0.0], [0.8, 0.18333333333333332, 0.016666666666666666]))

def plotDetailedInfo(data1,data2):
	legends=['Passenger A','Passenger B','Passenger C']
	xlabel_str='Takeoff hours'
	ylabel_str='Proporation of flights at different hours'
	xticks_str=[str(e) for e in range(6,24)]
	plotBars(xlabel_str,ylabel_str,3,data1,legends, xticks_str,None,'Proporation_hours.png')

	legends=['Passenger A','Passenger B','Passenger C']
	xlabel_str='Takeoff days'
	ylabel_str='Proporation of flights in different days'
	xticks_str=['Workday','Weekend','Holiday']
	plotBars(xlabel_str,ylabel_str,3,data2,legends, xticks_str,None,'Proporation_days.png')


from os.path import join
import matplotlib as mpl
mpl.use('agg')
from matplotlib import rc
import matplotlib.pylab as plt
rc('font', **{'family': 'serif', 'serif': ['Times New Roman'],'size':18})
rc('text', usetex=True)
rc('lines', **{'linewidth':2.0,'marker':'o','markersize':0})
WORK_PATH='plots'
BAR_FACE_COLORS=['darkred','navy','darkolivegreen','dimgray', 'cornflowerblue','teal']
GRAYS=['white','darkgray','cornflowerblue','lightgray']
PATTERNS=['/','\\','x','-',]
def plotBars(xlabel_str,ylabel_str,nlines,data,legends,xticks_str=None,axis_range=None,fig_name='barplot.png', add_text=True, text_is_int=True):
	x=data[0]
	N=len(data[1])
	ind = np.arange(0,3*N,3)+0.5  # the x locations for the groups
	print(ind)
	bar_width = 3/nlines*0.8 	# the width of the bars
	print(data[1])
	print(data[2])
	print(nlines)

	# fig=plt.figure(figsize=(26,13))
	fig=plt.figure()
	ax = plt.subplot(111)
	rects=[]
	for k in range(0,nlines):
		rect=ax.bar(ind+(k-(nlines-1.)/2)*bar_width, data[k+1],width=bar_width,color='white', hatch=PATTERNS[k],align='center',label=legends[k])
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
		ax.legend(rects,legends,loc=1)
	plt.savefig(join(WORK_PATH,fig_name))
	plt.close(fig)


from commonoperation import executeSQL
def comparePassengersStat():
	passengers=executeSQL('select passengerid from passenger_flight_counts where count>=20 ORDER BY RAND() limit 10000;')
	info_list=[extractStat(p[0]) for p in passengers]
	
	info_array=np.array(info_list)
	n,cols=info_array.shape
	results=[]
	for k in range(0,cols):
		avg=sum(info_array[:,k])*1./n
		std=np.std(info_array[:,k])
		results.append((avg,std))
	print(results)
	return results
# for k in range(0,10):
# 	comparePassengersStat()
#output of running 10 times 
# [(14.2052, 5.4540803954471313), (4.3135717022859073, 3.6821673338374579), (0.66693963219213226, 0.1303870029111118), (21.646007926136075, 6.6320047938176963)]
# [(14.21665, 5.2049291808342524), (4.3039383047196624, 3.6957587240735417), (0.66376502739655152, 0.1298413758457368), (21.614116075150097, 6.6340621497188801)]
# [(14.20255, 5.5001998597778163), (4.3371600922211071, 3.6824687703761692), (0.66645243854854208, 0.13178678938923993), (21.678728959960207, 6.6794808084389361)]
# [(14.2234, 5.1911841076963672), (4.2487410003881978, 3.6534730956757833), (0.66716579146342803, 0.12895793282402029), (21.505520713949757, 6.6351959345220308)]
# [(14.2826, 5.7372325419142456), (4.2446851529648439, 3.5886770202875451), (0.66842652022266869, 0.12920199756775266), (21.625892585542672, 6.7691916234818503)]
# [(14.156650000000001, 4.6760545096802062), (4.2882461003384948, 3.6508878042285131), (0.66870040653867546, 0.12958296424599286), (21.684513466951618, 6.6624651116620832)]
# [(14.17245, 5.3278124964660574), (4.2960271624082784, 3.6242177442813697), (0.66861000290976391, 0.12902653777820505), (21.755434008381275, 6.6863471183023027)]
# [(14.270949999999999, 5.3630785093544491), (4.2219424751658892, 3.5525296678954117), (0.66632601480552278, 0.12885600254207813), (21.528738578097006, 6.6533919782109239)]
# [(14.273, 5.8657370380882519), (42.703603488860091, 3.599369514218949), (0.6679634181944516, 0.12978143505871062), (21.716922677801996, 6.7436511301543192)]
# [(14.1845, 5.08091623135038), (4.2691445958734269, 3.6482678243155271), (0.66918817953438703, 0.12931357724936407), (21.669184758067235, 6.6650099135121765)]
# avg of all passengers: [14.218795, 5.3401224870609161, 8.1227060075225896, 3.6377817499190273, 0.66735374318061225, 0.1296735615412212, 21.642505975003797, 6.6760800561821201]

from basicmining import generateProfles
from basicmining import generateUnnormlaizedProfles
from basicmining import compareProfiles
def compareProfileVectors():
# MjIwMjg0MTk4NDA1MjcyNjE3	
# MTEwMTA1MTk2NTA0MDM1MDNY	
# NTEzMjI5MTk2ODAxMjUwMDEy
	passengers=['MjIwMjg0MTk4NDA1MjcyNjE3', 'MTEwMTA1MTk2NTA0MDM1MDNY', 'NTEzMjI5MTk2ODAxMjUwMDEy']
	profils_u=generateUnnormlaizedProfles(passengers)
	profils_n=generateProfles(passengers)

	print('print profiles')
	for p in passengers:
		print(p,profils_u[p],profils_n[p])

	print('print distances')
	for i in range(0,len(passengers)):
		for j in range(i+1,len(passengers)):
			A=passengers[i]
			B=passengers[j]
			print(A,B,compareProfiles(profils_n[A],profils_n[B]))


if __name__ == '__main__':
	comparePassengers_v1()
	# compareProfileVectors()




