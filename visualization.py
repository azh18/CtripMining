import os
import random
import comparemethods
import commonoperation
import matplotlib as mpl
mpl.use('agg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

def visualizeTrainedModel():
	'''
	3d bar plot
	'''
	modelName='mix-fKDE2_cv'
	modelPath='models/{0}'.format(modelName)
	# 3000939044_mix-fKDE2_1.5.model
	# mFile='models/{0}/{1}_{2}.model'.format(modelName, uId,modelName)
	trainedModels=random.sample(os.listdir(modelPath),10)
	for m in trainedModels:
		mFile= '{0}/{1}'.format(modelPath,m)
		print(mFile)
		pId=mFile.split('/')[-1].replace('.model','').replace('_{0}'.format(modelName),'')
		print(pId)
		m=comparemethods.mixfKDEModel(pId,modelFile=mFile)
		m.loadModel()
		kde=m.trainedModel

		# xs: advanced day
		# ys: price kpi 
		# fixed: seat class, daytype
		daytype=commonoperation.DayType.WORKDAY
		hourtype= 15/2
		seat=1
		discount=0.8

		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		for c, z in zip(['r', 'g', 'b','y'], [0.25,0.5,0.75,1]):
			data=[]
			xs=range(0,20,2)
			for x in xs:
				data.append([x,daytype, hourtype,seat,z,discount])
			ys = kde.computeProbs(data)
			# You can provide either a single color or an array. To demonstrate this,
			# the first bar of each set will be colored cyan.
			cs = [c] * len(xs)
			# cs[0] = 'c'
			ax.bar(xs, ys, zs=z, zdir='y', color=cs, alpha=0.8)
			# print(xs)
			# print(ys)
			# print(z
			# break
		ax.set_xlabel(r'#advanced days')
		ax.set_ylabel(r'price kpi')
		ax.set_zlabel(r'density')
		plt.savefig('plots/{0}_3d.pdf'.format(pId))
		plt.close(fig)
	return 


from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
def visualizeTrainedModel_v1(): 
	'''
	3d Surface plot
	'''
	modelName='mix-fKDE2_cv'
	modelPath='models/{0}'.format(modelName)
	trainedModels=random.sample(os.listdir(modelPath),10)
	for m in trainedModels:
		mFile= '{0}/{1}'.format(modelPath,m)
		print(mFile)
		pId=mFile.split('/')[-1].replace('.model','').replace('_{0}'.format(modelName),'')
		print(pId)
		m=comparemethods.mixfKDEModel(pId,modelFile=mFile)
		m.loadModel()
		kde=m.trainedModel

		# xs: advanced day
		# ys: price kpi 
		# fixed: seat class, daytype
		daytype=commonoperation.DayType.WORKDAY
		hourtype= 15/2
		seat=1
		discount=0.8

		fig = plt.figure()
		ax = fig.gca(projection='3d')
		X=np.linspace(0,30,61) #num of advanced days
		Y=np.linspace(0,1,51) #price kpi
		X,Y=np.meshgrid(X,Y)
		old_shape=X.shape
		N=X.shape[0]*X.shape[1]
		X1=np.reshape(X,(N,))
		Y1=np.reshape(Y,(N,))
		data=[]
		for k in range(0,N):
			data.append([X1[k],daytype, hourtype,seat,Y1[k],discount])
		print(len(data))
		Z1=np.array(kde.computeProbs(data)) 
		Z=np.reshape(Z1,old_shape)
		print(X.shape,Y.shape,Z.shape)

		surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
		ax.zaxis.set_major_locator(LinearLocator(10))
		ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))	
		fig.colorbar(surf, shrink=0.5, aspect=5)
		ax.set_xlabel(r'#advanced days')
		ax.set_ylabel(r'price kpi')
		ax.set_zlabel(r'density')

		plt.savefig('plots/{0}_3d.pdf'.format(pId))
		plt.close(fig)
	return 

def visualizeTrainedModel_v2(): 
	'''
	3d Surface plot
	'''
	modelName='mix-fKDE2_cv'
	modelPath='models/{0}'.format(modelName)
	trainedModels=random.sample(os.listdir(modelPath),10)
	for m in trainedModels:
		mFile= '{0}/{1}'.format(modelPath,m)
		pId=mFile.split('/')[-1].replace('.model','').replace('_{0}'.format(modelName),'')
		print(pId)
		m=comparemethods.mixfKDEModel(pId,modelFile=mFile)
		m.loadModel()
		kde=m.trainedModel

		# xs: advanced day
		# ys: price kpi 
		# fixed: seat class, daytype
		daytype=commonoperation.DayType.WORKDAY
		hourtype= 15/2
		seat=1
		# discount=0.8
		price_kpi=0.8

		fig = plt.figure()
		ax = fig.gca(projection='3d')
		X=np.linspace(0,30,61) #num of advanced days
		Y=np.linspace(0,1,51) #price discount
		X,Y=np.meshgrid(X,Y)
		old_shape=X.shape
		N=X.shape[0]*X.shape[1]
		X1=np.reshape(X,(N,))
		Y1=np.reshape(Y,(N,))
		data=[]
		for k in range(0,N):
			data.append([X1[k],daytype, hourtype,seat,price_kpi,Y1[k]])
		Z1=np.array(kde.computeProbs(data)) 
		Z=np.reshape(Z1,old_shape)

		surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
		ax.zaxis.set_major_locator(LinearLocator(10))
		ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))	
		fig.colorbar(surf, shrink=0.5, aspect=5)
		ax.set_xlabel(r'#advanced days')
		ax.set_ylabel(r'price discount')
		ax.set_zlabel(r'density')

		plt.savefig('plots/{0}_3d.pdf'.format(pId))
		plt.close(fig)
	return 

def visualizeTrainedModel_v3(): 
	'''
	3d Surface plot
	'''
	modelName='mix-fKDE2_cv'
	modelPath='models/{0}'.format(modelName)
	trainedModels=random.sample(os.listdir(modelPath),10)
	for m in trainedModels:
		mFile= '{0}/{1}'.format(modelPath,m)
		pId=mFile.split('/')[-1].replace('.model','').replace('_{0}'.format(modelName),'')
		print(pId)
		m=comparemethods.mixfKDEModel(pId,modelFile=mFile)
		m.loadModel()
		kde=m.trainedModel

		# xs: advanced day
		# ys: price kpi 
		# fixed: seat class, daytype
		daytype=commonoperation.DayType.WORKDAY
		# hourtype= 15
		seat=1
		discount=0.8
		price_kpi=0.8

		fig = plt.figure()
		ax = fig.gca(projection='3d')
		X=np.linspace(0,30,61) #num of advanced days
		Y=np.arange(6,24,1) #hour type
		X,Y=np.meshgrid(X,Y)
		old_shape=X.shape
		N=X.shape[0]*X.shape[1]
		X1=np.reshape(X,(N,))
		Y1=np.reshape(Y,(N,))
		data=[]
		for k in range(0,N):
			# data.append([X1[k],daytype, Y1[k],seat,price_kpi,discount])
			data.append([X1[k],daytype, Y1[k],seat,discount])
		Z1=np.array(kde.computeProbs(data)) 
		Z=np.reshape(Z1,old_shape)

		surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
		ax.zaxis.set_major_locator(LinearLocator(10))
		ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))	
		fig.colorbar(surf, shrink=0.5, aspect=5)
		ax.set_xlabel(r'#advanced days')
		ax.set_ylabel(r'hour type')
		ax.set_zlabel(r'density')

		plt.savefig('plots/{0}_3d.pdf'.format(pId))
		plt.close(fig)
	return 


labelFontSize=24
legendFontSize=24
tickFontSize=18
position=[0.2,0.2,0.7,0.7]
figureSize=(80,80)

#plot figures of features
#plot the number of travelers(departure/arrival) in each city
import numpy as np
import matplotlib.pyplot as plt
import operator
def plotVisitingNumOfCity():
	#read selected cities
	cityNameDict={}
	cityFile='temp/selected_cities.csv'
	with open(cityFile,'rb') as fh:
		lines=fh.readlines();
	for r in lines:
		fields=r.strip().split(',')
		i=int(fields[0])
		cityNameDict[i]=fields[4]
	cityNameDict=dict(sorted(cityNameDict.items(),key=operator.itemgetter(0)))

	#read data
	dataFile='temp/TravelersOfEachCity.csv'
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
	
	#plot barchar
	N=len(departureDict)
	departureMeans=departureDict.values()
	arrivalMeans=arrivalDict.values()

	ind = np.arange(N) # the x locations for the groups
	width = 0.45 # the width of the bars
	fig, ax = plt.subplots() 
	ax.set_position([0.1,0.1,0.85,0.85])
	rects1 = ax.bar(ind, departureMeans, width, color='r')
	rects2 = ax.bar(ind+width, arrivalMeans, width, color='y')

	ax.set_ylabel('Number of travelers',fontsize=labelFontSize) 
	ax.set_xlabel('City',fontsize=labelFontSize) 
	ax.set_xticks(ind+width) 
	ax.set_xticklabels(cityNameDict.values())
	ax.legend( (rects1[0], rects2[0]), ('Departure', 'Arrival') ,fontsize=legendFontSize)

	def autolabel(rects): # attach some text labels
		for rect in rects:
			height = rect.get_height()
			ax.text(rect.get_x()+rect.get_width()/2., 1.05*height, '%d'%int(height), ha='center', va='bottom')

	autolabel(rects1) 
	autolabel(rects2) 
	# plt.savefig('plots/visiting_num_of_city.png',figsize=(1000,400))  #not easy to control the size of the figure
	plt.show()
	pass


# import plotly.plotly as py
# from plotly.graph_objs import *
# def plotTakeoffDayDensity():
# 	#read data
# 	num=np.zeros((12,31),dtype=np.int)
# 	dataFile='temp/TakeoffDay.csv'
# 	with open(dataFile,'rb') as fh:
# 		lines=fh.readlines()
# 	for i in range(0,num.shape[0]):
# 		fields=lines[i+11].strip().split(',')
# 		for j in range(0,num.shape[1]):
# 			num[i][j]=int(fields[j])

# 	#plot using the tool 'plotly'
# 	data = Data([
# 	    Heatmap(
# 	        z=num,
# 	        # y=['Jan 2013', 'Feb 2013', 'Mar 2013', 'Apr 2013', 'May 2013','Jun 2013','Jul 2013','Aug 2013','Sep 2013','Oct 2013','Nov 2013','Dec 2013',
# 	        # 'Jan1 2014', 'Feb1 2014', 'Mar1 2014', 'Apr 2014', 'May 2014','Jun 2014','Jul 2014','Aug 2014','Sep 2014','Oct 2014','Nov 2014','Dec 2014'],
# 	        y=['Dec 2013', 'Jan1 2014', 'Feb1 2014', 'Mar1 2014', 'Apr 2014', 'May 2014','Jun 2014','Jul 2014','Aug 2014','Sep 2014','Oct 2014','Nov 2014'],
# 	        x=range(1,32)
# 	    )
# 	])
# 	plot_url = py.plot(data, filename='labelled-heatmap')

def plotTakeoffDayDensity():
	#read data
	# num=np.zeros((12,7),dtype=np.int)
	# dataFile='temp/TakeoffWeekday.csv'
	# with open(dataFile,'rb') as fh:
	# 	lines=fh.readlines()
	# for i in range(0,num.shape[0]):
	# 	fields=lines[i].strip().split(',')
	# 	for j in range(0,num.shape[1]):
	# 		num[i][j]=int(fields[j])

	# #plot using the tool 'plotly'
	# data = Data([
	#     Heatmap(
	#         z=num,
	#         y=['Jan1 2014', 'Feb1 2014', 'Mar1 2014', 'Apr 2014', 'May 2014','Jun 2014','Jul 2014','Aug 2014','Sep 2014','Oct 2014','Nov 2014','Dec 2014'],
	#         x=range(1,32)
	#     )
	# ])
	# plot_url = py.plot(data, filename='labelled-heatmap')
	X=np.arange(1,13,1)
	Y=np.arange(1,32,1)
	X,Y=np.meshgrid(X,Y)

	num=np.zeros((12,31),dtype=np.int)
	dataFile='temp/TakeoffWeekday.csv'
	with open(dataFile,'rb') as fh:
		lines=fh.readlines()
	for i in range(0,num.shape[0]):
		fields=lines[i].strip().split(',')
		for j in range(0,num.shape[1]):
			num[i][j]=int(fields[j])
	Z=np.array(num).reshape((12*31,))
	# Plot the density map using nearest-neighbor interpolation
	plt.pcolormesh(X,Y,Z)
	plt.colorbar()
	ax.set_yticklabels(['Jan1 2014', 'Feb1 2014', 'Mar1 2014', 'Apr 2014', 'May 2014','Jun 2014','Jul 2014','Aug 2014','Sep 2014','Oct 2014','Nov 2014','Dec 2014'])

	plt.savefig('heatmap_test.png')

# def plotFlightTimeDensity():
# 	#read data
# 	num=np.zeros((24,24))
# 	dataFile='temp/FlightHour.csv'
# 	with open(dataFile,'rb') as fh:
# 		lines=fh.readlines()
# 	for i in range(0,len(lines)):
# 		fields=lines[i].strip().split(',')
# 		for j in range(0,len(fields)):
# 			num[i][j]+=float(fields[j])

# 	#plot using the tool 'plotly'
# 	data = Data([
# 	    Heatmap(
# 	        z=num,
# 	        x=range(0,24),
# 	        y=range(0,24)
# 	    )
# 	])
# 	fig=Figure(data=data)
# 	fig['layout'].update(
# 		title='Number of travelers at different time per day',
# 		xaxis=XAxis(
# 			title='Arrival time (hour)',
# 			# titlefont=Font(family='Courier New, monospace',size=18,color='#7f7f7f')
# 			),
# 		yaxis=YAxis(
# 			title='Departure time (hour)',
# 			# titlefont=Font(family='Courier New, monospace',size=18,color='#7f7f7f')
# 			),
# 		width=700,
# 		height=700,
# 		autosize=False
# 		)
# 	plot_url = py.plot(fig, filename='labelled-heatmap')
# 	pass

#plots of user profiles
import commonplot
def plotProfileDensity():
	x=[]
	y=[]
	with open('temp/selected_user_profile.csv','rb') as fh:
		lines=fh.readlines()
	for r in lines:
		fields=r.strip().split(',')
		x.append(float(fields[2]))
		y.append(float(fields[3]))
	fileName='plots/profile_scatter.png'
	commonplot.plotScatterWithDensity(x,y,fileName)

def plotDistanceHist():
	d=[]
	with open('temp/pairwise_profile_distance.csv','rb') as fh:
		lines=fh.readlines()
	for r in lines:
		fields=r.strip().split(',')
		d.append(float(fields[2]))
	fileName='plots/distance_hist.png'
	commonplot.plotHistgram(d,50,fileName)

# plotVisitingNumOfCity()
# plotTakeoffDayDensity()
# plotFlightTimeDensity()
# plotProfileDensity()
# plotDistanceHist()

# visualizeTrainedModel()
# visualizeTrainedModel_v2()
# visualizeTrainedModel_v3()

plotTakeoffDayDensity()

