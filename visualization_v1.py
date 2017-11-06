import os
import random
import comparemethods
import commonoperation
import matplotlib as mpl
mpl.use('agg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Times New Roman'],'size':18})
rc('text', usetex=True)
rc('lines', **{'linewidth':2.0,'marker':'o','markersize':0})


def visualizeTrainedModel_v3(): 
	'''
	3d Surface plot
	'''
	modelName='mix-fKDE2_cv'
	modelPath='models/{0}'.format(modelName)
	# trainedModels=random.sample(os.listdir(modelPath),10)
	trainedModels=['MTUyMzAxMTk4MTA5Mjk1NTIx_mix-fKDE2_cv.model']
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

# def plotTakeoffDayDensity():
# 	X=np.arange(32)
# 	Y=np.arange(13)
# 	X,Y=np.meshgrid(X,Y)

# 	num=np.zeros((12,31),dtype=np.int)
# 	dataFile='temp/TakeoffDay.csv'
# 	with open(dataFile,'rb') as fh:
# 		lines=fh.readlines()
# 	for i in range(0,num.shape[0]):
# 		fields=lines[i+11].strip().split(',')
# 		for j in range(0,num.shape[1]):
# 			num[i][j]=int(fields[j])
# 	Z=np.array(num)
# 	# print(X.shape,Y.shape,Z.shape)
# 	# print(X,Y,Z)

# 	fig=plt.figure()
# 	ax = plt.subplot(111)
# 	# Plot the density map using nearest-neighbor interpolation
# 	plt.pcolormesh(X,Y,Z)
# 	plt.colorbar(orientation="horizontal")

# 	pos1 = ax.get_position() # get the original position 
# 	pos2 = [pos1.x0 + 0.005, pos1.y0,  pos1.width, pos1.height] 
# 	ax.set_position(pos2) # set a new position
# 	ax.set_xlabel(r'Day of month')
# 	# ax.set_ylabel(r'Month')
# 	ax.set_xticks(np.linspace(0.5,30.5,31))
# 	ax.set_xticklabels([str(e) for e in np.arange(1,32,1)],fontsize=12)
# 	ax.set_xlim(xmax=31)
# 	ax.set_yticks(np.linspace(0.5,11.5,12))
# 	ax.set_yticklabels(['Dec 2013', 'Jan 2014', 'Feb 2014', 'Mar 2014', 'Apr 2014', 'May 2014','Jun 2014','Jul 2014','Aug 2014','Sep 2014','Oct 2014','Nov 2014'])

# 	plt.savefig('plots/heatmap_of_takeoff_day.png')
# 	plt.close(fig)

# import commonplot
# def plotProfileDensity():
# 	x=[]
# 	y=[]
# 	with open('temp/selected_user_profile.csv','rb') as fh:
# 		lines=fh.readlines()
# 	for r in lines:
# 		fields=r.strip().split(',')
# 		x.append(float(fields[2]))
# 		y.append(float(fields[3]))
# 	fileName='plots/profile_scatter.png'
# 	commonplot.plotScatterWithDensity(x,y,fileName)

# def plotDistanceHist():
# 	d=[]
# 	with open('temp/pairwise_profile_distance.csv','rb') as fh:
# 		lines=fh.readlines()
# 	for r in lines:
# 		fields=r.strip().split(',')
# 		d.append(float(fields[2]))
# 	fileName='plots/distance_hist.png'
# 	commonplot.plotHistgram(d,50,fileName)

if __name__ == '__main__':
	visualizeTrainedModel_v3()
	# plotTakeoffDayDensity()
	# plotProfileDensity()
	# plotDistanceHist()
