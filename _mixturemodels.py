import math
import sys
import numpy as np

class BaseMixtureModels(object):
	nCom=2 #default
	params=None
	models=[]

	def __init__(self,parameters,models):
		if parameters is None:
			parameters=[0]*len(models)
		if len(parameters)!=len(models):
			raise Exception('can not create mixture models, size of parameters and size of models should be consistent')
		self.params=parameters
		self.models=models
		self.nCom=len(self.params)

	def computeProbs(self,data):
		pass

	def computeProbOfCompoment(self, sample, i):
		pass

	def computeLogLikelihood(self,data): #avg not sum
		probs=list(self.computeProbs(data))
		try:
			ll=sum([math.log(x) for x in probs])*1./len(data)
		except ValueError:
			print(sys.exc_info())
			# ll=float('-inf')
			minValue=-1000
			n=len(probs)
			ll=0
			for k in range(n-1,-1,-1):
				if probs[k]<=0:
					ll+=minValue
					del probs[k]
				else:
					ll+=math.log(probs[k])
			ll/=1.*len(data)
		return ll

from sklearn.neighbors.kde import KernelDensity
class FixBwMixtureModels(BaseMixtureModels):
	def computeProbs(self,data):
		sepProbs=[]
		for i in range(0,self.nCom):
			p=[math.exp(x) for x in self.models[i].score_samples(data)]
			sepProbs.append([self.params[i]*t for t in p])
		probs=[sum(x) for x in zip(*sepProbs)]
		return probs

	def computeProbOfCompoment(self, sample, i):
		p=math.exp(self.models[i].score_samples([sample])[0]) 
		if(np.isnan(p)):
			p=0.
		return p


