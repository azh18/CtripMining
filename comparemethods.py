# for evaluation
# methods: GMM, fKDE, mix-fKDE, mix-aKDE

# prepare data
# trainData=None
# testData=None

# train: construct models with each methods
# test: compare their performance

import os
import os.path
import pickle
import math
import numpy as np


# base class
class BehaviorModel(object):
    uid = None
    modelName = 'base'
    trainData = None
    testData = None
    # validationData=None
    ll = None

    modelFile = None
    trainedModel = None
    resultFile = None

    def __init__(self, uid, modelName='base', trainData=None, testData=None, modelFile=None):
        self.uid = uid
        self.modelName = modelName
        self.trainData = trainData
        self.testData = testData
        self.modelFile = modelFile
        pass

    def computeProbs(self, data):
        pass

    def setVariables(self):
        pass

        # to be override

    def train(self):
        self.trainedModel = None
        # #save the model to file
        # self.saveModel()
        pass

    # to be override
    def test(self):  # evaluate by log likelihood function
        self.ll = None
        pass

    def testACase(self, testData):
        self.ll = None
        pass

    # def prepareData(self):
    # 	pass

    def saveModel(self):
        # generate a folder that saves the trained models
        # modelFolder='models/{0}'.format(self.modelName)
        if os.path.exists('models') is False:
            os.mkdir('models')
        modelFolder = os.path.join('models', self.modelName)
        if os.path.exists(modelFolder) is False:
            os.mkdir(modelFolder)
        self.modelFile = os.path.join(modelFolder, '{0}_{1}.model'.format(self.uid, self.modelName))
        with open(self.modelFile, 'wb') as fh:
            pickle.dump(self.trainedModel, fh)
        pass

    def loadModel(self):
        with open(self.modelFile, 'rb') as fh:
            self.trainedModel = pickle.load(fh)
        pass

    def saveResult(self):
        if os.path.exists('results') is False:
            os.mkdir('results')
        self.resultFile = os.path.join('results', '{0}_result.csv'.format(self.modelName))
        with open(self.resultFile, 'ab+') as fh:
            fh.write('{0},{1},{2}\r\n'.format(self.uid, self.modelName, self.ll))
        pass

    def run(self):
        print('train')
        self.train()
        print('save model')
        self.saveModel()
        print('test')
        self.test()
        self.saveResult()
        return self


# GMM
from sklearn import mixture


class GMMModel(BehaviorModel):
    # variables
    nComponents = 0

    def setVariables(self, nComponents=3):
        self.nComponents = nComponents
        pass

    def train(self):
        self.trainedModel = mixture.GMM(self.nComponents)  # assume that the number of components is 3
        self.trainedModel.fit(self.trainData)  # fit using EM
        pass

    def test(self):
        self.ll = sum(self.trainedModel.score(self.testData)) * 1. / len(self.testData)  # compute log likelihood
        pass

    def testACase(self, testData):
        ll = sum(self.trainedModel.score(testData)) * 1. / len(testData)
        return ll



# fKDE
import sys
from sklearn.neighbors.kde import KernelDensity


class fKDEModel(BehaviorModel):
    # variables
    bandwidth = None
    kernel = None

    def setVariables(self, bandwidth='Scott', kernel='gaussian'):
        # if bandwidth=='Scott' or bandwidth=='Silverman':
        if isinstance(bandwidth, str):
            self.bandwidth = fKDEModel.computeBandwidth(self.trainData, rule=bandwidth, kernel=self.kernel)
        else:
            try:
                self.bandwidth = float(bandwidth)
            except:
                print(sys.exc_info())
        if isinstance(kernel, str):
            self.kernel = kernel
        else:
            print('Invalid kernel name.')
            raise Exception

    @staticmethod
    def computeBandwidth(data, rule='Scott', kernel='gaussian'):  # gloabal bandwidth
        # print('computing bandwidth by the given rule')
        n = len(data)
        d = len(data[0])
        bw = None
        if rule == 'Scott':
            bw = math.pow(n, -1. / (d + 4))
            pass
        elif rule == 'Silverman':
            bw = math.pow(n * (d + 2) / 4., -1. / (d + 4))
            pass
        elif rule == 'cv_ml':  # compute the bandwidth via cross-validation with the score of maximum likelihood
            from sklearn.grid_search import GridSearchCV
            grid = GridSearchCV(KernelDensity(kernel=kernel), {'bandwidth': np.linspace(0.5, 2.5, 20)}, cv=5)
            grid.fit(data, None)
            bw = grid.best_params_['bandwidth']
        # print(bw)
        else:
            print('error str for computing bandwidth')
            return None
        print('the bandwidth is {0}'.format(bw))
        return bw

    def train(self):
        self.trainedModel = KernelDensity(kernel=self.kernel, bandwidth=self.bandwidth).fit(self.trainData)

    def test(self):
        self.ll = sum(self.trainedModel.score_samples(self.testData)) * 1. / len(
            self.testData)  # compute log likelihood

    def testACase(self, testData):
        ll = sum(self.trainedModel.score_samples(testData)) * 1. / len(testData)
        return ll


# mixfKDE
import datapreparation
import _mixturemodels
import em
from sklearn.neighbors.kde import KernelDensity

class mixfKDEModelDim(fKDEModel):
    trainDataOfNeighborsDim = None
    # variables
    bandwidth = None
    bandwidth1 = None
    # bandwidth of different components
    bandwidthNeighbor = []
    kernel = None

    def __init__(self, uid, modelName='mix-fKDE', trainData=None, testData=None, trainDataOfNeighbors=None,
                 modelFile=None):
        super(mixfKDEModelDim, self).__init__(uid, modelName, trainData, testData, modelFile)
        self.trainDataOfNeighborsDim = trainDataOfNeighbors

    def setVariables(self, bandwidth='Silverman', bandwidth1='Silverman', kernel='gaussian'):
        if isinstance(bandwidth, str):
            # if bandwidth=='Scott' or bandwidth=='Silverman':
            self.bandwidth = mixfKDEModelDim.computeBandwidth(self.trainData, rule=bandwidth, kernel=kernel)
        else:
            try:
                self.bandwidth = float(bandwidth)
            except:
                print(sys.exc_info())

        if isinstance(bandwidth1, str):
            # if bandwidth=='Scott' or bandwidth=='Silverman':
            for i in range(0, len(self.trainDataOfNeighborsDim)):
                bandwidthNow = mixfKDEModelDim.computeBandwidth(self.trainDataOfNeighborsDim[i], rule=bandwidth1, kernel=kernel)
                self.bandwidthNeighbor.append(bandwidthNow)
            # self.bandwidth1 = mixfKDEModelDim.computeBandwidth(self.trainDataOfNeighborsDim, rule=bandwidth1, kernel=kernel)
        else:
            try:
                for i in range(0, len(self.trainDataOfNeighborsDim)):
                    self.bandwidthNeighbor.append(float(bandwidth1))
                # self.bandwidth1 = float(bandwidth1)
            except:
                print(sys.exc_info())

        if isinstance(kernel, str):
            self.kernel = kernel
        else:
            print('Invalid kernel name.')
            raise Exception

    def train(self):
        [parttraindata, validationdata] = datapreparation.splitTraindata(self.trainData)
        # individual modeling
        kdeModel = KernelDensity(kernel=self.kernel, bandwidth=self.bandwidth).fit(parttraindata)
        # modeling based on others' data
        # for each dim, train a model
        otherKdeModel = []
        for i in range(0,len(self.trainDataOfNeighborsDim)):
            aModel = KernelDensity(kernel=self.kernel, bandwidth=self.bandwidthNeighbor[i]).fit(self.trainDataOfNeighborsDim[i])
            otherKdeModel.append(aModel)

        # mixture modeling
        self.trainedModel = _mixturemodels.FixBwMixtureModels(parameters=None, models=[kdeModel] + otherKdeModel)
        em.runEM(validationdata, mixmodels=self.trainedModel)
        print(self.trainedModel.params)

    def test(self):
        # probs=self.computeProbs(self.testData)
        # self.ll= sum([math.log(x) for x in probs])
        self.ll = self.trainedModel.computeLogLikelihood(self.testData)

    def testACase(self, testData):
        ll = sum(self.trainedModel.computeLogLikelihood(testData)) * 1. / len(testData)
        return ll

class mixfKDEModel(fKDEModel):
    trainDataOfNeighbors = None
    # variables
    bandwidth = None
    bandwidth1 = None
    kernel = None

    def __init__(self, uid, modelName='mix-fKDE', trainData=None, testData=None, trainDataOfNeighbors=None,
                 modelFile=None):
        super(mixfKDEModel, self).__init__(uid, modelName, trainData, testData, modelFile)
        self.trainDataOfNeighbors = trainDataOfNeighbors

    def setVariables(self, bandwidth='Silverman', bandwidth1='Silverman', kernel='gaussian'):
        if isinstance(bandwidth, str):
            # if bandwidth=='Scott' or bandwidth=='Silverman':
            self.bandwidth = mixfKDEModel.computeBandwidth(self.trainData, rule=bandwidth, kernel=kernel)
        else:
            try:
                self.bandwidth = float(bandwidth)
            except:
                print(sys.exc_info())

        if isinstance(bandwidth1, str):
            # if bandwidth=='Scott' or bandwidth=='Silverman':
            self.bandwidth1 = mixfKDEModel.computeBandwidth(self.trainDataOfNeighbors, rule=bandwidth1, kernel=kernel)
        else:
            try:
                self.bandwidth1 = float(bandwidth1)
            except:
                print(sys.exc_info())

        if isinstance(kernel, str):
            self.kernel = kernel
        else:
            print('Invalid kernel name.')
            raise Exception

    def train(self):
        [parttraindata, validationdata] = datapreparation.splitTraindata(self.trainData)
        # individual modeling
        kdeModel = KernelDensity(kernel=self.kernel, bandwidth=self.bandwidth).fit(parttraindata)
        # modeling based on others' data
        otherKdeModel = KernelDensity(kernel=self.kernel, bandwidth=self.bandwidth1).fit(self.trainDataOfNeighbors)

        # mixture modeling
        self.trainedModel = _mixturemodels.FixBwMixtureModels(parameters=None, models=[kdeModel, otherKdeModel])
        em.runEM(validationdata, mixmodels=self.trainedModel)
        print(self.trainedModel.params)

    def test(self):
        # probs=self.computeProbs(self.testData)
        # self.ll= sum([math.log(x) for x in probs])
        self.ll = self.trainedModel.computeLogLikelihood(self.testData)

    def testACase(self, testData):
        ll = sum(self.trainedModel.computeLogLikelihood(testData)) * 1. / len(testData)
        return ll
