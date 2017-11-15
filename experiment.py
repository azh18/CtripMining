# coding:utf-8
import random
import usergeneration
import datapreparation
import featureextraction
import commonoperation
# import statsforplot
import comparemethods
import time
import threading
import os
import basicmining
import json


class myThread(threading.Thread):
    def __init__(self, threadID, userList):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.userList = userList

    def run(self):
        print "Starting thread " + str(self.threadID)
        # modelUsers(self.userList) # experiment 1
        modelUsers2(self.userList)  # experiment 2
        print "Exiting thread " + str(self.threadID)


def modelUsersMultiThread(allUsers, nThreads):
    if os.path.exists('models') == False:
        os.mkdir('models')
    if os.path.exists('results') == False:
        os.mkdir('results')

    userGroups = commonoperation.divideUsers(allUsers, nThreads)
    threadList = []
    for t in range(0, nThreads):
        threadList.append(myThread(t, userGroups[t]))

    for singleThread in threadList:
        singleThread.start()


def modelUsers(users):  # experiment 1
    # users=usergeneration.generateSamplesOfActiveUsers(10000)
    for u in users:
        print(u)
        datapreparation.prepareData(u)
        traindata = datapreparation.getTraindata(u)
        testdata = datapreparation.getTestdata(u)

        print('only train data')
        # GMM
        m1 = comparemethods.GMMModel(u, modelName='GMM1_2', trainData=traindata, testData=testdata)
        m1.setVariables(nComponents=2)
        m1.run()

        m1 = comparemethods.GMMModel(u, modelName='GMM1_3', trainData=traindata, testData=testdata)
        m1.setVariables(nComponents=3)
        m1.run()

        m2 = comparemethods.fKDEModel(u, modelName='fKDE1_Silverman', trainData=traindata, testData=testdata)
        m2.setVariables(bandwidth='Silverman')
        m2.run()

        neighbors = usergeneration.getSimilarUsers(u)
        featuresListOfNeighbors = []
        for n in neighbors:
            featuresListOfNeighbors += featureextraction.generateFeaturesList(n)

        print('with others data')
        print('GMM')
        m3 = comparemethods.GMMModel(u, modelName='GMM2_2', trainData=traindata + featuresListOfNeighbors,
                                     testData=testdata)
        m3.setVariables(nComponents=2)
        m3.run()

        m3 = comparemethods.GMMModel(u, modelName='GMM2_3', trainData=traindata + featuresListOfNeighbors,
                                     testData=testdata)
        m3.setVariables(nComponents=3)
        m3.run()

        print('fKDE2_Silverman')
        m4 = comparemethods.fKDEModel(u, modelName='fKDE2_Silverman', trainData=traindata + featuresListOfNeighbors,
                                      testData=testdata)
        m4.setVariables(bandwidth='Silverman')
        m4.run()

        print('mix-fKDE1_Silverman')
        m5 = comparemethods.mixfKDEModel(u, modelName='mix-fKDE1_Silverman', trainData=traindata, testData=testdata,
                                         trainDataOfNeighbors=featuresListOfNeighbors)
        m5.setVariables(bandwidth='Silverman', bandwidth1='Silverman')
        m5.run()

        print('mix-fKDE2_cv')
        m6 = comparemethods.mixfKDEModel(u, modelName='mix-fKDE2_cv', trainData=traindata, testData=testdata,
                                         trainDataOfNeighbors=featuresListOfNeighbors)
        m6.setVariables(bandwidth='cv_ml', bandwidth1='cv_ml')
        m6.run()


def modelUsers2(users):  # experiment 2
    '''
	Compare different methods with different bandwidths
	'''
    # users=usergeneration.generateSamplesOfActiveUsers(10000)
    for u in users:
        print(u)
        datapreparation.prepareData(u)
        traindata = datapreparation.getTraindata(u)
        testdata = datapreparation.getTestdata(u)

        print('only train data')
        # GMM
        m1 = comparemethods.GMMModel(u, modelName='GMM1_2', trainData=traindata, testData=testdata)
        m1.setVariables(nComponents=2)
        m1.run()

        m1 = comparemethods.GMMModel(u, modelName='GMM1_3', trainData=traindata, testData=testdata)
        m1.setVariables(nComponents=3)
        m1.run()

        m2 = comparemethods.fKDEModel(u, modelName='fKDE1_Silverman', trainData=traindata, testData=testdata)
        m2.setVariables(bandwidth='Silverman')
        m2.run()

        m2 = comparemethods.fKDEModel(u, modelName='fKDE1_0.5', trainData=traindata, testData=testdata)
        m2.setVariables(bandwidth=0.5)
        m2.run()

        m2 = comparemethods.fKDEModel(u, modelName='fKDE1_1', trainData=traindata, testData=testdata)
        m2.setVariables(bandwidth=1)
        m2.run()

        m2 = comparemethods.fKDEModel(u, modelName='fKDE1_1.5', trainData=traindata, testData=testdata)
        m2.setVariables(bandwidth=1.5)
        m2.run()

        neighbors = usergeneration.getSimilarUsers(u)
        featuresListOfNeighbors = []
        for n in neighbors:
            featuresListOfNeighbors += featureextraction.generateFeaturesList(n)

        # try another method: find neighbors on each dimension, each idx represents a dim
        featuresListOfNeighborsDims = []
        for i in range(0, len(basicmining.getProfile(u))):
            neighbors = usergeneration.getSimilarUsersDim(u, i)
            listTemp = []
            for n in neighbors:
                listTemp += (featureextraction.generateFeaturesList(n))
            featuresListOfNeighborsDims.append(listTemp)



        print('with others data')
        m3 = comparemethods.GMMModel(u, modelName='GMM2_2', trainData=traindata + featuresListOfNeighbors,
                                     testData=testdata)
        m3.setVariables(nComponents=2)
        m3.run()

        m3 = comparemethods.GMMModel(u, modelName='GMM2_3', trainData=traindata + featuresListOfNeighbors,
                                     testData=testdata)
        m3.setVariables(nComponents=3)
        m3.run()

        # print('fKDE2_Silverman')
        m4 = comparemethods.fKDEModel(u, modelName='fKDE2_Silverman', trainData=traindata + featuresListOfNeighbors,
                                      testData=testdata)
        m4.setVariables(bandwidth='Silverman')
        m4.run()

        # print('mix-fKDE2_bw')
        m6 = comparemethods.mixfKDEModel(u, modelName='mix-fKDE2_Silverman', trainData=traindata, testData=testdata,
                                         trainDataOfNeighbors=featuresListOfNeighbors)
        m6.setVariables(bandwidth='Silverman', bandwidth1='Silverman')
        m6.run()

        m6 = comparemethods.mixfKDEModel(u, modelName='mix-fKDE2_0.5', trainData=traindata, testData=testdata,
                                         trainDataOfNeighbors=featuresListOfNeighbors)
        m6.setVariables(bandwidth=0.5, bandwidth1=0.5)
        m6.run()

        m6 = comparemethods.mixfKDEModel(u, modelName='mix-fKDE2_1', trainData=traindata, testData=testdata,
                                         trainDataOfNeighbors=featuresListOfNeighbors)
        m6.setVariables(bandwidth=1, bandwidth1=1)
        m6.run()

        m6 = comparemethods.mixfKDEModel(u, modelName='mix-fKDE2_1.5', trainData=traindata, testData=testdata,
                                         trainDataOfNeighbors=featuresListOfNeighbors)
        m6.setVariables(bandwidth=1.5, bandwidth1=1.5)
        m6.run()

        m6 = comparemethods.mixfKDEModel(u, modelName='mix-fKDE2_cv', trainData=traindata, testData=testdata,
                                         trainDataOfNeighbors=featuresListOfNeighbors)
        m6.setVariables(bandwidth='cv_ml', bandwidth1='cv_ml')
        m6.run()
        # test set components according to the dim
        m7 = comparemethods.mixfKDEModelDim(u, modelName='mix-fKDE2_cv_moreComponents', trainData=traindata, testData=testdata,
                                         trainDataOfNeighbors=featuresListOfNeighborsDims)
        m7.setVariables(bandwidth='cv_ml', bandwidth1='cv_ml')
        m7.run()
        # add city do not work
        # add info about city
        # for i in range(0, len(basicmining.getProfile(u))):
        #     neighbors = usergeneration.getSimilarUsersDimWithExternalInfo(u, 1)
        #     listTemp = []
        #     for n in neighbors:
        #         listTemp += (featureextraction.generateFeaturesList(n))
        #     featuresListOfNeighborsDims.append(listTemp)
        # m8 = comparemethods.mixfKDEModelDim(u, modelName='mix-fKDE2_cv_moreComponentsWithIncome', trainData=traindata, testData=testdata,
        #                                  trainDataOfNeighbors=featuresListOfNeighborsDims)
        # m8.setVariables(bandwidth='cv_ml', bandwidth1='cv_ml')
        # m8.run()


def modelUsers3(users):
    '''
	Compare mix-fKDE2_cv with different kernels
	'''
    for u in users:
        print(u)
        datapreparation.prepareData(u)
        # train,test data from this user
        traindata = datapreparation.getTraindata(u)
        testdata = datapreparation.getTestdata(u)

        neighbors = usergeneration.getSimilarUsers(u)
        # feature from other users, to make mix KDE
        featuresListOfNeighbors = []
        for n in neighbors:
            featuresListOfNeighbors += featureextraction.generateFeaturesList(n)

        m2 = comparemethods.mixfKDEModel(u, modelName='mix-fKDE2_cv_tophat', trainData=traindata, testData=testdata,
                                         trainDataOfNeighbors=featuresListOfNeighbors)
        m2.setVariables(bandwidth='cv_ml', bandwidth1='cv_ml', kernel='tophat')
        m2.run()

        m3 = comparemethods.mixfKDEModel(u, modelName='mix-fKDE2_cv_epanechnikov', trainData=traindata,
                                         testData=testdata, trainDataOfNeighbors=featuresListOfNeighbors)
        m3.setVariables(bandwidth='cv_ml', bandwidth1='cv_ml', kernel='epanechnikov')
        m3.run()

        m4 = comparemethods.mixfKDEModel(u, modelName='mix-fKDE2_cv_exponential', trainData=traindata,
                                         testData=testdata, trainDataOfNeighbors=featuresListOfNeighbors)
        m4.setVariables(bandwidth='cv_ml', bandwidth1='cv_ml', kernel='exponential')
        m4.run()

        m5 = comparemethods.mixfKDEModel(u, modelName='mix-fKDE2_cv_linear', trainData=traindata, testData=testdata,
                                         trainDataOfNeighbors=featuresListOfNeighbors)
        m5.setVariables(bandwidth='cv_ml', bandwidth1='cv_ml', kernel='linear')
        m5.run()

        m6 = comparemethods.mixfKDEModel(u, modelName='mix-fKDE2_cv_cosine', trainData=traindata, testData=testdata,
                                         trainDataOfNeighbors=featuresListOfNeighbors)
        m6.setVariables(bandwidth='cv_ml', bandwidth1='cv_ml', kernel='cosine')
        m6.run()

        m1 = comparemethods.mixfKDEModel(u, modelName='mix-fKDE2_cv_gaussian', trainData=traindata, testData=testdata,
                                         trainDataOfNeighbors=featuresListOfNeighbors)
        m1.setVariables(bandwidth='cv_ml', bandwidth1='cv_ml', kernel='gaussian')
        m1.run()


import numpy as np
from os import listdir
from os.path import isfile, join
from math import isnan


def loadResults(resultFolder='results'):
    onlyfiles = [f for f in listdir(resultFolder) if isfile(join(resultFolder, f))]
    for f in onlyfiles:
        methodName = f.replace('_result.csv', '')
        resultFile = join(resultFolder, f)
        results = []
        with open(resultFile, 'rb') as fh:
            for r in fh:
                res = float(r.strip().split(',')[-1])
                if isnan(res) is False:
                    results.append(res)
        print('{0}: avg is {1}, std is {2}'.format(methodName, sum(results) / len(results), np.std(results)))


#################################################################################
if __name__ == '__main__':
    print('train and test')
    n = 1000  # initial 1000
    nThreads = 8
    # choose 1000 random samples
    userSet = usergeneration.userSet
    basicmining.profileDict = basicmining.generateProfilesMultiThread(userSet, nThreads)
    # test for passenger distribution
    dictTemp = basicmining.generateUnnormalizedProfilesMultiThread(userSet, nThreads)
    fileObj = open("userFeature.json", 'w+')
    jsObj = json.dump(dictTemp, fileObj)
    # fileObj.write(jsObj)
    #jsObj = json.dumps(basicmining.profileDict)
    #fileObj.write(jsObj)
    fileObj.close()
    # test for passenger distribution
    # choose n user to make models
    passengers = random.sample(userSet, n)
    # modelUsers2(passengers)
    modelUsers3(passengers)



    # print('compare results')
    # resultFolder='results'
    # loadResults(resultFolder)


    # print('debug')
    # n=10
    # nThreads=8
    # userSet=usergeneration.userSet
    # basicmining.profileDict=basicmining.generateProfilesMultiThread(userSet,nThreads)
    # passengers=random.sample(userSet,n)
    # modelUsers3(passengers)
