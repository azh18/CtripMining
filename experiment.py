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
        m7 = comparemethods.mixfKDEModelDim(u, modelName='mix-fKDE2_cv_moreComponents', trainData=traindata,
                                            testData=testdata,
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


class ModelRecommendationTest:
    def __init__(self, modelName, k):
        self.modelName = modelName
        self.userList = []
        self.modelList = {}
        self.timestamp = {}
        self.hitRate = {}
        self.realRecord = {}
        self.userTrainData = {}
        self.userTestData = {}
        self.featureListOfNeighbors = {}
        self.featureListOfNeighborsDim = {}
        self.lastRecordOfTrainingSet = {}
        self.k = k
        detailFile = "recommend_detail_%s_k%d.txt" % (self.modelName, self.k)
        hitrateFile = "hitrate_%s_k%d.txt" % (self.modelName, self.k)
        self.detailFile = open(detailFile, 'w+')
        self.hitrateFile = open(hitrateFile, 'w+')

    def closeFile(self):
        self.detailFile.close()
        self.hitrateFile.close()


    # generate train data, test data and similar user data, only once
    def generateTestUser(self, needNeighbor=False, needNeighborDim=False):
        self.userList = usergeneration.userSet
        nThreads = 4
        basicmining.profileDict = basicmining.generateProfilesMultiThread(usergeneration.userSet, nThreads)
        for u in self.userList:
            print(u)
            datapreparation.prepareData(u)
            self.userTrainData[u] = datapreparation.getTraindata(u)
            self.userTestData[u] = datapreparation.getTestdata(u)
            # generate timestamps
            # should only use test set?
            print 'stage 1'
            allRecord = commonoperation.getAllRecordsofAPassengerFromDB(u)
            proportion = 0.7
            splitlingPoint = int(len(allRecord) * proportion)
            self.lastRecordOfTrainingSet[u] = allRecord[splitlingPoint - 1]
            for record in allRecord[splitlingPoint:]:
                if u in self.realRecord.keys():
                    self.realRecord[u].append(record)
                    self.timestamp[u].append(record[11])
                else:
                    self.realRecord[u] = [record]
                    self.timestamp[u] = [record[11]]
            print 'stage 2'
            if needNeighbor:
                featuresListOfNeighbors = []
                neighbors = usergeneration.getSimilarUsers(u)
                for n in neighbors:
                    featuresListOfNeighbors += featureextraction.generateFeaturesList(n)
                self.featureListOfNeighbors[u] = featuresListOfNeighbors
            print 'stage 3'
            if needNeighborDim:
                featuresListOfNeighborsDims = []
                for i in range(0, len(basicmining.getProfile(u))):
                    neighbors = usergeneration.getSimilarUsersDim(u, i)
                    listTemp = []
                    for n in neighbors:
                        listTemp += (featureextraction.generateFeaturesList(n))
                    featuresListOfNeighborsDims.append(listTemp)
                self.featureListOfNeighborsDim[u] = featuresListOfNeighborsDims
            print 'stage 4'
        return 0

    # copy user data among models, no need to generate again
    def copyUserInfo(self, otherTestObject):
        self.userList = otherTestObject.userList
        self.userTrainData = otherTestObject.userTrainData
        self.userTestData = otherTestObject.userTestData
        self.featureListOfNeighbors = otherTestObject.featureListOfNeighbors
        self.featureListOfNeighborsDim = otherTestObject.featureListOfNeighborsDim
        self.realRecord = otherTestObject.realRecord
        self.timestamp = otherTestObject.timestamp
        self.lastRecordOfTrainingSet = otherTestObject.lastRecordOfTrainingSet

    def trainModel(self):
        # for u in self.userList:
        #     print(u)
        #     datapreparation.prepareData(u)
        #     traindata = datapreparation.getTraindata(u)
        #     testdata = datapreparation.getTestdata(u)
        #
        #     print('only train data')
        #     # GMM
        #     m1 = comparemethods.GMMModel(u, modelName='GMM1_2', trainData=traindata, testData=testdata)
        #     m1.setVariables(nComponents=2)
        #     m1.train()
        #
        #     m1 = comparemethods.GMMModel(u, modelName='GMM1_3', trainData=traindata, testData=testdata)
        #     m1.setVariables(nComponents=3)
        #     m1.run()
        #
        #     # m2 = comparemethods.fKDEModel(u, modelName='fKDE1_Silverman', trainData=traindata, testData=testdata)
        #     # m2.setVariables(bandwidth='Silverman')
        #     # m2.run()
        #
        #     # m2 = comparemethods.fKDEModel(u, modelName='fKDE1_0.5', trainData=traindata, testData=testdata)
        #     # m2.setVariables(bandwidth=0.5)
        #     # m2.run()
        #     #
        #     # m2 = comparemethods.fKDEModel(u, modelName='fKDE1_1', trainData=traindata, testData=testdata)
        #     # m2.setVariables(bandwidth=1)
        #     # m2.run()
        #
        #     m2 = comparemethods.fKDEModel(u, modelName='fKDE1_1.5', trainData=traindata, testData=testdata)
        #     m2.setVariables(bandwidth=1.5)
        #     m2.run()
        #
        #     neighbors = usergeneration.getSimilarUsers(u)
        #     featuresListOfNeighbors = []
        #     for n in neighbors:
        #         featuresListOfNeighbors += featureextraction.generateFeaturesList(n)
        #
        #     # try another method: find neighbors on each dimension, each idx represents a dim
        #     featuresListOfNeighborsDims = []
        #     for i in range(0, len(basicmining.getProfile(u))):
        #         neighbors = usergeneration.getSimilarUsersDim(u, i)
        #         listTemp = []
        #         for n in neighbors:
        #             listTemp += (featureextraction.generateFeaturesList(n))
        #         featuresListOfNeighborsDims.append(listTemp)
        #
        #     print('with others data')
        #     # m3 = comparemethods.GMMModel(u, modelName='GMM2_2', trainData=traindata + featuresListOfNeighbors,
        #     #                              testData=testdata)
        #     # m3.setVariables(nComponents=2)
        #     # m3.run()
        #     #
        #     # m3 = comparemethods.GMMModel(u, modelName='GMM2_3', trainData=traindata + featuresListOfNeighbors,
        #     #                              testData=testdata)
        #     # m3.setVariables(nComponents=3)
        #     # m3.run()
        #
        #     # print('fKDE2_Silverman')
        #     # m4 = comparemethods.fKDEModel(u, modelName='fKDE2_Silverman', trainData=traindata + featuresListOfNeighbors,
        #     #                               testData=testdata)
        #     # m4.setVariables(bandwidth='Silverman')
        #     # m4.run()
        #     #
        #     # # print('mix-fKDE2_bw')
        #     # m6 = comparemethods.mixfKDEModel(u, modelName='mix-fKDE2_Silverman', trainData=traindata, testData=testdata,
        #     #                                  trainDataOfNeighbors=featuresListOfNeighbors)
        #     # m6.setVariables(bandwidth='Silverman', bandwidth1='Silverman')
        #     # m6.run()
        #     #
        #     # m6 = comparemethods.mixfKDEModel(u, modelName='mix-fKDE2_0.5', trainData=traindata, testData=testdata,
        #     #                                  trainDataOfNeighbors=featuresListOfNeighbors)
        #     # m6.setVariables(bandwidth=0.5, bandwidth1=0.5)
        #     # m6.run()
        #     #
        #     # m6 = comparemethods.mixfKDEModel(u, modelName='mix-fKDE2_1', trainData=traindata, testData=testdata,
        #     #                                  trainDataOfNeighbors=featuresListOfNeighbors)
        #     # m6.setVariables(bandwidth=1, bandwidth1=1)
        #     # m6.run()
        #     #
        #     # m6 = comparemethods.mixfKDEModel(u, modelName='mix-fKDE2_1.5', trainData=traindata, testData=testdata,
        #     #                                  trainDataOfNeighbors=featuresListOfNeighbors)
        #     # m6.setVariables(bandwidth=1.5, bandwidth1=1.5)
        #     # m6.run()
        #
        #     m6 = comparemethods.mixfKDEModel(u, modelName='mix-fKDE2_cv', trainData=traindata, testData=testdata,
        #                                      trainDataOfNeighbors=featuresListOfNeighbors)
        #     m6.setVariables(bandwidth='cv_ml', bandwidth1='cv_ml')
        #     m6.run()
        #     # test set components according to the dim
        #     m7 = comparemethods.mixfKDEModelDim(u, modelName='mix-fKDE2_cv_moreComponents', trainData=traindata,
        #                                         testData=testdata,
        #                                         trainDataOfNeighbors=featuresListOfNeighborsDims)
        #     m7.setVariables(bandwidth='cv_ml', bandwidth1='cv_ml')
        #     m7.run()
        print 'please run sub-class function: trainModel'

    def getUserLikelihood(self, u):
        model = self.modelList[u]
        model.test()
        return model.ll

    # generate a ticket pool in which tickets are ordered in date of timestamp
    def generateTicketPool(self, timestamp, dcity, acity):
        print 'generating TicketPool ...'
        validRecords = commonoperation.executeSQL('select * from TravelRecords '
                                                  'where orderdate = DATE_FORMAT(\'%s\',\'%%Y-%%m-%%d\') and dcity=\'%s\''
                                                  ' and acity=\'%s\'' % (timestamp, dcity, acity))
        print 'ticketpool generated.'
        if validRecords == None:
            print 'No valid record: from %s to %s at %s' % (dcity, acity, timestamp)
            return []
        validTickets = []
        flights = []
        for r in validRecords:
            if (r[3], r[12], r[15]) not in flights:
                validTickets.append(r)
                flights.append((r[3], r[12], r[15]))
        return validTickets

    def generateTicketFeature(self, ticket, booktime, lastTakeOffTime):
        features = []
        record = ticket
        features.append(commonoperation.computeAdvancedDays(booktime, record[12]))  # num of advanced days 0
        features.append(commonoperation.computeTimeDiff(lastTakeOffTime, record[12]))  # difference between flights 1
        features.append(commonoperation.getDayType(record[12]))  # business day or not 2
        features.append(commonoperation.getHourType(record[12]))  # 0-23h /2 3
        features.append(
            featureextraction.seatClassDict[record[15]])  # seat class: first class, commerical, econimical seat 4
        features.append(record[19])  # zbw: use price kpi? can represent choice more procise? 5
        features.append(record[5])  # price discount 6
        #numCompanion = 1  # try 1

        #features.append(numCompanion)  # #companion 7
        return features

    # one recommendation, return whether recommendation is successful
    def recommend(self, timestamp, user, record, k, lastTakeoffTime):
        # use a record to represent a ticket (only feature about ticket, no user info)
        dcity = record[7]
        acity = record[8]
        tickets = self.generateTicketPool(timestamp, dcity, acity)
        if len(tickets) == 0:
            # no valid ticket
            return -1
        # form feature list
        # a rank list to maintain tickets
        ticketLL = {}
        cntTicket = 0
        for ticket in tickets:
            features = self.generateTicketFeature(ticket, timestamp, lastTakeoffTime)
            model = self.modelList[user]
            ll = model.testACase(features)
            ticketLL[cntTicket] = ll
            cntTicket += 1
        chooseTicket = sorted(ticketLL.items(), key=lambda t: t[1], reverse=True)
        topkTickets = []
        for i in xrange(0, (k if k < len(chooseTicket) else len(chooseTicket))):
            topkTickets.append(tickets[chooseTicket[i][0]])
        # chooseTicket = [tickets[pair[0]] for pair in chooseTicket]
        # chooseTicket is a list of records corresponding to the ticket
        # chooseTicket = chooseTicket[0: (k - 1 if k < len(chooseTicket) else len(chooseTicket) - 1)]
        (isHit, hitrank) = self.hitTest(topkTickets, record)  # True or False
        if isHit:
            self.writeDetailResult(self.modelName, self.k, user, [item[1] for item in chooseTicket], len(tickets), hitrank)
        return isHit

    # test if realRecord match the chosen ticket: flight, takeoff time and seat class
    def hitTest(self, chooseTicket, realRecord):
        Hit = False
        hitrank = 0
        for ticket in chooseTicket:
            hitrank += 1
            match = True and (ticket[3] == realRecord[3])
            match = True and (ticket[12] == realRecord[12])
            match = match and (ticket[15] == realRecord[15])
            Hit = Hit or match
            if Hit:
                return (Hit, hitrank)
        return (Hit, hitrank)

    def runRecommend(self, k):
        self.k = k
        totalNumRecord = 0
        totalSuccess = 0
        for u in self.userList:
            print 'Running recommendation for user %s' % u
            totalNumRecordUser = len(self.realRecord[u])
            totalNumRecord += totalNumRecordUser
            successRecommendNumUser = 0
            lastTakeoffTime = self.lastRecordOfTrainingSet[u][12]
            for (timestamp, record) in zip(self.timestamp[u], self.realRecord[u]):
                print 'Recommend: time %s' % timestamp
                success = self.recommend(timestamp, u, record, k, lastTakeoffTime)
                lastTakeoffTime = record[12]
                if success == -1:
                    totalNumRecordUser -= 1
                    totalNumRecord -= 1
                else:
                    if success:
                        successRecommendNumUser += 1
            if totalNumRecordUser > 0:
                self.hitRate[u] = successRecommendNumUser * 1. / totalNumRecordUser
            else:
                print 'user %s do not have valid recommendation.' % u
            totalSuccess += successRecommendNumUser
            overallLL = self.getUserLikelihood(u)
            self.writeHitRate(u, totalNumRecordUser, successRecommendNumUser, overallLL)
        print 'Finish. Recommend on model %s at k=%d, accurate = %f.' % (self.modelName, k,
                                                                         (totalSuccess * 1. / totalNumRecord))
        return (totalSuccess * 1. / totalNumRecord)

    def writeDetailResult(self, modelName, k, uid, lllist, poolSize, hitRank):
        fp = self.detailFile
        fp.write('Model:%s; k:%d; User :%s; PoolSize:%d; HitRank:%d; ll:%s;\n'%
                 (modelName, k, uid, poolSize, hitRank, str(lllist)))

    def writeHitRate(self, u, totalRecomm, hitNum, overallLL):
        print 'User %s Hitting Rate:%f' % (u, self.hitRate[u])
        fp = self.hitrateFile
        fp.write('Model:%s; k:%d; User :%s; Total:%d; Hit:%d; Hitting Rate:%f; Overall LL:%f;\n' %
                 (self.modelName, self.k, u, totalRecomm, hitNum, self.hitRate[u], overallLL))

    def showHitRate(self, outputFile=None):
        for u in self.hitRate:
            print 'User %s Hitting Rate:%f' % (u, self.hitRate[u])
            if outputFile != None:
                fp = open(outputFile, 'a+')
                fp.write('Model:%s; k:%d; User :%s; Hitting Rate:%f;\n' % (self.modelName, self.k, u, self.hitRate[u]))

class ModelRecommendationTestGMM(ModelRecommendationTest):
    def train(self, nComponents=2):
        for u in self.userList:
            m1 = comparemethods.GMMModel(u, modelName=self.modelName, trainData=self.userTrainData[u],
                                         testData=self.userTestData[u])
            m1.setVariables(nComponents=nComponents)
            m1.train()
            self.modelList[u] = m1
        return 0





class ModelRecommendationTestfKDE(ModelRecommendationTest):
    def train(self, bandwidth=1.5):
        for u in self.userList:
            m2 = comparemethods.fKDEModel(u, modelName=self.modelName, trainData=self.userTrainData[u],
                                          testData=self.userTestData[u])
            m2.setVariables(bandwidth=bandwidth)
            m2.train()
            self.modelList[u] = m2
        return 0


class ModelRecommendationTestmixKDE(ModelRecommendationTest):
    def train(self, simiDefine='Euclid'):
        for u in self.userList:
            if simiDefine == 'Euclid':
                m6 = comparemethods.mixfKDEModel(u, modelName=self.modelName, trainData=self.userTrainData[u]
                                                 , testData=self.userTestData[u],
                                                 trainDataOfNeighbors=self.featureListOfNeighbors[u])
            else:
                m6 = comparemethods.mixfKDEModelDim(u, modelName=self.modelName, trainData=self.userTrainData[u]
                                                 , testData=self.userTestData[u],
                                                 trainDataOfNeighbors=self.featureListOfNeighborsDim[u])
            m6.setVariables(bandwidth='cv_ml', bandwidth1='cv_ml')
            m6.train()
            self.modelList[u] = m6
        return 0


#################################################################################
if __name__ == '__main__':
    # print('train and test')
    # n = 10  # initial 1000
    # nThreads = 8
    # # # choose 1000 random samples
    # userSet = usergeneration.userSet
    # basicmining.profileDict = basicmining.generateProfilesMultiThread(userSet, nThreads)
    # # # test for passenger distribution
    # # dictTemp = basicmining.generateUnnormalizedProfilesMultiThread(userSet, nThreads)
    # # fileObj = open("userFeature.json", 'w+')
    # # jsObj = json.dump(dictTemp, fileObj)
    # # # fileObj.write(jsObj)
    # # # jsObj = json.dumps(basicmining.profileDict)
    # # # fileObj.write(jsObj)
    # # fileObj.close()
    # # # test for passenger distribution
    # # # choose n user to make models
    # passengers = random.sample(userSet, n)
    # modelUsers2(passengers)
    # # modelUsers3(passengers)

    # recommendation experiment
    for k in range(3, 18, 3):
        gmm1 = ModelRecommendationTestGMM('gmm2', k)
        gmm1.generateTestUser(needNeighbor=True, needNeighborDim=True)
        gmm1.train(nComponents=2)
        gmm1.runRecommend(k)
        # gmm1.showHitRate(outputFile='recommendTest_gmm2_k%d.txt'%k)
        gmm1.closeFile()

        gmm2 = ModelRecommendationTestGMM('gmm3', k)
        gmm2.copyUserInfo(gmm1)
        gmm2.train(nComponents=3)
        gmm2.runRecommend(k)
        # gmm2.showHitRate(outputFile='recommendTest_gmm3_k%d.txt'%k)
        gmm2.closeFile()


        fkde = ModelRecommendationTestfKDE('fKDE', k)
        fkde.copyUserInfo(gmm1)
        fkde.train(bandwidth=1.5)
        fkde.runRecommend(k)
        # fkde.showHitRate(outputFile='recommendTest_fkde_k%d.txt'%k)
        fkde.closeFile()

        mixKDE_Euclid = ModelRecommendationTestmixKDE('mixKDE-Euclid', k)
        mixKDE_Euclid.copyUserInfo(gmm1)
        mixKDE_Euclid.train(simiDefine='Euclid')
        mixKDE_Euclid.runRecommend(k)
        # mixKDE_Euclid.showHitRate(outputFile='recommendTest_mixKDE_Euclid_k%d.txt'%k)
        mixKDE_Euclid.closeFile()

        mixKDE_Dim = ModelRecommendationTestmixKDE('mixKDE-Dim', k)
        mixKDE_Dim.copyUserInfo(gmm1)
        mixKDE_Dim.train(simiDefine='Dim')
        mixKDE_Dim.runRecommend(k)
        # mixKDE_Dim.showHitRate(outputFile='recommendTest_mixKDE_Dim_k%d.txt'%k)
        mixKDE_Dim.closeFile()
