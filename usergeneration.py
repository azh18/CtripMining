import os
import pickle
import sys
import random
import basicmining
import commonoperation
import time
import threading
from os.path import join


def loadNums():
    nflights_npid_dict = {}
    fileName = join(workpath, 'nflights_npid.txt')
    with open(fileName, 'r') as fh:
        fh.readline()
        for line in fh:
            nflight, npid = line.replace('\r\n', '').split('\t')
            nflight = int(nflight)
            npid = int(npid)
            nflights_npid_dict[nflight] = npid
    return nflights_npid_dict


def getSimilarUsersCluster(uID, clusterType):
    pass


def getSimilarUsersDim(uId, dimID):
    similarUsersList = []
    # get number of filghts from a user
    num = commonoperation.getNFlightsOfAPassengerFromDB(uId)

    # 1000 candidates
    # why??? for performance?
    bounds = (num - 10, num + 10 + 1)
    # candidatesList=numsDict[num]
    candidatesList = commonoperation.getPassengersFromDB(bounds)
    i = 1
    while True:
        if i >= num * 0.1 or len(candidatesList) > 500:
            break
        candidatesList = commonoperation.getPassengersFromDB((bounds[0] - i, bounds[1] + i))
        i += 1
    candidatesList.remove(uId)

    # select 500 neighbors
    if len(candidatesList) > 500:
        candidatesList = random.sample(candidatesList, 500)
    candidatesDict = {}
    iii = 0

    basicmining.profileDict_neighbors = basicmining.generateProfilesMultiThread(candidatesList, nThreads=4)
    for c in candidatesList:
        candidatesDict[c] = basicmining.getUserDistanceDim(uId, c, dimID)
        # print candidatesDict[c]
        print(iii)
        iii += 1
    s = sorted(candidatesDict, key=candidatesDict.__getitem__)
    neighbors = s[:100]
    return neighbors

# find nearest neighbors in city level
def getSimilarUsersDimWithExternalInfo(uId, externalInfo):
    similarUsersList = []
    # get number of filghts from a user
    num = commonoperation.getNFlightsOfAPassengerFromDB(uId)

    # 1000 candidates
    # why??? for performance?
    bounds = (num - 10, num + 10 + 1)
    # candidatesList=numsDict[num]
    candidatesList = commonoperation.getPassengersFromDB(bounds)
    i = 1
    while True:
        if i >= num * 0.1 or len(candidatesList) > 500:
            break
        candidatesList = commonoperation.getPassengersFromDB((bounds[0] - i, bounds[1] + i))
        i += 1
    candidatesList.remove(uId)

    # select 500 neighbors
    if len(candidatesList) > 500:
        candidatesList = random.sample(candidatesList, 500)
    candidatesDict = {}
    iii = 0

    # basicmining.profileDict_neighbors = basicmining.generateProfilesMultiThreadWithExternalInfo(candidatesList, nThreads=4,externalInfo=1)
    for c in candidatesList:
        candidatesDict[c] = basicmining.getUserDistanceExternal(uId, c, 1)
        # print candidatesDict[c]
        print(iii)
        iii += 1
    s = sorted(candidatesDict, key=candidatesDict.__getitem__)
    neighbors = s[:100]
    return neighbors


def getSimilarUsers(uId):
    similarUsersList = []
    # get number of filghts from a user
    num = commonoperation.getNFlightsOfAPassengerFromDB(uId)

    # 1000 candidates
    # why??? for performance?
    bounds = (num - 10, num + 10 + 1)
    # candidatesList=numsDict[num]
    candidatesList = commonoperation.getPassengersFromDB(bounds)
    i = 1
    while True:
        if i >= num * 0.1 or len(candidatesList) > 500:
            break
        candidatesList = commonoperation.getPassengersFromDB((bounds[0] - i, bounds[1] + i))
        i += 1
    candidatesList.remove(uId)

    # select 500 neighbors
    if len(candidatesList) > 500:
        candidatesList = random.sample(candidatesList, 500)
    candidatesDict = {}
    iii = 0

    basicmining.profileDict_neighbors = basicmining.generateProfilesMultiThread(candidatesList, nThreads=4)
    for c in candidatesList:
        candidatesDict[c] = basicmining.getUserDistance(uId, c)
        # print candidatesDict[c]
        print(iii)
        iii += 1
    s = sorted(candidatesDict, key=candidatesDict.__getitem__)
    neighbors = s[:100]
    return neighbors


def generateSamplesOfActivePassengers(nSamples):
    selectedBounds = [20, 50]  # valid num of user is between 20-50
    passengers = commonoperation.getPassengersFromDB(selectedBounds)
    print nSamples
    return random.sample(passengers, nSamples)


# ===========================================
# homeCity of user
def getHomeCityOfUsers(userList):
    uId_city_dict = {}
    for uId in userList:
        print(uId)
        records = commonoperation.getAllRecordsofAUserFromDB(uId)
        city = basicmining.getHomeCity(records)
        uId_city_dict[uId] = city
    return uId_city_dict


def singleThread(threadId, fileName, userList):
    print('thread {0} begins'.format(threadId))
    tempDict = getHomeCityOfUsers(userList)
    with open(fileName, 'wb') as fh:
        for k in tempDict.keys():
            fh.write('{0},{1}\r\n'.format(k, tempDict[k]))
    print('thread {0} ends'.format(threadId))


class myThread(threading.Thread):
    def __init__(self, threadID, fileName, userList):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.fileName = fileName
        self.userList = userList

    def run(self):
        print "Starting thread " + str(self.threadID)
        singleThread(self.threadID, self.fileName, self.userList)
        print "Exiting thread " + str(self.threadID)


# test code
def test():
    userDict = loadValidNumsDic()
    userList = []
    for k in range(50, 200 + 1):
        userList += userDict[k]

    l = len(userList)
    thread1 = myThread(1, 'temp/uId_city_dict_1.txt', userList[:l / 4])
    thread2 = myThread(2, 'temp/uId_city_dict_2.txt', userList[l / 4:2 * l / 4])
    thread3 = myThread(3, 'temp/uId_city_dict_3.txt', userList[2 * l / 4:3 * l / 4])
    thread4 = myThread(4, 'temp/uId_city_dict_4.txt', userList[3 * l / 4:])

    thread1.start()
    thread2.start()
    thread3.start()
    thread4.start()
    print "Exiting Main Thread"


workpath = 'temp'
# remove inactive or abnormal users
# inactive users: users that has few records (<30)
# abnormal usrs: users that has too many records(>500)
bounds = (30, 500)
N = 10  # initial 1000

# userSet = generateSamplesOfActivePassengers(N)
userSet = ['MzUwNTgyMTk5MDAzMTEwNTU5', 'MjEwNDA0MTk2ODA3MTAwNjQ4', 'NjQwMTAyMTk3ODA5MTAxNTE5',
           'MTMwMTAzMTk4MzA1MTMwMDI2', 'MjMxMDA1MTk3NTAzMTg0MDE0', 'MzIxMDg1MTk3OTEyMjk1MDE4',
           'MzYyMTMyMTk3MTA4MTgzODE3', 'NDIwMTA2MTk3NzAyMTQyODE3', 'MzEwMTAyMTk4MjEyMDcxNjg1',
           'MzIwMTAzMTk3MDA4MjAxNzcz']
# print(userSet[:10])
# print(len(userSet))
# userDict=loadValidUserDic()
# numsDict=loadValidNumsDic()
