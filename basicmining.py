import scipy.stats
import datetime
import sklearn.metrics.pairwise
import commonoperation
import numpy as np


#################################################################################
def getFrequentCities(records):  # assumption: the most frequent city is the traveler's home
    freqDic = {}  # key is the city No, value is the frequency(no matter it is the departure city or the arrival city)
    for row in records:
        dCity = row[7]
        aCity = row[8]
        if dCity not in freqDic:
            freqDic[dCity] = 0
        freqDic[dCity] += 1
        if aCity not in freqDic:
            freqDic[aCity] = 0
        freqDic[aCity] += 1
    home = max(freqDic, key=freqDic.get)
    # home=commonoperation.convertCityCodeToCityCode1(home)  #since I use different code for city(i.e., cityCode1)
    proportion = freqDic[home] * 1.0 / (2 * len(records))
    return (home, proportion)

# get home city of user from records
def getHomeCity(records):
    return getFrequentCities(records)[0]

def getAccessCityWithProportion(records):
    return getFrequentCities(records)

# get the expect income
def getExpectIncome(records):
    freqDic = {}  # key is the city No, value is the frequency(no matter it is the departure city or the arrival city)
    for row in records:
        dCity = row[7]
        aCity = row[8]
        if dCity not in freqDic:
            freqDic[dCity] = 0
        freqDic[dCity] += 1
        if aCity not in freqDic:
            freqDic[aCity] = 0
        freqDic[aCity] += 1
    home = max(freqDic, key=freqDic.get)
    income = commonoperation.getIncomeFromCityID(home)
    proportion = freqDic[home] * 1.0 / (2 * len(records))
    return (income, proportion)

#################################################################################
def getProportionOfWorkDays(records):
    dayTypeDict = {}
    for row in records:
        dayType = commonoperation.getDayType(row[12])  # field 12 is the takeOffTime
        if dayType not in dayTypeDict:
            dayTypeDict[dayType] = 0
        dayTypeDict[dayType] += 1
    p = dayTypeDict[commonoperation.DayType.WORKDAY] * 1.0 / len(records)
    p = round(p, 3)
    return (p, p)


#################################################################################
def getSeatPreference(records):
    numOfEconomy = 0
    for row in records:
        seatClass = row[15].upper();
        if seatClass == 'Y':  # economy seat
            numOfEconomy += 1
    p = 1.0 * numOfEconomy / len(records)
    p = round(p, 3)
    return (p, p)


def getDiversityOfSeats(records, numTypes=3):
    # extract all seat class
    seatDict = {}
    for row in records:
        seatClass = row[15].upper()
        if seatClass not in seatDict:
            seatDict[seatClass] = 0
        seatDict[seatClass] += 1
    # compute entropy
    e = scipy.stats.entropy(seatDict.values())
    # max value and min value
    maxValue = scipy.stats.entropy(
        np.ones(numTypes))  # at most 3 kinds of seat: F,C,Y. The entropy is maximized if they have equal probability
    minValue = 0
    if e <= minValue:
        e = minValue
    if e > maxValue:
        print 'function error getDiversityOfSeats: wrong entropy'
    normalizedEntropy = (e - minValue) * 1.0 / (maxValue - minValue)
    e = round(e, 3)
    normalizedEntropy = round(normalizedEntropy, 3)
    return (e, normalizedEntropy)


#################################################################################
# get number of destination from ctrip database
# select count(distinct(cityCode)) from Airports;
def getDiversityOfDestinations(records, numTypes=191):
    # extract desitination, only aCity
    seatDict = {}
    for row in records:
        seatClass = row[8]
        if seatClass not in seatDict:
            seatDict[seatClass] = 0
        seatDict[seatClass] += 1
    # compute entropy
    e = scipy.stats.entropy(seatDict.values())
    # max value and min value
    maxValue = scipy.stats.entropy(
        np.ones(numTypes))  # at most 3 kinds of seat: F,C,Y. The entropy is maximized if they have equal probability
    minValue = 0
    if e <= minValue:
        e = minValue
    if e > maxValue:
        print 'function error getDiversityOfSeats: wrong entropy'
    normalizedEntropy = (e - minValue) * 1.0 / (maxValue - minValue)
    e = round(e, 3)
    normalizedEntropy = round(normalizedEntropy, 3)
    return (e, normalizedEntropy)


#################################################################################
def getAdvancedDays(records):
    total = 0
    for row in records:
        days = commonoperation.computeAdvancedDays(row[11], row[12])
        total += days
    avg = total * 1.0 / len(records)
    # assume the max value is 2 month
    if avg > 60:
        avg = 60
    maxValue = 60
    normalizedAvg = avg * 1.0 / maxValue
    avg = round(avg, 3)
    normalizedAvg = round(normalizedAvg, 3)
    return (avg, normalizedAvg)


def getDeltaDays(records):
    total = 0
    # for row in records:
    N = len(records) - 1
    for i in range(1, N):
        days = commonoperation.computeTimeDiff(records[i][12], records[i - 1][12])
        # print(days)
        if days < 0:
            print(records[i][12], records[i - 1][12])
        total += days
    avg = total * 1. / N
    maxValue = 20
    if avg > maxValue:
        avg = maxValue
    normalizedAvg = avg * 1. / maxValue
    avg = round(avg, 3)
    normalizedAvg = round(normalizedAvg, 3)
    return (avg, normalizedAvg)


#################################################################################

def getTraveFrequency(records):
    # assume max travel frequency is 700
    maxValue = 60
    n = int(1.0 * len(records) / 2)  # the data lasts for 23 months
    if n > maxValue:
        n = maxValue
    normalizedFrequency = n * 1.0 / maxValue
    n = round(n, 3)
    normalizedFrequency = round(normalizedFrequency, 3)
    return (n, normalizedFrequency)


def getAge(records):
    maxValue = 70
    age = records[0][21]
    if age > maxValue:
        age = maxValue
    normalizedAge = age * 1.0 / maxValue
    return (age, normalizedAge)


#################################################################################

profilesPool_n = {}  # pool for normalized profiles
profilesPool_u = {}
import time

#################################################################################
# def getProfile(uId, isNormalized=True):
# 	if isNormalized==True and uId in profilesPool_n:
# 		return profilesPool_n[uId]
# 	elif isNormalized==False and uId in profilesPool_u:
# 		return profilesPool_u[uid]

# 	records=commonoperation.getAllRecordsofAUserFromDB(uId)
# 	profile_n=[]
# 	profile_u=[]
# 	#home, avg # of advanced days, frequency, proportion of first class, proportion of workdays, variety of traval cities
# 	#the distance between home is ovious, we ignore the field so far
# 	#"variety of traval cities" seems no use
# 	p=getTraveFrequency(records)
# 	profile_u.append(p[0])
# 	profile_n.append(p[1])
# 	# p=getDiversityOfSeats(records)
# 	p=getSeatPreference(records)
# 	profile_u.append(p[0])
# 	profile_n.append(p[1])
# 	p=getProportionOfWorkDays(records)
# 	profile_u.append(p[0])
# 	profile_n.append(p[1])
# 	p=getAdvancedDays(records)
# 	profile_u.append(p[0])
# 	profile_n.append(p[1])
# 	p=getDiversityOfDestinations(records)
# 	profile_u.append(p[0])
# 	profile_n.append(p[1])

# 	profilesPool_n[uId]=profile_n
# 	profilesPool_u[uId]=profile_u
# 	if isNormalized ==True:
# 		return profile_n
# 	else:
# 		return profile_u

profileDict = None
profileDict_neighbors = None


def getProfile(uId):
    if profileDict is not None and uId in profileDict:
        return profileDict[uId]
    elif profileDict_neighbors is not None and uId in profileDict_neighbors:
        return profileDict_neighbors[uId]
    else:
        return None


def generateProfles(userList):
    profilesPool_n = {}
    for uId in userList:
        print('generating profile {0}'.format(uId))
        # records=commonoperation.getAllRecordsofAUserFromDB(uId)
        records = commonoperation.getAllRecordsofAPassengerFromDB(uId)
        profile_n = []
        # home, avg # of advanced days, frequency, proportion of first class, proportion of workdays, variety of traval cities
        # the distance between home is ovious, we ignore the field so far
        # "variety of traval cities" seems no use
        p = getTraveFrequency(records)
        profile_n.append(p[1])
        # p=getDiversityOfSeats(records)
        p = getSeatPreference(records)
        profile_n.append(p[1])
        p = getProportionOfWorkDays(records)
        profile_n.append(p[1])
        p = getAdvancedDays(records)
        profile_n.append(p[1])
        p = getDeltaDays(records)
        profile_n.append(p[1])
        p = getDiversityOfDestinations(records)
        profile_n.append(p[1])
        p = getAge(records)
        profile_n.append(p[1])
        profilesPool_n[uId] = profile_n
    return profilesPool_n

# test for the effect of the city info
def generateProflesWithCityInfo(userList):
    profilesPool_n = {}
    for uId in userList:
        print('generating profile {0}'.format(uId))
        # records=commonoperation.getAllRecordsofAUserFromDB(uId)
        records = commonoperation.getAllRecordsofAPassengerFromDB(uId)
        profile_n = []
        p = getExpectIncome(records)
        profile_n.append(p[0])
        profilesPool_n[uId] = profile_n
    return profilesPool_n


def generateUnnormlaizedProfles(userList):
    profilesPool_u = {}
    for uId in userList:
        print('generating profile {0}'.format(uId))
        # records=commonoperation.getAllRecordsofAUserFromDB(uId)
        records = commonoperation.getAllRecordsofAPassengerFromDB(uId)
        profile_u = []
        # home, avg # of advanced days, frequency, proportion of first class, proportion of workdays, variety of traval cities
        # the distance between home is ovious, we ignore the field so far
        # "variety of traval cities" seems no use
        p = getTraveFrequency(records)
        profile_u.append(p[0])
        # p=getDiversityOfSeats(records)
        p = getSeatPreference(records)
        profile_u.append(p[0])
        p = getProportionOfWorkDays(records)
        profile_u.append(p[0])
        p = getAdvancedDays(records)
        profile_u.append(p[0])
        p = getDeltaDays(records)
        profile_u.append(p[0])
        p = getDiversityOfDestinations(records)
        profile_u.append(p[0])
        p = getAge(records)
        profile_u.append(p[0])
        profilesPool_u[uId] = profile_u
    return profilesPool_u


import threading


class getProfilesThread(threading.Thread):
    def __init__(self, threadID, userList):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.userList = userList
        self.profileDict = None

    def run(self):
        print "Starting thread " + str(self.threadID)
        self.profileDict = generateProfles(self.userList)
        print "Exiting thread " + str(self.threadID)

class getProfilesThreadWithExternalInfo(threading.Thread):
    def __init__(self, threadID, userList, externalInfo):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.userList = userList
        self.profileDict = None
        self.externalInfo = externalInfo

    def run(self):
        print "Starting thread " + str(self.threadID)
        self.profileDict = generateProflesWithCityInfo(self.userList)
        print "Exiting thread " + str(self.threadID)

def generateProfilesMultiThread(allUsers, nThreads=4):
    userGroup = commonoperation.divideUsers(allUsers, nThreads)
    threadList = []
    for t in range(0, nThreads):
        threadList.append(getProfilesThread(t, userGroup[t]))

    for singleThread in threadList:
        singleThread.start()

    for singleThread in threadList:
        singleThread.join()

    # all generated user profile
    profileDictGroup = []
    for singleThread in threadList:
        profileDictGroup.append(singleThread.profileDict)

    # print(profileDictGroup)
    profileDict = commonoperation.mergeDicts(*profileDictGroup)
    print('after merging, the len of profile is {0}'.format(len(profileDict)))
    # for p in profileDictGroup:
    # 	print(len(p))
    return profileDict


def generateProfilesMultiThreadWithExternalInfo(allUsers, nThreads=4, externalInfo=1):
    userGroup = commonoperation.divideUsers(allUsers, nThreads)
    threadList = []
    for t in range(0, nThreads):
        threadList.append(getProfilesThreadWithExternalInfo(t, userGroup[t], externalInfo))

    for singleThread in threadList:
        singleThread.start()

    for singleThread in threadList:
        singleThread.join()

    # all generated user profile
    profileDictGroup = []
    for singleThread in threadList:
        profileDictGroup.append(singleThread.profileDict)

    # print(profileDictGroup)
    profileDict = commonoperation.mergeDicts(*profileDictGroup)
    print('after merging, the len of profile is {0}'.format(len(profileDict)))
    # for p in profileDictGroup:
    # 	print(len(p))
    return profileDict


class getUnnormalizedProfilesThread(threading.Thread):
    def __init__(self, threadID, userList):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.userList = userList
        self.profileDict = None

    def run(self):
        print "Starting thread " + str(self.threadID)
        self.profileDict = generateUnnormlaizedProfles(self.userList)
        print "Exiting thread " + str(self.threadID)


def generateUnnormalizedProfilesMultiThread(allUsers, nThreads=4):
    userGroup = commonoperation.divideUsers(allUsers, nThreads)
    threadList = []
    for t in range(0, nThreads):
        threadList.append(getUnnormalizedProfilesThread(t, userGroup[t]))

    for singleThread in threadList:
        singleThread.start()

    for singleThread in threadList:
        singleThread.join()

    profileDictGroup = []
    for singleThread in threadList:
        profileDictGroup.append(singleThread.profileDict)

    # print(profileDictGroup)
    profileDict = commonoperation.mergeDicts(*profileDictGroup)
    print('after merging, the len of profile is {0}'.format(len(profileDict)))
    # for p in profileDictGroup:
    # 	print(len(p))
    return profileDict

def compareProfiles(x, y):
    MAX = 10000
    if x is None or y is None:
        print 'one of profile is not in db\n'
        return MAX
    d = sklearn.metrics.pairwise.euclidean_distances(np.array(x).reshape(-1, 1), np.array(y).reshape(-1, 1))[0][0]
    d = round(d, 3)
    return d


def getUserDistance(uId1, uId2):
    profile1 = getProfile(uId1)
    profile2 = getProfile(uId2)
    return compareProfiles(profile1, profile2)

# return the delta on given dimension
def getUserDistanceDim(uId,c,dimID):
    profile1 = getProfile(uId)
    profile2 = getProfile(c)
    MAX = 10000
    if profile1 is None or profile2 is None:
        print 'one of profile is not in db\n'
        return MAX
    return abs(profile1[dimID] - profile2[dimID])

def getUserDistanceExternal(uId,c,externalID):
    profile1 = getExpectIncome(commonoperation.getAllRecordsofAPassengerFromDB(uId))[0]
    profile2 = getExpectIncome(commonoperation.getAllRecordsofAPassengerFromDB(c))[0]
    MAX = 10000
    if profile1 is None or profile2 is None:
        print 'one of profile is not in db\n'
        return MAX
    return abs(profile1 - profile2)

#################################################################################

