import os
import numpy as np


class ShowTestResult:
    def __init__(self):
        self.modelName = []
        self.testUserSet = set()
        self.resultDictList = []
        self.resultFiles = []
        self.resultAvg = []
        self.resultStd = []

    def testAFile(self, fileName, path):
        modelName = fileName
        resultFile = os.path.join(path, fileName)
        resultFile = open(resultFile, 'r')
        resultDict = {}
        for lines in resultFile:
            resultDict[lines.split(',')[0]] = (float(lines.split(',')[2]))
        if len(self.testUserSet) == 0:
            self.testUserSet = resultDict.keys()
        else:
            self.testUserSet = set(resultDict.keys()).intersection(self.testUserSet)
        self.resultDictList.append(resultDict)
        return 0

    def findAllResultFiles(self, floder):
        files = os.listdir(floder)
        validFiles = [f for f in files if '.csv' in f]
        self.modelName = [f.split('.cs')[0] for f in files if '.csv' in f]
        return validFiles

    def parseResultDict(self):
        for model in self.resultDictList:
            for k in model.keys():
                if k not in self.testUserSet:
                    model.pop(k)
            resultThisModel = np.array([v for k, v in model.items()])
            avg = np.average(resultThisModel)
            std = np.std(resultThisModel)
            self.resultAvg.append(avg)
            self.resultStd.append(std)
        return 0

    def run(self, floder):
        validFiles = self.findAllResultFiles(floder)
        for f in validFiles:
            self.testAFile(f, floder)
        self.parseResultDict()
        for idx in range(0,len(self.modelName)):
            print '{0}:\nAvg:{1};Std:{2}.'.format(self.modelName[idx], self.resultAvg[idx], self.resultStd[idx])
        return 0


if __name__ == "__main__":
    testResult = ShowTestResult()
    testResult.run('./results_8features')

# twoCompResultFile = open("./results/mix-fKDE2_cv_result.csv", "r+")
# moreCompResultFile = open("./results/mix-fKDE2_cv_moreComponents_result.csv", "r+")
# twoCompResult = {}
# moreCompResult = {}
# for lines in twoCompResultFile:
#     twoCompResult[lines.split(',')[0]] = (float(lines.split(',')[2]))
# for lines in moreCompResultFile:
#     moreCompResult[lines.split(',')[0]] = (float(lines.split(',')[2]))
# for user in twoCompResult.keys():
#     if user not in moreCompResult.keys():
#         twoCompResult.pop(user)
# for user in moreCompResult.keys():
#     if user not in twoCompResult.keys():
#         moreCompResult.pop(user)
# twoCompResult = np.array([v for k, v in twoCompResult.items()])
# moreCompResult = np.array([v for k, v in moreCompResult.items()])
#
# print 'Two Component:\nAvg:{0};Std:{1}.\n'.format(np.average(twoCompResult), np.std(twoCompResult))
# print 'More Component:\nAvg:{0};Std:{1}.\n'.format(np.average(moreCompResult), np.std(moreCompResult))
