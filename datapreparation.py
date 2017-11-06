import featureextraction

traindataDict={}
testdataDict={}

def prepareData(uId):
	if uId not in traindataDict or uId not in testdataDict:
		featuresList=featureextraction.generateFeaturesList(uId)
		splitData(uId,featuresList)
	pass

#split data into three parts: training data, validation data, testing data
def splitData(uId, featuresList,proportion=0.7):
	n=len(featuresList)
	splitlingPoint=int(n*proportion)
	traindata=featuresList[:splitlingPoint]
	testdata=featuresList[splitlingPoint:]
	traindataDict[uId]=traindata
	testdataDict[uId]=testdata

def splitTraindata(featuresList,proportion=0.5):
	n=len(featuresList)
	splitlingPoint=int(n*proportion)
	traindata=featuresList[:splitlingPoint]
	validationdata=featuresList[splitlingPoint:]
	return (traindata,validationdata)

def getTraindata(uId):
	if uId not in traindataDict:
		prepareData(uId)
	return traindataDict[uId]

def getTestdata(uId):
	if uId not in testdataDict:
		prepareData(uId)
	return testdataDict[uId]
