import commonoperation

# uid, 0
# orderid, 1 
# sequence,2 
# flight, 3
# price, 4
# discount, 5 
# airline, 6
# dcity, 7
# acity, 8
# dport, 9
# aport, 10
# ordertime, 11
# takeofftime, 12
# arrivaltime, 13
# aircrafttype, 14
# seat, 15
# seatclass, 16 
# policy, 17
# orderstatus, 18 
# pricekpi, 19
# passengerid, 20 
# age, 21
# gender, 22 
# isforeigner, 23

seatClassDict = {'Y': 1, 'F': 2, 'C': 3}  # Y: economy class, F: first class, C: bussiness class

# if this record is the last one, use previous beta
def extractFeatures(record, nextRecord = None, lastBeta = None):
    features = []
    features.append(commonoperation.computeAdvancedDays(record[11], record[12]))  # num of advanced days 0
    if nextRecord != None:
        features.append(commonoperation.computeTimeDiff(nextRecord[12], record[12])) # difference between flights 1
    else:
        features.append(lastBeta)
    features.append(commonoperation.getDayType(record[12]))  # business day or not 2
    features.append(commonoperation.getHourType(record[12]))  # 0-23h /2 3
    features.append(seatClassDict[record[15]])  # seat class: first class, commerical, econimical seat 4
    features.append(record[19]) #zbw: use price kpi? can represent choice more procise? 5
    features.append(record[5])  # price discount 6
    numCompanion = commonoperation.executeSQL('select count from freq_orderid '
                                               'where orderid=\'%s\'' % record[1])
    if len(numCompanion)==0:
        numCompanion = 1
    else:
        numCompanion = numCompanion[0][0]
    features.append(numCompanion) # #companion 7
    return features


def generateFeaturesList(uId):
    # records=commonoperation.getAllRecordsofAUserFromDB(uId)
    # get all values of a user, waiting for extract features
    records = commonoperation.getAllRecordsofAPassengerFromDB(uId)
    # feature only contains reservation and flight
    featuresList = []
    reclen = len(records)
    for i in range(0, len(records)):
        row = records[i]
        if i < reclen - 1:
            nextRow = records[i+1]
            featuresList.append(extractFeatures(row, nextRecord=nextRow))
        else:
            lastBeta = featuresList[i-1][1]
            featuresList.append(extractFeatures(row, lastBeta=lastBeta))
        # featuresList[len(featuresList)-1] = [featuresList[len(featuresList)-1][i] for i in [0, 2, 3, 4, 6]]
    return featuresList
