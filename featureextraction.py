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


def extractFeatures(record):
    features = []
    features.append(commonoperation.computeAdvancedDays(record[11], record[12]))  # num of advanced days
    features.append(commonoperation.getDayType(record[12]))  # business day or not
    features.append(commonoperation.getHourType(record[12]))  # 0-23h /2
    features.append(seatClassDict[record[15]])  # seat class: first class, commerical, econimical seat
    # features.append(record[19]) #price kpi
    features.append(record[5])  # price discount
    return features


def generateFeaturesList(uId):
    # records=commonoperation.getAllRecordsofAUserFromDB(uId)
    # get all values of a user, waiting for extract features
    records = commonoperation.getAllRecordsofAPassengerFromDB(uId)
    # feature only contains reservation and flight
    featuresList = []
    for row in records:
        featuresList.append(extractFeatures(row))
    return featuresList
