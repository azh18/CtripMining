import MySQLdb
import datetime
import basicmining
import os
import sys


# define type of days
class DayType:
    WORKDAY = 1
    WEEKEND = 2  # except holidays
    HOLIDAY = 3
# THE_DAY_BEFORE_HOLIDAY=4
# THE_DAY_AFTER_HOLIDAY=5


def getDayType(takeOffTime):  # holiday, workday, weekend,
    holidays_2013 = ['2013-01-01', '2013-01-02', '2013-01-03'] + \
                    ['2013-02-09', '2013-02-10', '2013-02-11', '2013-02-12', '2013-02-13', '2013-02-14', '2013-02-15'] + \
                    ['2013-04-04', '2013-04-05', '2013-04-06'] + \
                    ['2013-04-29', '2013-04-30', '2013-05-01'] + \
                    ['2013-06-10', '2013-06-11', '2013-06-12'] + \
                    ['2013-09-19', '2013-09-20', '2013-09-21'] + \
                    ['2013-10-01', '2013-10-02', '2013-10-03', '2013-10-04', '2013-10-05', '2013-10-06', '2013-10-07']
    additional_workdays_2013 = ['2013-01-05', '2013-01-06'] + ['2013-02-16', '2013-02-17'] + ['2013-04-07'] + [
        '2013-04-27', '2013-04-28'] + \
                               ['2013-06-08', '2013-06-09'] + ['2013-09-22'] + ['2013-09-29', '2013-10-12']
    holidays_2014 = ['2013-12-30', '2013-12-31', '2014-01-01'] + \
                    ['2014-01-31', '2014-02-01', '2014-02-02', '2014-02-03', '2014-02-04', '2014-02-05', '2014-02-06'] + \
                    ['2014-04-05', '2014-04-06', '2014-04-07'] + \
                    ['2014-05-01', '2014-05-03', '2014-05-03'] + \
                    ['2014-05-31', '2014-06-01', '2014-06-02'] + \
                    ['2014-09-06', '2014-09-07', '2014-09-08'] + \
                    ['2014-10-01', '2014-10-02', '2014-10-03', '2014-10-04', '2014-10-05', '2014-10-06', '2014-10-07']
    additional_workdays_2014 = [] + ['2014-01-26', '2014-02-08'] + [] + ['2014-05-04'] + \
                               [] + [] + ['2014-09-28', '2013-10-11']
    holidays = holidays_2013 + holidays_2014
    additional_workdays = additional_workdays_2013 + additional_workdays_2014

    takeOffDate = datetime.datetime(takeOffTime.year, takeOffTime.month, takeOffTime.day)
    takeOffDateStr = "{0}-{1}-{2}".format(takeOffDate.year, ('0' + str(takeOffDate.month))[-2:],
                                          ('0' + str(takeOffDate.day))[-2:])

    dayType = DayType.WORKDAY
    if takeOffDateStr in holidays:
        dayType = DayType.HOLIDAY
    elif takeOffDateStr in additional_workdays:
        dayType = DayType.WORKDAY
    elif takeOffDate.weekday() < 5:  # [0,1,2,3,4] [5,6]
        dayType = DayType.WORKDAY
    else:
        dayType = DayType.WEEKEND
    return dayType


def getHourType(time):
    return time.hour


def computeAdvancedDays(orderTime, takeOffTime):
    timespan = datetime.datetime(takeOffTime.year, takeOffTime.month, takeOffTime.day) - datetime.datetime(
        orderTime.year, orderTime.month, orderTime.day)
    return timespan.days


def computeTimeDiff(takeofftime, ordertime):
    duration = takeofftime - ordertime
    d1 = duration.days
    d2 = duration.seconds * 1. / (24 * 60 * 60)
    return d1 + d2


db_ip = 'localhost'
db_ip = '202.120.37.78'
db_user = 'admin'
db_pwd = '2016_NRL_admin123'
db_database = 'ctrip_air_travel'
# db_sock = '/home/mysql/mysql.sock'
db_sock = '/var/lib/mysql/mysql.sock'
# default sock is '/var/lib/mysql/mysql.sock'

pidRecordsDict = {}


def getAllRecordsofAPassengerFromDB(pid):
    if pid in pidRecordsDict:
        return pidRecordsDict[pid]
    results = None
    db = MySQLdb.connect(host=db_ip, user=db_user, passwd=db_pwd,
                         db=db_database, unix_socket=db_sock, port=10002)
    cursor = db.cursor()
    sql = "SELECT uid, orderid, sequence, flight, price, discount, airline, \
	dcity, acity, dport, aport, ordertime, takeofftime, arrivaltime, \
	aircrafttype, seat, seatclass, policy, orderstatus, pricekpi, \
	passengerid, age,gender, isforeigner \
	FROM TravelRecords WHERE passengerid='{0}' \
	ORDER BY takeofftime;".format(pid)
    try:
        cursor.execute(sql)
        results = cursor.fetchall()
    except:
        print "Error: unable to fecth data"
    db.close()
    pidRecordsDict[pid] = results
    return results

def generateProfilesToDB():
    results = []
    db =  MySQLdb.connect(host=db_ip, user=db_user, passwd=db_pwd,
                         db=db_database, unix_socket=db_sock, port=10002)
    cursor = db.cursor()
    sql = "SELECT passengerid from passenger_flight_counts"
    try:
        cursor.execute(sql)
        results = cursor.fetchall()
    except:
        print "Error: unable to fecth data"
    db.close()
    userList = [x[0] for x in results]
    pieceSize = 10000
    userSamllList = [userList[i:i+pieceSize] for i in range(0,len(userList),pieceSize)]
    for userSlice in userSamllList:
        profiles = basicmining.generateProfilesMultiThread(userSlice, 4) # return a dict, k is uid, v is a list sized 6
        for k, v in profiles.items():
            sql = 'insert into UserProfile values(%s,%f,%f,%f,%f,%f,%f,%f)'.format(k, v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7])
            cursor.execute(sql)
        db.commit()
        print 'finish insert %d items\n'.format(len(userSamllList))

def getIncomeFromCityID(city):
    results = []
    db =  MySQLdb.connect(host=db_ip, user=db_user, passwd=db_pwd,
                         db=db_database, unix_socket=db_sock, port=10002)
    cursor = db.cursor()
    sql = "SELECT airport_income.income from airport_income,city_port_char " \
          "where city_port_char.dport=airport_income.`Port` and city_port_char.dcity = " + str(city)
    try:
        cursor.execute(sql)
        results = cursor.fetchall()
    except:
        print "Error: unable to fecth data"
    db.close()
    try:
        income = results[0][0]/50000 # normalized
        return income
    except:
        print "cannot prase income from home:%s" % city
        income = None # far more than normal distance, to avoid chosen as the near because cannot find the real income
        return income




def getPassengersFromDB(nFlightsBound=(20, 50)):
    results = []
    db = MySQLdb.connect(host=db_ip, user=db_user, passwd=db_pwd,
                         db=db_database, unix_socket=db_sock, port=10002)
    cursor = db.cursor()
    sql = "SELECT passengerid from passenger_flight_counts where count>={0} and count<{1};".format(nFlightsBound[0],
                                                                                                   nFlightsBound[
                                                                                                       1])  # add '' to uId. must
    try:
        cursor.execute(sql)
        results = cursor.fetchall()
    except:
        print "Error: unable to fecth data"
    db.close()
    results = [x[0] for x in results]
    return results


def getNFlightsOfAPassengerFromDB(pid):
    db = MySQLdb.connect(host=db_ip, user=db_user, passwd=db_pwd,
                         db=db_database, unix_socket=db_sock, port=10002)
    cursor = db.cursor()
    sql = "SELECT count from passenger_flight_counts where passengerid='{0}';".format(pid)
    try:
        cursor.execute(sql)
        result = cursor.fetchall()
    except:
        print "Error: unable to fecth data"
    db.close()
    return result[0][0]


def executeSQL(sql_str):
    db = MySQLdb.connect(host=db_ip, user=db_user, passwd=db_pwd,
                         db=db_database, unix_socket=db_sock, port=10002)
    cursor = db.cursor()
    try:
        cursor.execute(sql_str)
        result = cursor.fetchall()
    except:
        print "Error: unable to fecth data. FUNCTION:executeSQL"
    db.close()
    return result


# error airport
airportSet = []
cityPortDict = {}


def loadCityAndAirports():
    global airportSet
    global cityPortDict
    with open('temp/city_airport_pair_correct.csv', 'rb') as fh:
        lines = fh.readlines()
    for row in lines:
        [city, ports] = row.replace('\r\n', '').split(':')
        city = int(city)
        ports = ports.split(',')
        if city not in cityPortDict:
            cityPortDict[city] = ports
            airportSet += ports
    pass


def clearDirtyData(records):
    records = list(records)
    if len(airportSet) == 0:
        loadCityAndAirports()
    # errorList=[]
    # clear dPort, aPort
    for k in range(0, len(records)):
        try:
            r = list(records[k])
            dCity = r[7]
            aCity = r[8]
            dPort = r[9]
            aPort = r[10]
            if dPort not in airportSet:
                if dCity == 1:  # beijing
                    dPort = 'PEK'
                elif dCity == 2:  # shanghai
                    if dPort == 'PV' or dPort == 'PVH' or (orderId == '620490566' and sequence == 1):
                        dPort = 'PVG'
                    else:
                        dPort = 'SHA'
                else:
                    dPort = cityPortDict[dCity][0]
                r[9] = dPort
                records[k] = r
                sql = "update TravelRecords set dPort=\'{0}\' where id={1};".format(dPort, r[-1])
                updateDB(sql)
            if aPort not in airportSet:
                if aCity == 1:
                    aPort = 'PEK'
                elif aCity == 2:
                    if aPort == 'PV' or aPort == 'PVH' or (orderId == '620490566' and sequence == 1):
                        aPort = 'PVG'
                    else:
                        aPort = 'SHA'
                else:
                    aPort = cityPortDict[aCity][0]
                r[10] = aPort
                records[k] = r
                sql = "update TravelRecords set aPort=\'{0}\' where id={1};".format(aPort, r[-1])
                # print(sql)
                updateDB(sql)
        except Exception:
            print(sys.exc_info())
            print(r)
        # errorList.append(k)
        # for p in reversed(errorList):
        # 	del records[p]
        # print('the error list is :{0}'.format(errorList))


def updateDB(sql):
    print('update DB begins')
    print(sql)
    db = MySQLdb.connect("localhost", "root", "", "ctrip")
    cursor = db.cursor()
    try:
        cursor.execute(sql)
    except:
        print "Error: unable to update database"
    db.commit()
    db.close()
    print('update DB ends')


portCityDict = None


def convertPortToCity(port):  # convert portcode to cityCode1
    global portCityDict
    if portCityDict is None:
        db = MySQLdb.connect("localhost", "root", "", "ctrip")
        cursor = db.cursor()
        sql = "SELECT portCode, cityCode1 from Airports"  # add '' to uId. must
        try:
            cursor.execute(sql)
            results = cursor.fetchall()
            portCityDict = {}
            for row in results:
                portCityDict[row[0]] = row[1]
        except:
            print "Error: unable to fecth data"
        db.close()
    if port is None:
        print('the key is None')
    if port not in portCityDict:
        print(port)
    return portCityDict[port]


def getRecordsFromFile(dataFile):
    records = []
    with open(dataFile, 'rb') as fh:
        fh.readline()  # the first line is the meaning of fields
        lines = fh.readlines()
        for row in lines:
            fields = row.strip().split(',')
            if len(fields) > 20:
                print(fields)
                c = len(fields) - 20
                str0 = ''.join(fields[0:c + 1])
                fields = fields[c + 1:]
                fields.insert(0, str0)
                print(fields)
            records.append(fields)
    return records


def divideUsers(allUsers, nParts):
    N = len(allUsers)
    n = N / nParts
    userGroups = []
    for k in range(0, nParts - 1):
        userGroups.append(allUsers[k * n:(k + 1) * n])
    userGroups.append(allUsers[(nParts - 1) * n:])
    return userGroups


def mergeDicts(*dict_args):
    result = {}
    for dictionary in dict_args:
        if dictionary is not None:
            result.update(dictionary)
    return result


    # uid='2100565857'
    # records=getAllRecordsofAUserFromDB(uid)

    # sql='update TravelRecords set dPort=\'XXX\' where id=73422167;'
    # updateDB(sql)

    # results=getPassengersFromDB()
    # print(len(results))
    # print(results[:10])

    # pid='MTIwMTAyMTk3OTA1MDcxMDcx'
    # results=getAllRecordsofAPassengerFromDB(pid)
    # print(len(results))
    # print(results[:10])

    # print(getNFlightsOfAPassengerFromDB(pid))
    # print(results)
if __name__ =='__main__':
    cityID = 158
    c = getIncomeFromCityID(cityID)
    print c