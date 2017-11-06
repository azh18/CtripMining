import os

twoCompResultFile = open("./results/mix-fKDE2_cv_result.csv", "r+")
moreCompResultFile = open("./results/mix-fKDE2_cv_moreComponents_result.csv", "r+")
twoCompResult = {}
moreCompResult = {}
avg1 = 0
std1 = 0
cnt1 = 0
avg2 = 0
std2 = 0
cnt2 = 0
for lines in twoCompResultFile:
    twoCompResult[lines.split(',')[0]] = (float(lines.split(',')[2]))
for lines in moreCompResultFile:
    moreCompResult[lines.split(',')[0]] = (float(lines.split(',')[2]))
for user in twoCompResult.keys():
    if user not in moreCompResult.keys():
        twoCompResult.pop(user)
for user in moreCompResult.keys():
    if user not in twoCompResult.keys():
        moreCompResult.pop(user)
for k, v in twoCompResult.items():
    avg1 += v
    cnt1 += 1
for k, v in moreCompResult.items():
    avg2 += v
    cnt2 += 1
avg1 /= cnt1
avg2 /= cnt2
print 'Two Component'
print avg1
print 'more Component'
print avg2