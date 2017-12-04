import sys
for k in xrange(3,18,3):
    filename = 'hitrate_mixKDE-Dim_k%d.txt' % k
    total = 0.0
    hit = 0.0
    for line in open(filename):
        totalThisTime = int(line.split(':')[4].split(';')[0])
        hitThisTime = int(line.split(':')[5].split(';')[0])
        total += totalThisTime
        hit += hitThisTime
    print 'for k=%d, accuracy=%f' % (k, hit/total)
