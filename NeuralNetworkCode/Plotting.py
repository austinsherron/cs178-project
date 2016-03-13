import matplotlib.pyplot as plt
import numpy as np


dat = np.genfromtxt('testdat2.csv',delimiter=',')


#dat = dat[9:]

dat = dat[:]


best = None
bestval = 1000
for d in dat:
    if d[0] < bestval:
        best = d
        bestval = d[0]

print 'best:',best

print dat

#plt.plot(np.log2(dat[:,3]),dat[:,0])
#plt.plot(np.log2(dat[:,3]),dat[:,1])


#plt.plot(np.log2(dat[:,3]),dat[:,0])
#plt.plot(np.log2(dat[:,3]),dat[:,1])
#plt.scatter(np.log2(dat[:,3]),dat[:,0],c=dat[:,4])
#plt.scatter(np.log2(dat[:,3]),dat[:,1],c=dat[:,4])

plt.plot(dat[:,5],dat[:,0])
plt.plot(dat[:,5],dat[:,1])

plt.legend(['testing','training'])
plt.show()