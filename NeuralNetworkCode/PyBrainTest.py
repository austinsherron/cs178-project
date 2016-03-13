from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet
import numpy as np
import mltools as ml
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import VarianceThreshold

from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.structure import FullConnection
from pybrain.structure import TanhLayer, SoftmaxLayer
from sklearn.preprocessing import PolynomialFeatures


X = np.genfromtxt("data/kaggle.X1.train.txt",delimiter=',')
Y = np.genfromtxt("data/kaggle.Y.train.txt",delimiter=',')
Xtest = np.genfromtxt("data/kaggle.X1.test.txt",delimiter=',')

poly = PolynomialFeatures(2)

X = poly.fit_transform(X)


#X = SelectKBest(f_classif, k=35).fit_transform(X, Y)
#X = VarianceThreshold(threshold=(.8*.2)).fit_transform(X)

net = FeedForwardNetwork()

w = X.shape[1]

l1 = TanhLayer(w)
l2 = SigmoidLayer(8)
l3 = SigmoidLayer(8)
l4 = SigmoidLayer(8)
l5 = LinearLayer(1)

net.addInputModule(l1)
net.addModule(l2)
net.addModule(l3)
net.addModule(l4)
net.addOutputModule(l5)

c12 = FullConnection(l1,l2)
c23 = FullConnection(l2,l3)
c34 = FullConnection(l3,l4)
c45 = FullConnection(l4,l5)
net.addConnection(c12)
net.addConnection(c23)
net.addConnection(c34)
net.addConnection(c45)
net.sortModules()



print "xshape:",X.shape[1]

ds = SupervisedDataSet(X.shape[1],1)

dst = SupervisedDataSet(X.shape[1],1)

Xtr,Xte,Ytr,Yte = ml.splitData(X,Y,0.8)

for i in range(Xtr.shape[0]):
    ds.addSample(Xtr[i,:],Ytr[i])

for i in range(Xte.shape[0]):
    dst.addSample(Xte[i,:],Yte[i])

#net = buildNetwork(ds.indim,ds.indim,ds.indim,ds.indim,ds.outdim,recurrent=False)
trainer = BackpropTrainer(net,learningrate=0.05,momentum=0.3,verbose=True)
#trainer.trainOnDataset(ds,30)
trainer.trainUntilConvergence(ds,30)

#trainer.testOnData(verbose=True)





mse = 0.0
for i in range(Xte.shape[0]):
    mse += pow(net.activate(Xte[i])-Yte[i],2)
mse /= Xte.shape[0]
print 'mse',mse