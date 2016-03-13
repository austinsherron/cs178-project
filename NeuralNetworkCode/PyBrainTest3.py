from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet
import numpy as np
import mltools as ml
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import VarianceThreshold
import pickle

from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.structure import FullConnection
from pybrain.structure import TanhLayer

X = np.genfromtxt("data/kaggle.X1.train.txt",delimiter=',')
Y = np.genfromtxt("data/kaggle.Y.train.txt",delimiter=',')
Xtest = np.genfromtxt("data/kaggle.X1.test.txt",delimiter=',')

#X = SelectKBest(f_classif, k=35).fit_transform(X, Y)
#X = VarianceThreshold(threshold=(.8*.2)).fit_transform(X)
Xtr,Xte,Ytr,Yte = ml.splitData(X,Y,0.8)

#testdat = open('testdat.csv','w')

netbags = []


for iter in range(100):
    for moment in [0.3]:
        for learnRate in [0.05]:
            for epochs in [30]:
                for depth in [3]:
                    for hidw in [8]:

                        Xboot, Yboot = ml.bootstrapData(Xtr,np.array([Ytr]).T,Xtr.shape[0]//50)
                        print Xboot.shape
                        Yboot = Yboot.T
                        print Yboot

                        net = FeedForwardNetwork()

                        w = X.shape[1]
                        hw = hidw#8

                        inl = TanhLayer(w)
                        net.addInputModule(inl)

                        last = inl

                        for i in range(3):
                            newl = SigmoidLayer(hw)
                            net.addModule(newl)
                            net.addConnection(FullConnection(last,newl))
                            last = newl

                        outl = LinearLayer(1)
                        net.addOutputModule(outl)
                        net.addConnection(FullConnection(last,outl))
                        net.sortModules()



                        #print "xshape:",X.shape[1]

                        ds = SupervisedDataSet(X.shape[1],1)

                        dst = SupervisedDataSet(X.shape[1],1)



                        #Xtr = X
                        #Ytr = Y

                        for i in range(Xboot.shape[0]):
                            ds.addSample(Xboot[i,:],Yboot[i])

                        for i in range(Xte.shape[0]):
                            dst.addSample(Xte[i,:],Yte[i])

                        #net = buildNetwork(ds.indim,ds.indim,ds.indim,ds.indim,ds.outdim,recurrent=False)
                        trainer = BackpropTrainer(net,learningrate=learnRate,momentum=moment,verbose=True)
                        #trainer.trainOnDataset(ds,30)
                        trainer.trainUntilConvergence(ds,10)

                        #trainer.testOnData(verbose=True)

                        netbags.append(net)

                        mse = 0.0
                        for i in range(Xte.shape[0]):
                            mse += pow(net.activate(Xte[i])[0]-Yte[i],2)
                        mse /= Xte.shape[0]
                        mseTrain = 0.0
                        for i in range(Xtr.shape[0]):
                            mseTrain += pow(net.activate(Xtr[i])[0]-Ytr[i],2)
                        mseTrain /= Xtr.shape[0]
                        print 'mse(test):{},mse(train):{},epoch:{},width:{},depth:{},momentum:{},learnrate:{}'.format(mse,mseTrain,epochs,hidw,depth,moment,learnRate)
                        #testdat.write('{},{},{},{},{},{},{}\n'.format(mse,mseTrain,epochs,hidw,depth,learnRate,moment))



def predict(entry):
    ymean = 0.0
    for n in netbags:
        ymean += n.activate(entry)[0]
    ymean /= len(netbags)
    return ymean


mse = 0.0
for i in range(Xte.shape[0]):
    mse += pow(predict(Xte[i])-Yte[i],2)
mse /= Xte.shape[0]
mseTrain = 0.0
for i in range(Xte.shape[0]):
    mseTrain += pow(predict(Xtr[i])-Ytr[i],2)
mseTrain /= Xtr.shape[0]

print 'final mse(test):{},mse(train):{}'.format(mse,mseTrain)




#testdat.close()
#modelfile = open('model.dat','w')
#pickle.dump(net,modelfile)
#modelfile.close()

#fh = open('predictions.csv','w')    # open file for upload
#fh.write('ID,Prediction\n')         # output header line
#for i in range(Xtest.shape[0]):
#    yi = net.activate(Xtest[i][0])
#    fh.write('{},{}\n'.format(i+1,yi)) # output each prediction
#fh.close()
