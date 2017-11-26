import numpy as np;


def tanh(x):
    return np.tanh(x);

def tanh_deriv(x):
    return 1.0-tanh(x)*tanh(x);

def Logistic(x):
    return 1/(1+np.exp(-x));

def Logistic_deriv(x):
    return Logistic(x)*(1-Logistic(x));


class NeuralNetwork:
    inputLayerNodeNum = None;
    layerNodeList = None;
    activationFunc = None;
    activationFuncDeriv = None;
    __layers = None;
    __errLimit = None;

    def __init__(self, inputLayerNodeNum,
                 layerNodeList=[2,1],
                 activationFunc='logistic'):
        self.inputLayerNodeNum = inputLayerNodeNum;
        self.layerNodeList = layerNodeList;
        self.__layers = len(layerNodeList);
        if (activationFunc == "logistic"):
            self.activationFunc = Logistic;
            self.activationFuncDeriv = Logistic_deriv;
        else:
            self.activationFunc = tanh;
            self.activationFuncDeriv = tanh_deriv;

        self.weightMatList = [];
        self.biasMatList = [];
        # bulid up weight mat and bias mat
        for layerIndex in range(len(self.layerNodeList)):
            if(layerIndex == 0):# 1st layer
                # tempWeightMat = np.ones( [layerNodeList[layerIndex], inputLayerNodeNum] );
                size = (layerNodeList[layerIndex], inputLayerNodeNum)
                tempWeightMat = (np.random.randint(5,size=size))*1.0;

            else:
                # tempWeightMat = np.ones( [layerNodeList[layerIndex], layerNodeList[layerIndex-1]]);
                size = (layerNodeList[layerIndex], layerNodeList[layerIndex-1])
                tempWeightMat = (np.random.randint(5,size=size))*1.0;
            self.weightMatList.append(tempWeightMat);
            self.biasMatList.append((np.random.randint(5,size=(layerNodeList[layerIndex],1)))*1.0);







    def GetEachLayerOutput(self, features):
        # calculate each layer's output
        features = np.array(features)[:, np.newaxis];
        layerOutputList = [features];
        for layerIndex in range(len(self.layerNodeList)):
            if (layerIndex == 0):  # 1st layer)
                layerRes = \
                    self.activationFunc(
                        np.dot(self.weightMatList[layerIndex], features)
                        + self.biasMatList[layerIndex]
                    );
            else:
                layerRes = \
                    self.activationFunc(
                        np.dot(self.weightMatList[layerIndex], layerOutputList[layerIndex])
                        + self.biasMatList[layerIndex]
                    );
            layerOutputList.append(layerRes);
        return layerOutputList;





    def fit(self, trainSet, labelSet, learingRate, targetMinErrorPercent):
        trainSet = np.array(trainSet);
        labelSet = np.array(labelSet);
        self.__errLimit = targetMinErrorPercent;
        fitting=True###########
        while(fitting):###################
            # calculate each layer's output
            for sampleInedex, eachSample in enumerate(trainSet):
                layerOutputList = self.GetEachLayerOutput(eachSample);
                errList = labelSet[sampleInedex][:,np.newaxis] - layerOutputList[-1];
                print("Output layer err percent List:\n", np.abs(errList*100.0/1) );
                # print("layer Output:\n", layerOutputList[-1]);
                if( max(np.abs(errList*100.0/1)) <= targetMinErrorPercent ):
                    # see if error is small enough
                    print("fit done\n");
                    print("Output layer err percent List:\n", np.abs(errList*100.0/1) );
                    print("layer Output:\n", layerOutputList[-1]);
                    fitting = False;#############
                    break;
                else:
                    deltaWeightList = [];
                    deltaBiasList = [];
                    # backward propagation
                    for outLayIdx in range(len(layerOutputList)-1,0,-1):
                        if(outLayIdx == len(layerOutputList)-1): #output layer
                            derivOutList = errList * self.activationFuncDeriv(layerOutputList[-1]);
                            deltaWeight = learingRate*derivOutList*layerOutputList[-1];
                            deltaBias = learingRate * derivOutList;

                        else:
                            errList = self.weightMatList[outLayIdx].T.dot(derivOutList)
                            derivOutList = errList * self.activationFuncDeriv(layerOutputList[outLayIdx]);
                            deltaWeight = learingRate * derivOutList * layerOutputList[outLayIdx];
                            deltaBias = learingRate * derivOutList;
                        deltaWeightList.insert(0,deltaWeight);
                        deltaBiasList.insert(0,deltaBias);

                    for layIdx in range(len(self.layerNodeList)):
                        self.weightMatList[layIdx] += deltaWeightList[layIdx];
                        self.biasMatList[layIdx] += deltaBiasList[layIdx];
            # print("weightMatList",self.weightMatList);
            # print("biasMatList", self.biasMatList);
            # print("deltaWeightList",deltaWeightList)



    def predict(self,trainSample):
        layerOutputList = self.GetEachLayerOutput(trainSample);
        outputList = np.where( layerOutputList[-1] >= (1-self.__errLimit/100.0),
                     1,
                     0);
        return outputList;









net = NeuralNetwork(3,[4,4,4,2],activationFunc='tanh');

x = np.array([[4,5,6],
              [5,6,7],
              [7,6,4],
              [20,21,23],
              [24,22,28],
              [29,30,25]]);
# due to Logistic function, output can only be 0 or 1
y = [[1,0],[1,0],[1,0],[1,1],[1,1],[1,1]];


net.fit(x,y,0.4,0.1);


res = net.predict([7,6,4]);
print("res:\n",res);
