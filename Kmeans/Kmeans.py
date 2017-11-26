import numpy as np;
import random;
import math;
import operator;


class Kmeans():
    Xset = None;
    Yset = None;
    num_clusters = None;

    def __init__(self, Xset, num_clusters, showInitCenters=True):
        self.Xset = np.array(Xset);
        self.num_clusters = num_clusters;
        random.seed();
        # set initial centroids for each cluster
        self.centroids = [];
        IdxList = [];
        while(True):
            newIdx = random.randint(0, len(self.Xset) - 1);
            IdxList.append(newIdx);
            IdxList = list(set(IdxList));
            if(len(IdxList)==num_clusters):
                break;
        for i in range(self.num_clusters):
            self.centroids.append(Xset[IdxList[i]]);
        if(showInitCenters):
            print("initial centroids:",self.centroids);

    # return final clustering centroids and cluster label
    def fit(self, showFinalResult=True):
        def cluster():
            typeList = [];
            for eachX in self.Xset:
                distDict = {};
                for eachType, eachCenter in enumerate(self.centroids):
                    distDict[str(eachType)] = math.sqrt(np.sum(np.power(eachX - eachCenter,2)));
                sortedDict = sorted(distDict.items(), key=operator.itemgetter(1));
                typeList.append(int(sortedDict[0][0]));
            return typeList;


        def updateCentroids(typeList):
            self.newCentroids = [0] * self.num_clusters;
            for enumType in range(self.num_clusters):
                # newCenter = np.zeros([1,self.Xset.shape[1]]);
                # # print("newCenter",newCenter);
                ptList = [];
                for idx, eachX in enumerate(self.Xset):
                    Xtype = typeList[idx];
                    if (Xtype == enumType):
                        ptList.append(np.array(eachX));

                self.newCentroids[enumType] = np.mean(ptList, axis=0);
            # print("update centroids:", self.newCentroids);
            return self.newCentroids;


        def run():
            return not np.array_equal(self.centroids, self.newCentroids);


        typeList = cluster();
        updateCentroids(typeList);
        while( run() ):
            self.centroids = self.newCentroids;
            typeList = cluster();
            updateCentroids(typeList);
        self.centroids = self.newCentroids;
        clusterLabel = [i for i in range(0,len(self.centroids))];
        if(showFinalResult):
            print("final centroids:",self.centroids, "\ncluster labels:",clusterLabel);
        return self.centroids, clusterLabel;


    def predict(self,Xset):
        typeList = [];
        for eachX in Xset:
            distDict = {};
            for eachType, eachCenter in enumerate(self.centroids):
                distDict[str(eachType)] = math.sqrt(np.sum(np.power(eachX - eachCenter, 2)));
            sortedDict = sorted(distDict.items(), key=operator.itemgetter(1));
            typeList.append(int(sortedDict[0][0]));
        return typeList;














X = np.array([ [0,0,1],
               [0,0,2],
               [0,0,3],

               [10,10,11],
               [10,10,12],
               [10,10,13],

               [20,20,21],
               [20,20,22],
               [20,20,23]
             ]);

kmeans = Kmeans(X,3);
kmeans.fit();
resCluster = kmeans.predict([ [0,0,0],
                              [1,0,1]
                            ]);

print(resCluster);
