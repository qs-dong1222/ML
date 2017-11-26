import numpy as np
import random
import math
import operator
import matplotlib.pyplot as plt;


class GeneticOptFinder():
    def __init__(self, chromosomeBitLength, chroNbrInPop, mutaProb=0.1,decodeFunc=None,fitnessFunc=None,findMaxOpt=True):
        self.bitLength = chromosomeBitLength;
        self.chroNbrInPop = chroNbrInPop;
        self.mutaProb = mutaProb;
        self.decodeFunc = decodeFunc;
        self.fitnessFunc = fitnessFunc;
        self.findMaxOpt = findMaxOpt;
        self.population = self.__GenPopulation();




    def Evolve(self,evolution_times):
        for i in range(evolution_times):
            self.PoplulationFit();
            self.Crossover();
            self.child_1, self.child_2 = self.Mutation(self.child_1, self.child_2, self.mutaProb);
            self.Obsolute();
        return self.population[0]






    def PoplulationFit(self):
        self.fitValDict = {}
        for eachChrom in self.population:
            decodedVal = self.decodeFunc(eachChrom);
            fitVal = self.fitnessFunc(decodedVal);
            self.fitValDict[str(eachChrom)] = fitVal

        if(self.findMaxOpt):
            self.fitValDict = sorted(self.fitValDict.items(), key=operator.itemgetter(1),reverse=True);
        else:
            self.fitValDict = sorted(self.fitValDict.items(), key=operator.itemgetter(1), reverse=False);
        # print(self.fitValDict)
        self.population = list(map(lambda x: int(x[0]), self.fitValDict))
        self.fitValList = list(map(lambda x: x[1], self.fitValDict))
        # print("population", self.population)
        # print("fitValList", self.fitValList)


    def Select(self):
        populationBkup = self.population.copy();
        # regularize and mult by 10 to enhance the difference on exp
        fitValList_norm = self.fitValList / np.sum(self.fitValList) * 10
        if(self.findMaxOpt):
            softmaxProbs = self.__softMax(fitValList_norm);
        else:
            softmaxProbs = self.__softMax(fitValList_norm);
            softmaxProbs = [1-x for x in softmaxProbs]
        # prepare probility axis
        lowerBound = 0;
        self.selProbs = [lowerBound];
        for idx in range(len(softmaxProbs)):
            upperBound = lowerBound + softmaxProbs[idx];
            self.selProbs.append(upperBound);
            lowerBound = upperBound;
        if (self.selProbs[-1] != 1):
            self.selProbs[-1] = 1;
        # print("prob axis", self.selProbs)
        # select 1st parent
        tmp1 = random.uniform(self.selProbs[0],self.selProbs[-1]);
        for idx in range(len(self.selProbs)-1):
            lowerBound = self.selProbs[idx];
            upperBound = self.selProbs[idx+1];
            if(lowerBound <= tmp1 < upperBound):
                parent_1 = int(populationBkup[idx]);
                # print(str(lowerBound) + " < tmp1 < " + str(upperBound))
                # print("selected chromosome1", parent_1)
                self.selProbs[idx+1:] = [x-(upperBound-lowerBound) for x in self.selProbs[idx+1:]]
                self.selProbs = sorted(self.selProbs)
                self.selProbs.pop(-1)
                populationBkup.pop(idx)
                break;
        # print("prob axis", self.selProbs)
        # select 2nd parent
        tmp2 = random.uniform(self.selProbs[0], self.selProbs[-1]);
        for idx in range(len(self.selProbs)-1):
            lowerBound = self.selProbs[idx];
            upperBound = self.selProbs[idx+1];
            if(lowerBound <= tmp2 < upperBound):
                # print("populationBkup",populationBkup)
                # print("$$$$$$  idx",idx)
                parent_2 = int(populationBkup[idx]);
                # print(str(lowerBound) + " < tmp2 < " + str(upperBound))
                # print("selected chromosome2", parent_2)
                break;
        del populationBkup;
        return parent_1,parent_2



    def IsCrossoverble(self, parent_1, parent_2):
        diffMask = parent_1 ^ parent_2;
        if(len(self.__onesIndices(diffMask))<=1):
            return False
        else:
            return True;


    def SelectNewWeekParent(self, parent_1, parent_2):
        # select form the rest of elements different from parent_1,2
        tmpPop = self.population.copy();
        tmpPop.remove(parent_1);
        tmpPop.remove(parent_2);
        # if trying to find minimum, then replace the smaller parent
        if (self.findMaxOpt):
            if (parent_1 > parent_2):
                parent_2 = random.choice(tmpPop)
            else:
                parent_1 = random.choice(tmpPop)
        # vice versa
        else:
            if (parent_1 < parent_2):
                parent_2 = random.choice(tmpPop)
            else:
                parent_1 = random.choice(tmpPop)
        del tmpPop;
        return parent_1, parent_2




    def Crossover(self):
        """ Crossover: 用两个父代基因交配出两个子代基因 """
        times2forceMutation = 5;  # 如果尝试交配5次都出不来不同的后代，那么我们就强制父代进行变异，以寻求子代的改变
        count = times2forceMutation;

        # 查看两个父代基因是不是有交配的有效性,即是否能交配出不一样的后代
        # child will be the sams as parents if parents have one bit diffrence only,
        # select again the parents

        parent_1, parent_2 = self.Select();
        while(not self.IsCrossoverble(parent_1,parent_2)):
            if(count == 0):
                # 尝试多次交配，都没有效果，那么强制父代中的一个进行'临时性'变异
                parent_1, parent_2 = self.Mutation(parent_1, parent_2,1);
                while(not self.IsCrossoverble(parent_1, parent_2)):
                    parent_1, parent_2 = self.Mutation(parent_1, parent_2,1);
                break;
                count = times2forceMutation;
            else:
                count -= 1;
            parent_1, parent_2 = self.SelectNewWeekParent(parent_1, parent_2);

        # 进行交配，原则是：
        # 1.不能所有不同位上都互换，那等于没换
        # 2.不能压根就不互换，那也等于没换
        # 结论：n位不同，选 1 ~ n-1 个随机位进行互换
        diffMask = parent_1 ^ parent_2;
        exchTimes = random.choice(range(1,len(self.__onesIndices(diffMask))));
        posList = self.__onesIndices(diffMask);
        self.child_1 = parent_1;
        self.child_2 = parent_2;
        for eachTime in range(exchTimes):
            # select an index and exchange bit in between
            pos = random.choice(posList);
            self.child_1 = (1 << pos) ^ self.child_1;
            self.child_2 = (1 << pos) ^ self.child_2;




    def Mutation(self, chromoesome_1, chromosome_2, mutaProb):
        tmp = random.uniform(0,1);
        # 如果达到变异概率，那么就两个个体中选一个进行变异
        if(tmp < mutaProb):
            # 随机抽取变异 bit 的位置
            mutaBitPos = random.choice(range(self.bitLength));
            # 两个子代基因而选一进行变异
            if(random.choice([True,False])):
                # print("chromoesome_1 muta")
                chromoesome_1 = (1 << mutaBitPos) ^ chromoesome_1;
            else:
                # print("chromoesome_2 muta")
                chromosome_2 = (1 << mutaBitPos) ^ chromosome_2;
        return chromoesome_1, chromosome_2





    def Obsolute(self):
        """ Obsolute: 如果子代基因优秀, 用两个子代基因淘汰差等父代基因 """
        self.childFitVal_1 = self.fitnessFunc(self.decodeFunc(self.child_1));
        self.childFitVal_2 = self.fitnessFunc(self.decodeFunc(self.child_2));
        # print(self.child_1, self.childFitVal_1, self.child_2, self.childFitVal_2)

        if(self.child_1 not in self.population):
            for idx in range(len(self.fitValList)-1,-1,-1):
                if(self.findMaxOpt):
                    if(self.childFitVal_1>self.fitValList[idx]):
                        self.population[idx] = self.child_1;
                        self.fitValList[idx] = self.childFitVal_1;
                        break;
                else:
                    if (self.childFitVal_1 < self.fitValList[idx]):
                        self.population[idx] = self.child_1;
                        self.fitValList[idx] = self.childFitVal_1;
                        break;
        if (self.child_2 not in self.population):
            for idx in range(len(self.fitValList)-1,-1,-1):
                if (self.findMaxOpt):
                    if (self.childFitVal_2 > self.fitValList[idx]):
                        self.population[idx] = self.child_2;
                        self.fitValList[idx] = self.childFitVal_2;
                        break;
                else:
                    if (self.childFitVal_2 < self.fitValList[idx]):
                        self.population[idx] = self.child_2;
                        self.fitValList[idx] = self.childFitVal_2;
                        break;




    def __softMax(self,array):
        return np.exp(array) / np.sum(np.exp(array))


    def __GenChromosome(self):
        chromosome = 0;
        for i in range(self.bitLength):
            chromosome |= ((random.randint(0,1))<<i);
        return chromosome;



    def __GenPopulation(self):
        pop = [];
        # for i in range(self.chroNbrInPop):
        while(len(pop)<self.chroNbrInPop):
            pop.append(self.__GenChromosome());
            pop = list(set(pop))
        return pop;


    def __onesIndices(self, n):
        onesIdx = [];
        for i in range(0, self.bitLength):
            if (n & 1):
                onesIdx.append(i)
            n = n >> 1
        return onesIdx









chromosomeBitLength = 17;
chroNbrInPop = 18;

def fitFunc(x):
    return  x + 10*math.sin(5*x) + 7*math.cos(4*x);

def decodeFunc(chromosome):
    return chromosome * ((9-0) / (2**chromosomeBitLength-1))

ga = GeneticOptFinder(chromosomeBitLength
                      ,chroNbrInPop
                      ,decodeFunc=decodeFunc
                      ,fitnessFunc=fitFunc
                      ,findMaxOpt=False
                      ,mutaProb=0.1);
bestChromo = ga.Evolve(200)
deco = decodeFunc(bestChromo);

print("x=",deco,"y=",fitFunc(deco))





# li = [2,3,4,5,6]
# for i in range