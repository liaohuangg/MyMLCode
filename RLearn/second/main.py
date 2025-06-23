import sys
import os
import pickle
import numpy as np
import scipy.io as sio
import matplotlib
import matplotlib.pyplot as plt
from libsvm.svmutil import *
from numpy import *

filename = 'krkopt.data'
'''
the data format of 'krkopt.data' like
    a,1,b,3,c,2,draw
    b,1,a,4,h,4,fourteen
The first six entries represent the positions of the chess pieces. 
"Draw" indicates a forced stalemate. 
The number following it, such as "six," signifies that White can 
checkmate Black in a minimum of six moves.
'''
fr = open(filename)
arrayOLines = fr.readlines()
del arrayOLines[0] #remove the title of this file
numberOfLines = len(arrayOLines) #get the length of this list
numberOfFeatureDimension = 6

# prepare matrix to return
data = zeros((numberOfLines, numberOfFeatureDimension))
label = zeros(numberOfLines)

# process the input date as matrix
for index in range(len(arrayOLines)):
    line = arrayOLines[index]
    listFromLine = line.split(',') # split str according to ','
    data[index, 0] = ord(listFromLine[0])-96 # transf as ASCII
    data[index, 1] = ord(listFromLine[1]) - 48
    data[index, 2] = ord(listFromLine[2])-96
    data[index, 3] = ord(listFromLine[3]) - 48
    data[index, 4] = ord(listFromLine[4])-96
    data[index, 5] = ord(listFromLine[5]) - 48
    if listFromLine[6] == 'draw\n':
        label[index] = 1
    else:
        label[index] = -1

# construct the train data set
permutatedData = zeros((numberOfLines, numberOfFeatureDimension))
permutatedLabel = zeros(numberOfLines)
# randomly generate an index list
p = random.permutation(numberOfLines)
for i in range(numberOfLines):
    permutatedData[i,:] = data[p[i],:]
    permutatedLabel[i] = label[p[i]]
# partition train set
numberOfTrainingData = 5000
xTrain = permutatedData[:numberOfTrainingData]
yTrain = permutatedLabel[:numberOfTrainingData]
xTest =  permutatedData[numberOfTrainingData:]
yTest = permutatedLabel[numberOfTrainingData:]

# subtract mean and divide by standard deviation
# get average data
averageData = zeros((1, numberOfFeatureDimension))
for i in range(len(xTrain)):
    averageData += xTrain[i,:]
averageData = averageData/len(xTrain)

# get standard deviation
standardDeviation = zeros((1,numberOfFeatureDimension))
for i in range(len(xTrain)):
    standardDeviation+=(xTrain[i]-averageData[0,:])**2
standardDeviation = (standardDeviation/(len(xTrain)-1))**0.5

# pre-process train data set and test data set
for i in range(len(xTrain)):
    xTrain[i] = (xTrain[i] -averageData)/standardDeviation

for i in range(len(xTest)):
    xTest[i] = (xTest[i] -averageData)/standardDeviation
    
# using the Gaussian kernel, perform a brute-force search for two hyperparameters
# CScale = [-5, -3, -1, 1, 3, 5, 7, 9, 11, 13, 15]
# gammaScale = [-15, -13, -11, -9, -7, -5, -3, -1 , 1, 3]
CScale = [-5, 15] # for fast
gammaScale = [-15, 3]
maxRecognitionRate = 0
'''
How to define the best hyperparameters
1. Using cross-validation, divide 5,000 samples into 5 groups, 
each containing 1,000 samples. (called as 5-fold cross-validation)
Train with groups A, B, C, and D, and test with group E. 
Train with groups A, B, C, and E, and test with group D, etc. (ensuring that training samples do not participate in testing).

2. It is necessary to make the most use of the training samples.

For each combination of C and gamma, perform 5-fold cross-validation 
to obtain an accuracy rate, and select the hyperparameters with the 
highest accuracy rate.
'''
# transfor the numpy array as list format
arr = np.array(xTrain)
newX = arr.tolist()
arr = np.array(yTrain)
newY = arr.tolist()
# For each combination of C and gamma
for i in range(len(CScale)):
    testC = 2 ** CScale[i]
    for j in range(len(gammaScale)):
        # using RBF kernel, -t
        cmd = '-t 2 -c '
        # define hyperparameter C, -c
        cmd += str(testC)
        cmd += ' -g '
        # define hyperparameter gamma, -g
        testGamma = 2**gammaScale[j]
        cmd += str(testGamma)
        # using 5-fold, -v
        cmd += ' -v 5'
        # not need help information, -h
        cmd +=' -h 0'
        recognitionRate = svm_train(newY, newX, cmd)
        if recognitionRate > maxRecognitionRate:
            maxRecognitionRate = recognitionRate
            print(maxRecognitionRate)
            maxCIndex = i
            maxGammaIndex = j
            
#Search for good hyper parameters. Second, refined search.
# n = 10
n = 2 # for fast
print("\nC\n", CScale[maxCIndex])
print("\ngamma\n", gammaScale[maxGammaIndex])
print("\n")
minCScale = CScale[maxCIndex] - 0.2
maxCScale = CScale[maxCIndex] + 0.2
newCScale = arange(minCScale, maxCScale+0.001,(maxCScale-minCScale)/n)
minGammaScale = gammaScale[maxGammaIndex] - 0.2
maxGammaScale = gammaScale[maxGammaIndex] + 0.2
newGammaScale = arange(minGammaScale,maxGammaScale+0.001,(maxGammaScale-minGammaScale)/n)

maxRecognitionRate = 0
for testCScale in newCScale:
    testC = 2 ** testCScale
    for testGammaScale in newGammaScale:
        testGamma = 2**testGammaScale
        cmd = '-t 2 -c '
        cmd += str(testC)
        cmd += ' -g '
        cmd += str(testGamma)
        cmd += ' -v 5'
        cmd +=' -h 0'
        recognitionRate = svm_train(newY,newX, cmd)
        if recognitionRate > maxRecognitionRate:
            maxRecognitionRate = recognitionRate
            maxC = testC
            maxGamma = testGamma
            
#Input all training data to train again.
cmd = '-t 2 -c '
cmd += str(maxC)
cmd += ' -g '
cmd += str(maxGamma)
cmd += ' -h 0'
model = svm_train(newY,newX,cmd)

#Test
arr = np.array(xTest)
newX = arr.tolist()
arr = np.array(yTest)
newY = arr.tolist()
yPred, accuracy, decisionValues = svm_predict(newY, newX, model)
sio.savemat('yTest.mat', {'yTest': yTest})
sio.savemat('decisionValues.mat', {'decisionValues': decisionValues})

#drawROC
'''
True Positives (TP):
True Positives refer to the cases where the model correctly predicts a positive class sample as positive.

False Positives (FP):
False Positives refer to the cases where the model incorrectly predicts a negative class sample as positive.
'''
totalScores = sorted(decisionValues)
index = sorted(range(len(decisionValues)), key=decisionValues.__getitem__)
labels = zeros(len(yTest))
for i in range(len(labels)):
    labels[i] = yTest[index[i]]

truePositive = zeros(len(labels) + 1)
falsePositive = zeros(len(labels) + 1)
for i in range(len(totalScores)):
    if labels[i] > 0.5:
        truePositive[0] += 1
    else:
        falsePositive[0] += 1

for i in range(len(totalScores)):
    if labels[i] > 0.5:
        truePositive[i + 1] = truePositive[i] - 1
        falsePositive[i + 1] = falsePositive[i]
    else:
        falsePositive[i + 1] = falsePositive[i] - 1
        truePositive[i + 1] = truePositive[i]

truePositive = truePositive / truePositive[0]
falsePositive = falsePositive / falsePositive[0]
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(falsePositive, truePositive)
# plt.show()
plt.savefig('roc_curve.png')