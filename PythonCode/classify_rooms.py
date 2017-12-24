import numpy as numpy
import matplotlib.pyplot as pyplot
from mpl_toolkits.mplot3d import Axes3D


def readDataSet(filename):
 
    fr = open(filename)                
 
    numberOfLines = len(fr.readlines())
 
    returnMat = numpy.zeros((numberOfLines-1,3))
 
    classLabelVector = []
    classColorVector = []
    
    fr = open(filename)
    index = 0
    
    for line in fr.readlines(): 
        if index != 0:          
            line = line.strip()
            listFromLine = line.split('\t')
 
            returnMat[index-1,:] = listFromLine[1:4]
            
            classLabel = listFromLine[4] 
            
            if classLabel == "Buero":
                color = 'yellow'
            elif classLabel == "Wohnung":
                color = 'red'
            else:
                color = 'blue'
                        
            classLabelVector.append(classLabel)
            classColorVector.append(color)     
        
        index += 1

    return returnMat, classLabelVector, classColorVector



dataSet, classLabelVector, classColorVector = readDataSet("D:\REPOS\MachineLearningPython\PythonCode\\rooms_dataset.txt")



def normalizeDataSet(dataSet):
    
    dataSet_n = numpy.zeros(numpy.shape(dataSet))    
                                                     
    minValues = dataSet.min(0)                       
    ranges = dataSet.max(0) - dataSet.min(0)         
    
    minValues = dataSet.min(0)                       
    maxValues = dataSet.max(0)                       
 
    ranges = maxValues - minValues                   
 
    rowCount = dataSet.shape[0]                      
    
   
 
    dataSet_n = dataSet - numpy.tile(minValues, (rowCount, 1)) 
                                                   
 
    dataSet_n = dataSet_n / numpy.tile(ranges, (rowCount, 1))  

    return dataSet_n, ranges, minValues
 
dataSet_n, ranges, minValues = normalizeDataSet(dataSet)


def classify(inX, dataSet, labels, k):
 
    rowCount = dataSet.shape[0]             
 
    diffMat = numpy.tile(inX, (rowCount,1)) - dataSet
     
    sqDiffMat = diffMat**2                  
    sqDistances = sqDiffMat.sum(axis=1)     
    distances = sqDistances**0.5            
    sortedDistIndicies = distances.argsort()
    
    classCount = {}

    for i in range(k):                                       
        closest = labels[sortedDistIndicies[i]]              
        classCount[closest] = classCount.get(closest, 0) + 1 
    
    sortedClassCount = sorted(classCount, key = classCount.get, reverse=True)
    
    return sortedClassCount[0]  
                                
errorCount = 0

k = 10    # Test with different K's                        

rowCount = dataSet_n.shape[0]    

numTestVectors = 30              
                               

for i in range(0, numTestVectors):

    result = classify(dataSet_n[i,:], dataSet_n[numTestVectors:rowCount,:], classLabelVector[numTestVectors:rowCount], k)

    print("%s - the classifier came back with: %s, the real answer is: %s" %(i, result, classLabelVector[i]))

    if (result != classLabelVector[i]):
        errorCount += 1.0

print("Error Count: %d" % errorCount)
