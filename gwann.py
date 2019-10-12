import math
import numpy as np
import pandas as pd
import keras
import tensorflow as tf
import gwpy
from numpy.random import random
from gwpy.timeseries import TimeSeries
from gwpy.frequencyseries import FrequencySeries

def main():
    data = pd.read_csv('trainingset_v1d1_metadata.csv')
    features = [ # Selects the features that will go into the model
    "duration",
    "peak_frequency",
    "central_freq",
    "bandwidth",
    "amplitude",
    "snr"] # snr = signalToNoiseRation

    filtered = np.array(data[features]) # gets only the selected features for the datapoints
    labels = np.array(data["label"]) # gets glitch classification for validation purposes
    labelConvertionDict = createLabelConvertionDict(labels) # dict to convert strLabels to int representations for the model
    numLabels = convertStringLabelsToNumeric(labels, labelConvertionDict) # labels are now converted to numerical values

    dataSize = len(data) # number of datapoints in the file
    trainSize = math.ceil(0.8 * dataSize) # first 80% elements of dataset become training. Remaining 20% is used for validation
    inputSize = len(features) # number of features going into the model

    trainingSet = filtered[:trainSize]
    trainingLabels = numLabels[:trainSize]
    validationSet = filtered[trainSize:]
    validationLabels = numLabels[trainSize:]

    model = keras.Sequential() # creates classifier
    model.add(keras.layers.Dense(50, input_dim=inputSize, activation='relu'))
    model.add(keras.layers.Dense(50, activation='relu'))
    model.add(keras.layers.Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    history = model.fit(trainingSet, trainingLabels, epochs=2048, batch_size=32)

    preds = model.predict(validationSet) # uses features from 20% of the original set to try to predict the glitch type
    score = score(preds, validationLabels, labelConvertionDict) # compares the predicted glitch to the actual glitch to check accuracy

def score(predictions, validation, labelConvertionDict):
    correct = 0
    count = 0
    # predictions and validation are floats and must be converted to the glitch string label
    predictions = convertNumericLabelsToString(predictions.round(0), labelConvertionDict)
    validation = convertNumericLabelsToString(validation, labelConvertionDict)
    for i in range(len(predictions)):
        #print(predictions[i] + ' <- pred. | actual -> ' + validation[i])
        if predictions[i] == validation[i]:
            correct += 1
        count += 1
    print("Correct: " + str(correct))
    print("Total: " + str(count))
    print("Accuracy: " + str(round((100.0 * correct) / count, 2)) + "%")
    return 100.0 * correct / count

def createLabelConvertionDict(strLabels): # Creates a dict to convert between the string labels to numeric values
    distinctLabels = [elem for elem in strLabels] # deep copy
    labelConvertionDict = dict()
    distinctLabels = list(set(distinctLabels)) # gets just the distinct labels
    numVal = 0
    for label in distinctLabels:
        labelConvertionDict[label] = numVal # assigns each glitch name to a distinct num value
        numVal += 1
    return labelConvertionDict

def convertStringLabelsToNumeric(strLabels, labelConvertionDict): # Converts the labels set to their num values from the dict
    numLabels = [elem for elem in strLabels] # deep copy
    for i in range(len(numLabels)):
        numLabels[i] = labelConvertionDict[numLabels[i]] # replace string label value for corresponding the numValue from dict
    return numLabels

def convertNumericLabelsToString(intLabels, labelConvertionDict):
    converted = [i for i in intLabels] # deep copy
    converted = [i % len(labelConvertionDict) for i in converted] # Prevents prediction to be outside range of num values. Basically takes a random guess if it happens
    strValues = list(labelConvertionDict.keys())
    intToStrDict = dict()
    key = 0
    for val in strValues: # values will be in the same order as the labelConvertionDict so this is a safe operation
        intToStrDict[key] = val
        key += 1
    for i in range(len(converted)):
        converted[i] = str(intToStrDict[int(converted[i])]) # puts in the converted list the stringVal corresponding to the intLabel
    return converted
