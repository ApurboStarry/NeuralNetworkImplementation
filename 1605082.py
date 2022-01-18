import pickle
import numpy as np
import pandas as pd


def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


def getNumberOfFeatures(file_name):
    dataFrame = pd.read_csv(file_name, delim_whitespace=True, header=None)
    return dataFrame.shape[1] - 1


def getNumberOfClasses(file_name):
    dataFrame = pd.read_csv(file_name, delim_whitespace=True, header=None)
    return len(set(dataFrame[dataFrame.columns[-1]].values))


def getFeatureVector(fileName):
    dataFrame = pd.read_csv(fileName, delim_whitespace=True, header=None)
    featureVector = dataFrame[dataFrame.columns[:-1]].values
    featureVector = (featureVector - featureVector.mean(axis=0)
                     ) / featureVector.std(axis=0)

    return featureVector


def getOHEClassVector(fileName):
    dataFrame = pd.read_csv(fileName, delim_whitespace=True, header=None)
    classValues = dataFrame[dataFrame.columns[-1]].values

    from sklearn.preprocessing import OneHotEncoder
    encoder = OneHotEncoder(sparse=False)
    classValues = encoder.fit_transform(classValues.reshape(-1, 1))
    return classValues


def getClassValues(fileName):
    dataFrame = pd.read_csv(fileName, delim_whitespace=True, header=None)
    classValues = dataFrame[dataFrame.columns[-1]].values
    return classValues


class Layer:
    def __init__(self, numberOfNeurons, numberOfFeatures, activationFunction):
        self.numberOfNeurons = numberOfNeurons
        self.numberOfFeatures = numberOfFeatures
        self.activationFunction = activationFunction.lower()

        self.wT = np.random.randn(numberOfNeurons, numberOfFeatures)
        self.b = np.random.randn(numberOfNeurons, 1)

        self.x = None
        self.z = None
        self.a = None

    def getActivationOutput(self):
        if self.activationFunction == "sigmoid":
            return 1.0 / (1.0 + np.exp(-self.z))
        elif self.activationFunction == "relu":
            return np.where(self.z <= 0, 0, self.z)
        elif self.activationFunction == "tanh":
            return np.tanh(self.z)

    def getActivationDerivative(self):
        if self.activationFunction == "sigmoid":
            sigma = self.getActivationOutput()
            return sigma * (1 - sigma)
        elif self.activationFunction == "relu":
            return np.where(self.z <= 0, 0, 1)
        elif self.activationFunction == "tanh":
            tanh = np.tanh(self.z)
            return (1 - tanh ** 2)


class NeuralNetwork:

    def __init__(self):
        self.numberOfFeatures = None
        self.numberOfClasses = None

        self.layers = []
        self.deltas = []

    def createInputLayer(self, numberOfFeatures):
        self.numberOfFeatures = numberOfFeatures

    def createHiddenLayers(self, hiddenLayers):
        # the first hidden layer
        self.layers.append(
            Layer(hiddenLayers[0][0], self.numberOfFeatures, hiddenLayers[0][1]))

        # rest of the hidden layers
        for i in range(1, len(hiddenLayers)):
            self.layers.append(
                Layer(hiddenLayers[i][0], self.layers[-1].numberOfNeurons, hiddenLayers[i][1]))

    def createOutputLayer(self, numberOfClasses, activationFunction):
        self.numberOfClasses = numberOfClasses
        self.layers.append(Layer(self.numberOfClasses,
                           self.layers[-1].numberOfNeurons, activationFunction))

    def forwardProp(self, featureVector):
        previousLayerOutput = featureVector.copy().T

        for layer in self.layers:
            layer.x = previousLayerOutput.copy()
            layer.z = np.matmul(layer.wT, layer.x) + layer.b
            layer.a = layer.getActivationOutput()

            previousLayerOutput = layer.a

        return previousLayerOutput

    def getCost(self):
        return np.sum((self.layers[-1].a - self.classVector.T) ** 2) / 2.0

    def backProp(self):
        numberOfLayers = len(self.layers)
        self.deltas = [0] * numberOfLayers
        self.deltas[-1] = (self.layers[-1].a - self.classVector.T) * \
            self.layers[-1].getActivationDerivative()

        for i in range(numberOfLayers - 2, -1, -1):
            self.deltas[i] = np.matmul(
                self.layers[i+1].wT.T, self.deltas[i+1]) * self.layers[i].getActivationDerivative()

    def updateWeightsAndBiases(self):
        numberOfLayers = len(self.layers)
        for i in range(numberOfLayers):
            previousLayerOutput = None
            if i == 0:
                previousLayerOutput = self.featureVector.T
            else:
                previousLayerOutput = self.layers[i-1].a

            dw = -self.learningRate * \
                np.matmul(self.deltas[i], previousLayerOutput.T)
            self.layers[i].wT += dw

            db = -self.learningRate * \
                np.sum(self.deltas[i], axis=1, keepdims=True)
            self.layers[i].b += db

    def saveNetworkStructureAndParameters(self):
        save_object(self.layers, 'networkStructureAndParameters.txt')
        featuresAndClasses = np.array(
            [self.numberOfFeatures, self.numberOfClasses])
        np.savetxt("numberOfFeaturesAndClasses", featuresAndClasses)

    # def saveNetworkStructureAndParameters(self):
    #     fileName = "networkStructureAndParameters.txt"
    #     file = open(fileName, "w+")

    #     # # save input layer(number of features)
    #     # file.write(str(self.numberOfFeatures))
    #     # file.write("\n")

    #     # save hidden layers and output layer along with their weights and biases
    #     hiddenLayers = []
    #     for layer in self.layers:
    #         hiddenLayers.append((layer.numberOfNeurons, layer.activationFunction, layer.wT, layer.b))
    #     file.write(str(hiddenLayers))

    def trainNeuralNetwork(self, featureVector, classVector, learningRate, maxIterations, stoppingCriteriaValue):
        self.featureVector = featureVector
        self.classVector = classVector

        previousCost = -1
        self.learningRate = learningRate

        for i in range(maxIterations):
            self.forwardProp(self.featureVector)
            currentCost = self.getCost()

            if i == 0 or (currentCost < previousCost and previousCost - currentCost > stoppingCriteriaValue):
                previousCost = currentCost
            else:
                break

            self.backProp()
            self.updateWeightsAndBiases()

        self.saveNetworkStructureAndParameters()

    def loadNetworkStructureAndParameters(self):
        fileName = "networkStructureAndParameters.txt"
        with open(fileName, 'rb') as inp:
            layers = pickle.load(inp)
            self.layers = layers

        numberOfFeaturesAndClasses = np.loadtxt("numberOfFeaturesAndClasses")
        numberOfF = int(numberOfFeaturesAndClasses[0])
        numberOfC = int(numberOfFeaturesAndClasses[1])
        self.numberOfFeatures = numberOfF
        self.numberOfClasses = numberOfC

    def testNeuralNetwork(self, testFeatureVector, testClassValues):
        self.loadNetworkStructureAndParameters()
        lastLayerOutput = self.forwardProp(testFeatureVector)

        predictedClasses = []
        for row in lastLayerOutput.T:
            predictedClasses.append(np.argmax(row) + 1)

        predictedClasses = np.asarray(predictedClasses)

        numberOfMatches = 0
        for i in range(len(testClassValues)):
            if predictedClasses[i] == testClassValues[i]:
                numberOfMatches += 1
            else:
                print(testFeatureVector[i])

        accuracy = numberOfMatches / len(testFeatureVector) * 100
        return accuracy


def run_offline():
    trainFileName = "trainNN.txt"
    testFileName = "testNN.txt"

    numberOfFeatures = getNumberOfFeatures(trainFileName)
    numberOfClasses = getNumberOfClasses(trainFileName)
    featureVector = getFeatureVector(trainFileName)
    classVector = getOHEClassVector(trainFileName)

    nn = NeuralNetwork()

    nn.createInputLayer(numberOfFeatures)
    nn.createHiddenLayers([
        (3, "sigmoid"),
        (5, "sigmoid"),
        (4, "sigmoid"),
        (10, "sigmoid"),
        (7, "sigmoid")
    ])
    nn.createOutputLayer(numberOfClasses, "sigmoid")

    nn.trainNeuralNetwork(featureVector, classVector, 0.001, 10000, 1e-20)

    # for testing
    testFeatureVector = getFeatureVector(testFileName)
    testClassValues = getClassValues(testFileName)

    nnTest = NeuralNetwork()
    accuracy = nnTest.testNeuralNetwork(testFeatureVector, testClassValues)
    print(accuracy)


run_offline()
