import pickle

with open('networkStructureAndParameters.txt', 'rb') as inp:
    layers = pickle.load(inp)
    print(layers)
