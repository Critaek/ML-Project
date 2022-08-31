import numpy
import Normalization as n

def saveRawFolds(D, L, K):
    numpy.random.seed(0)
    idx = numpy.random.permutation(D.shape[1])

    N = D.shape[1]
    M = round(N/K)

    folds = []

    for i in range(K): #K=3 -> 0,1,2
        idxTrain = numpy.concatenate([idx[0:i*M], idx[(i+1)*M:N]])
        idxTest = idx[i*M:(i+1)*M]
        DTR = D[:, idxTrain]
        LTR = L[idxTrain]
        DTE = D[:, idxTest]
        LTE = L[idxTest]
        obj = (DTR, LTR, DTE, LTE)
        folds.append(obj)

    folds = numpy.array(folds, dtype=object)
    numpy.save("data/raw.npy", folds)

def saveNormFolds(D, L, K):
    numpy.random.seed(0)
    idx = numpy.random.permutation(D.shape[1])

    N = D.shape[1]
    M = round(N/K)

    folds = []

    for i in range(K): #K=3 -> 0,1,2
        idxTrain = numpy.concatenate([idx[0:i*M], idx[(i+1)*M:N]])
        idxTest = idx[i*M:(i+1)*M]
        DTR = D[:, idxTrain]
        LTR = L[idxTrain]
        DTE = D[:, idxTest]
        LTE = L[idxTest]
        DTR = n.NormDTE(DTR, DTR)
        DTE = n.NormDTE(DTR, DTE)
        obj = (DTR, LTR, DTE, LTE)
        folds.append(obj)

    folds = numpy.array(folds, dtype=object) #Questi oggetti hanno forma (DTR, LTR, DTE, LTE)
    #numpy.save("data/norm.npy", folds)
    return folds

def loadRawFolds():
    raw = numpy.load("data/raw.npy", allow_pickle=True)

    return raw

def loadNormFolds():
    raw = numpy.load("data/norm.npy", allow_pickle=True)

    return raw




