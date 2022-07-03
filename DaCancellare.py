import sklearn.datasets
import numpy
import scipy.optimize
import time
import Load


def mcol(vect):
    return vect.reshape(vect.size, 1)

def vrow(vect):
    return vect.reshape(1, vect.size)

def numberOfCorrectLabreturn(Predictions, LTE):
    n = 0
    for i in range(0,Predictions.shape[0]):
        if Predictions[i] == LTE[i]:
            n += 1
            
    return n

def split_db_2to1(D, L, seed = 0):
    nTrain = int(D.shape[1]*2.0/3.0)
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    
    return (DTR, LTR), (DTE, LTE)

def trainSVMLinear(DTE, LTE, DTR, LTR, K, C):
    expandedD = numpy.vstack([DTR, K * numpy.ones(DTR.shape[1])])

    Z = numpy.zeros(LTR.shape)
    Z[LTR == 1] = 1
    Z[LTR == 0] = -1
    
    H = numpy.dot(expandedD.T, expandedD)
    H = mcol(Z) * vrow(Z) * H

    def JDual(alpha):
        Ha = numpy.dot(H, mcol(alpha))
        aHa = numpy.dot(vrow(alpha), Ha)
        a1 = alpha.sum()

        return -0.5 * aHa.ravel() + a1, -Ha.ravel() + numpy.ones(alpha.size)

    def LDual(alpha):
        loss, grad = JDual(alpha)
        return -loss, -grad

    def JPrimal(w):
        S = numpy.dot(vrow(w), expandedD)
        loss = numpy.maximum(numpy.zeros(S.shape), 1 - Z*S).sum()
        return 0.5 * numpy.linalg.norm(w)**2 + C * loss

    alphaStar, x, y = scipy.optimize.fmin_l_bfgs_b(LDual,
                                                   numpy.zeros(DTR.shape[1]),
                                                   bounds=[(0, C)] * DTR.shape[1],
                                                   factr = 1.0
                                                  )
    
    wStar = numpy.dot(expandedD, mcol(alphaStar) * mcol(Z))

    primalLoss = JPrimal(wStar)
    dualLoss = JDual(alphaStar)[0]
 
    print("Primal Loss", primalLoss)
    print("Dual Loss", dualLoss)
    print("Duality Gap", primalLoss - dualLoss)

    expandedDTE = numpy.vstack([DTE, K * numpy.ones(DTE.shape[1])])
    score = numpy.dot(wStar.T, expandedDTE)
    Predictions = score > 0

    #print(Predictions[0])

    n = numberOfCorrectLabreturn(Predictions[0], LTE)
    acc = n / LTE.shape[0]
    err = 1 - acc
    print("err -", err*100, "%")




def trainSVMPoly(DTE, LTE, DTR, LTR, K, C, d, c):
    Z = numpy.zeros(LTR.shape)
    Z[LTR == 1] = 1
    Z[LTR == 0] = -1
    
    epsilon = K**2

    product = numpy.dot(DTR.T, DTR)
    Kernel = (product + c)**d + epsilon
    H = mcol(Z) * vrow(Z) * Kernel

    def JDual(alpha):
        Ha = numpy.dot(H, mcol(alpha))
        aHa = numpy.dot(vrow(alpha), Ha)
        a1 = alpha.sum()

        return -0.5 * aHa.ravel() + a1, -Ha.ravel() + numpy.ones(alpha.size)

    def LDual(alpha):
        loss, grad = JDual(alpha)
        return -loss, -grad

    alphaStar, x, y = scipy.optimize.fmin_l_bfgs_b(LDual,
                                                   numpy.zeros(DTR.shape[1]),
                                                   bounds=[(0, C)] * DTR.shape[1],
                                                   factr = 1.0,
                                                   maxiter=100000,
                                                   maxfun=100000
                                                  )

    dualLoss = JDual(alphaStar)[0]
 
    print("Dual Loss", dualLoss)
    
    scores = []
    for x_t in DTE.T:
        score = 0
        for i in range(DTR.shape[1]):
            Kernel = (numpy.dot(DTR.T[i].T, x_t) + c)**d
            score += alphaStar[i] * Z[i] * Kernel
        scores.append(score)
    
    scores = numpy.vstack(scores)
     
    Predictions = scores > 0
    Predictions = numpy.hstack(Predictions)

    #print(Predictions[0])

    n = numberOfCorrectLabreturn(Predictions, LTE)
    acc = n / LTE.shape[0]
    err = 1 - acc
    print("err -", err*100, "%")




def trainSVM_RBF(DTE, LTE, DTR, LTR, K, C, gamma):
    Z = numpy.zeros(LTR.shape)
    Z[LTR == 1] = 1
    Z[LTR == 0] = -1
    
    epsilon = K**2

    Dist = numpy.zeros([DTR.shape[1], DTR.shape[1]])

    for i in range(DTR.shape[1]):
        for j in range(DTR.shape[1]):
            xi = DTR[:, i]
            xj = DTR[:, j]
            Dist[i, j] = numpy.linalg.norm(xi - xj)**2

    Kernel = numpy.exp(- gamma * Dist) + epsilon
    H = mcol(Z) * vrow(Z) * Kernel

    def JDual(alpha):
        Ha = numpy.dot(H, mcol(alpha))
        aHa = numpy.dot(vrow(alpha), Ha)
        a1 = alpha.sum()

        return -0.5 * aHa.ravel() + a1, -Ha.ravel() + numpy.ones(alpha.size)

    def LDual(alpha):
        loss, grad = JDual(alpha)
        return -loss, -grad

    alphaStar, x, y = scipy.optimize.fmin_l_bfgs_b(LDual,
                                                   numpy.zeros(DTR.shape[1]),
                                                   bounds=[(0, C)] * DTR.shape[1],
                                                   factr = 1.0,
                                                   maxiter=100000,
                                                   maxfun=100000
                                                  )

    dualLoss = JDual(alphaStar)[0]
 
    print("Dual Loss", dualLoss)
    
    scores = []
    for x_t in DTE.T:
        score = 0
        for i in range(DTR.shape[1]):
            Dist = numpy.linalg.norm(DTR[:, i] - x_t)
            Kernel = numpy.exp(- gamma * Dist) + epsilon
            score += alphaStar[i] * Z[i] * Kernel
        scores.append(score)
    
    scores = numpy.vstack(scores)
     
    Predictions = scores > 0
    Predictions = numpy.hstack(Predictions)

    #print(Predictions[0])

    n = numberOfCorrectLabreturn(Predictions, LTE)
    acc = n / LTE.shape[0]
    err = 1 - acc
    print("err -", err*100, "%")



def load_iris_binary():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    D = D[:, L != 0] # We remove setosa from D
    L = L[L!=0] # We remove setosa from L
    L[L==2] = 0 # We assign label 0 to virginica (was label 2)
    return D, L

if __name__ == "__main__":
    D, L = load_iris_binary()
    #D, L = Load.load()
    D 
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
    start = time.time()

    trainSVMLinear(DTE, LTE, DTR, LTR, K=0.0, C=0.1)
    print("\n")

    trainSVMPoly(DTE, LTE, DTR, LTR, K=0.0, C=1.0, d=2.0, c=1.0)
    print("\n")
    
    trainSVM_RBF(DTE, LTE, DTR, LTR, K=1.0, C=1.0, gamma=10.0)
    
    end = time.time()
    print(end - start)