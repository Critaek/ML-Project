import numpy

def mcol(v):
    return v.reshape(v.size, 1)

def PCA(D, L, m):
    mu = D.mean(1)
    mu = mcol(mu)
    DC = D - mu
    N = D.shape[1]
    C = numpy.dot(DC,DC.T)/N
    s, U = numpy.linalg.eigh(C)
    P = U[:, ::-1][:, 0:m]
    DP = numpy.dot(P.T, D)

    return DP