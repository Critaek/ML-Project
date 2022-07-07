import numpy
import LinearRegression as lr

def calibrate_scores(scores, labels, prior_t, l = 1e-3):
    numpy.random.seed(0)
    idx = numpy.random.permutation(labels.size)
    labels = labels[idx]
    alpha, beta = lr.LinearRegression_w_b(scores, labels, l, prior_t) #insert optimal lambda
    return alpha*scores + beta - numpy.log(prior_t/(1-prior_t))