import numpy
import LinearRegression as lr

def calibrate_scores(scores, labels, prior_t):
    alpha, beta = lr.LinearRegression_w_b(scores, labels, 1e-5, prior_t) #insert optimal lambda
    return alpha*scores + beta - numpy.log(prior_t/(1-prior_t))