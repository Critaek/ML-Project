import numpy
import LinearRegression as lr
import ModelEvaluation as me
import Load

'''
def calibrate_scores(scores, labels, prior_t, l = 1e-3):
    numpy.random.seed(0)
    idx = numpy.random.permutation(labels.size)
    labels = labels[idx]
    alpha, beta = lr.LinearRegression_w_b(scores, labels, l, prior_t) #insert optimal lambda
    return alpha*scores + beta - numpy.log(prior_t/(1-prior_t))
'''

def calibrate_scores(scores, labels, prior_t, l = 1e-5):
    #scores shufflati, labels no
    numpy.random.seed(0)
    idx = numpy.random.permutation(labels.size)
    labels = labels[idx]

    numpy.random.seed(10)
    idx = numpy.random.permutation(labels.size)
    scores = scores[idx]
    labels = labels[idx]

    splits = numpy.array_split(scores, 2)
    splitLabel = numpy.array_split(labels, 2)

    firstHalf = splits[0]
    secondHalf = splits[1]
    firstLabels = splitLabel[0]
    secondLabels = splitLabel[1]


    alpha1, beta1 = lr.LinearRegression_w_b(firstHalf, firstLabels, l, prior_t) #insert optimal lambda
    alpha2, beta2 = lr.LinearRegression_w_b(secondHalf, secondLabels, l, prior_t) #insert optimal lambda

    var = numpy.log(prior_t/(1-prior_t))

    newFirst = numpy.dot(alpha2[0], firstHalf) + beta2 - var
    newSecond = numpy.dot(alpha1[0], secondHalf) + beta1 - var

    return numpy.concatenate((newFirst, newSecond)), labels