import numpy as np
# import kdemodels
# import commoncomputation
import _mixturemodels
from _mixturemodels import FixBwMixtureModels


# expectation-maximization algorithm for mixture kde models

# hidden variables
# Z=[z_ij]
# i: i-th component
# j: j-th observation
# z_ij: whether the j-th observation belongs to the i-th component


# em framework
def runEM(dataset, mixmodels, initialValue=None):
    nComponents = mixmodels.nCom
    nObservations = len(dataset)
    Z = np.zeros((nComponents, nObservations))  # hidden variables
    # theta=[0]*nComponents #parameters that are to be estimated
    # mixtureKDE=(theta,models)

    if initialValue is None:
        init(mixmodels)  # init
    epsilon = 0.0001
    # logLikelihoodValue=commoncomputation.logLikelihood(mixtureKDE,dataset)
    logLikelihoodValue = mixmodels.computeLogLikelihood(dataset)
    # print 'initial ll:{0}'.format(logLikelihoodValue)
    nIteration = 100  # maximum number of iterations
    while True:
        EStep(Z, mixmodels, dataset)
        MStep(Z, mixmodels, dataset)
        nIteration -= 1
        # check whether to terminate
        # updatedLogLikelihoodValue=commoncomputation.logLikelihood(mixtureKDE,dataset)
        updatedLogLikelihoodValue = mixmodels.computeLogLikelihood(dataset)
        # print 'll:{0}'.format(updatedLogLikelihoodValue)
        delta = abs(updatedLogLikelihoodValue - logLikelihoodValue)
        # print 'delta:{0}'.format(delta)
        if delta <= epsilon or nIteration == 0:
            break
        logLikelihoodValue = updatedLogLikelihoodValue
    print 'em ends'


# return theta

# initial value
def init(mixtureKDE):
    for i in range(0, len(mixtureKDE.params)):  # sum over all elements is 1
        mixtureKDE.params[i] = 1.0 / len(mixtureKDE.params)
    # print('em.py function:init---{0}'.format(mixtureKDE.params))


# E-step
def EStep(Z, mixtureKDE, dataset):  # compute Z
    theta = mixtureKDE.params
    models = mixtureKDE.models
    for j in range(0, len(dataset)):
        sumOfProbs = 0.0
        for i in range(0, len(theta)):
            # Z[i,j]=theta[i]*kdemodels.getProbabilityDensity(models[i],dataset[j])
            # Z[i,j]=theta[i]*models[i].computeProbs([dataset[j]])[0]
            Z[i, j] = theta[i] * mixtureKDE.computeProbOfCompoment(dataset[j], i)
            sumOfProbs += Z[i, j]
        if sumOfProbs != 0:
            for i in range(0, len(theta)):  # normalization
                Z[i, j] /= sumOfProbs
        else:
            MIN = 1e-100
            for i in range(0, len(theta)):  # normalization
                Z[i, j] /= MIN


# M-step
def MStep(Z, mixtureKDE, dataset):  # update theta
    theta = mixtureKDE.params
    models = mixtureKDE.models
    for i in range(0, len(theta)):
        for j in range(0, len(dataset)):
            theta[i] += Z[i, j]
        theta[i] /= len(dataset)
    sumOfProbs = sum(theta)
    for i in range(0, len(
            theta)):  # normalization. Although it is not needed in theory, but it is needed when we use float numbers.
        theta[i] /= sumOfProbs
