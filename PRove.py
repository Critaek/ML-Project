def k_fold(D, L, seed=0, K=5):
    """
    K-fold algorithm

    Parameters
    ----------
    D : Dataset to split    

    L : Labels of the input dataset

    seed : optional (default = 0). Seed the legacy random number generator.

    K : number of output subset, optional (default=5)

    Returns
    -------
    subsets : array of tuple ((DTrain_i, LTrain_i), (DTest_i, LTest_i))
    """

    folds = []  # array of tuple (D_i, L_i)
    subsets = []  # array of tuple ((DTrain_i, LTrain_i), (DTest_i, LTest_i) )

    # Split the dataset into k folds
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    l = 0
    r0 = int(D.shape[1] / K)
    r = r0
    for i in range(K):
        if i == K-1:
            r = D.shape[1]
        subset_i = idx[l:r]
        D_i = D[:, subset_i]
        L_i = L[subset_i]
        folds.append((D_i, L_i))
        l = r
        r = r + r0

    # Generate the k subsets
    for i in range(K):
        test_i = folds[i]
        first = True
        for j in range(K):
            if j != i:
                if first:
                    Dtrain_i = folds[j][0]
                    Ltrain_i = folds[j][1]
                    first = False
                else:
                    Dtrain_i = numpy.hstack([Dtrain_i, folds[j][0]])
                    Ltrain_i = numpy.hstack([Ltrain_i, folds[j][1]])
        subsets.append(((Dtrain_i, Ltrain_i), test_i))

    return subsets


    #-------------------------------------------------------------------------------------------------------------------#

    def MVG_Full(DT,LT,DE,LE,prior):
    
    mean0=empirical_mean(DT[:, LT == 0]) # Mean of the Gaussian Curve for Class 0
    mean1=empirical_mean(DT[:, LT == 1]) # Mean of the Gaussian Curve for Class 1
    
    sigma0=empirical_cov(DT[:, LT == 0]) # Sigma of the Gaussian Curve for Class 0
    sigma1=empirical_cov(DT[:, LT == 1]) # Sigma of the Gaussian Curve for Class 1
    
    LS0 = logpdf_GAU_ND(DE, mean0, sigma0) # Log Densities with parameters of Class 0
    LS1 = logpdf_GAU_ND(DE, mean1, sigma1) # Log Densities with parameters of Class 1
    
    SJoint = numpy.zeros((2, DE.shape[1]))
    
    SJoint[0, :] = numpy.exp(LS0) * (1-prior)        #Product Between Densities LS0 and PriorProb
    SJoint[1, :] = numpy.exp(LS1) * (prior)          #Product Between Densities LS1 and PriorProb
    
    SMarginal = vrow(SJoint.sum(0))
    SPost = SJoint / SMarginal
    LabelPred=SPost.argmax(0)
    
    llr = LS1-LS0 # log-likelihood ratios
    
    return LabelPred, llr


def MVG_Diag(DT,LT,DE,LE,prior):
    
    mean0=empirical_mean(DT[:, LT == 0]) # Mean of the Gaussian Curve for Class 0
    mean1=empirical_mean(DT[:, LT == 1]) # Mean of the Gaussian Curve for Class 1
    
    sigma0=numpy.diag(numpy.diag(empirical_cov(DT[:, LT == 0]) )) # Diagonal Sigma of the Gaussian Curve for Class 0
    sigma1=numpy.diag(numpy.diag(empirical_cov(DT[:, LT == 1])))  # Diagonal Sigma of the Gaussian Curve for Class 1
  
    LS0 = logpdf_GAU_ND(DE, mean0, sigma0) # Log Densities with parameters of Class 0
    LS1 = logpdf_GAU_ND(DE, mean1, sigma1) # Log Densities with parameters of Class 1
    
    SJoint = numpy.zeros((2, DE.shape[1]))
    
    SJoint[0, :] = numpy.exp(LS0) * (1-prior)        #Product Between Densities LS0 and PriorProb
    SJoint[1, :] = numpy.exp(LS1) * (prior)          #Product Between Densities LS1 and PriorProb
   
    SMarginal = vrow(SJoint.sum(0))
    SPost = SJoint / SMarginal
    LabelPred=SPost.argmax(0)
    
    llr = LS1-LS0 # log-likelihood ratios
   
    return LabelPred, llr


    #-------------------------------------------------------------------------------------------------------

    def computeDCFu(TrueLabel,PredLabel,pi,CostMatrix):
    '''
    Compute the DCFu value. This function can work in binary case and also in multiclass
    case. In Binary case we suggest to insert for pi values the float number of the prior
    probability for True Class, but, if not possible, can also be used the prior vec probabilites
    formatted in this way: [False Class Prob, True Class Prob]

    Parameters
    ----------
    TrueLabels : Type: Numpy Array 
                 Description: Array of correct labels.
    PredictedLabels : Type: Numpy Array 
                      Description: Array of predicted labels.
    pi : Type: Numpy Array or Float Single Value 
         Description: Array of prior probabilies or Single probabilities of TrueClass.
    CostMatrix : Type: Numpy Array 
                 Description: Matrix of costs. Depends from the application.

    Returns
    -------
    DCFu : Type: Float Value
           Description: DCFu Value.

    '''
    
    # Compute the number of classes == dimension of the CostMatrix
    dimension=CostMatrix.shape[0]
    
    # Compute Confusion Matrix using TrueLabels and PredLabesl
    ConfusionM=computeConfusionMatrix(TrueLabel,PredLabel,dimension)
    
    # Compute MissclassRateos by dividing each element by the sum of values of its column
    MisclassRateos=ConfusionM/ ConfusionM.sum(axis=0)
    
    # If Dimension is 2 and pi is only a float, then we need to create the pi vec
    # Else we don't need to create nothing
    if(dimension==2 and (isinstance(pi, float))):
        pi_vec=numpy.array([(1-pi),pi])
    else:
        pi_vec=pi
    
    # Calculate the product between MisclassRateos and CostMatrix
    # We use the MisclassRateos transposed because the product is calculated
    # In the formula colum by column, and this can be replicated in Matrix product
    # Transposing one of the 2 matrices
    # In conclusion we takes only the diagonal elements because corrisponding
    # To the correct products
    # What we obtain correspond to the following formula
    # SemFin=Sum_{i=1}^k R_{ij}C_{ij}
    # With R the MisclassRateos and C the CostMatrix
    SemFin=numpy.dot(MisclassRateos.T,CostMatrix).diagonal()
    
    # Return the product by SemFin and pi_vec
    # Sum_{j=1}^k pi_{j} * SemFin
    return numpy.dot(SemFin,pi_vec.T)


def computeNormalizedDCF(TrueLabel,PredLabel,pi,CostMatrix):
    '''
    Compute the Normalized DCF value. This function can work in binary case and also in multiclass
    case. In Binary case we suggest to insert for pi values the float number of the prior
    probability for True Class, but, if not possible, can also be used the prior vec probabilites
    formatted in this way: [False Class Prob, True Class Prob]

    Parameters
    ----------
    TrueLabels : Type: Numpy Array 
                 Description: Array of correct labels.
    PredictedLabels : Type: Numpy Array 
                      Description: Array of predicted labels.
    pi : Type: Numpy Array or Float Single Value 
         Description: Array of prior probabilies or Single probabilities of TrueClass.
    CostMatrix : Type: Numpy Array 
                 Description: Matrix of costs. Depends from the application.

    Returns
    -------
    DCF : Type: Float Value
           Description: DCF Normalized Value.

    '''
    
    # Calculate the DCFu value
    dcf_u=computeDCFu(TrueLabel,PredLabel,pi,CostMatrix)
    
    # If Dimension is 2 and pi is only a float, then we need to create the pi vec
    # Else we don't need to create nothing
    if(CostMatrix.shape[0]==2 and ( isinstance(pi, float))):
        pi_vec=numpy.array([(1-pi),pi])
    else:
        pi_vec=pi
    
    # Return the DCFu value divided by the minimum values from the product 
    # By CostMatrix and pi_vec
    return dcf_u/numpy.min(numpy.dot(CostMatrix, pi_vec))



    #--------------------------------------------------------------------------------------------

    def computeMinDCF(TrueLabel,llRateos,pi,CostMatrix):
    '''
    Compute the minimum DCF value. This function can work only in binary case We suggest to insert 
    for pi values the float number of the prior probability for True Class, but, if not possible, 
    can also be used the prior vec probabilites formatted in this way: [False Class Prob, True Class Prob]

    Parameters
    ----------
    TrueLabels : Type: Numpy Array 
                 Description: Array of correct labels.
    llRateos : Type: Numpy Array 
               Description: Array of posterior probabilities rateos.
    pi : Type: Numpy Array or Float Single Value 
         Description: Array of prior probabilies or Single probabilities of TrueClass.
    CostMatrix : Type: Numpy Array 
                 Description: Matrix of costs. Depends from the application.

    Returns
    -------
    Min DCF : Type: Float Value
              Description: Min DCF Value.
    '''
    
    # Concatenate the ordered llRateos with -inf and +inf
    T=numpy.concatenate([ numpy.array([-numpy.inf]), numpy.sort(llRateos) , numpy.array([numpy.inf])])
    
    # Set the minimum value of DCF to +inf
    minDCF=numpy.inf
    
    # Iterate over all the llRateos plus +inf and -inf
    # For all this values used as threashold to classificate labels
    # We calculate the Normalized DCF and after we select the minimum 
    # Between all the DCF generated
    for z,t in enumerate(T):
        PredictedLabel=numpy.int32(llRateos>t)
        DCF=computeNormalizedDCF(TrueLabel,PredictedLabel,pi,CostMatrix)
        minDCF=min(DCF,minDCF)
    
    # Return the minimu DCF
    return minDCF