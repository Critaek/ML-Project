from ast import Load
import numpy
import Load
import SVM
import ModelEvaluation as me
import Normalization
import DimReduction as dr
import MVG
import QuadraticRegression as qr
import GMM

if __name__ == "__main__":
    DTR, LTR = Load.load()
    DTE, LTE = Load.load_test()

    DTR = Normalization.Normalization(DTR)
    DTE = Normalization.Normalization(DTE)
    
    #NormDTE function computes DTE rankings from DTR, for some reasons, it doesn't work
    #see implementations in Normalization.py with comment

    #DTE = Normalization.NormDTE(DTR, DTE)

    P = dr.PCA_P(DTR, 10)

    DTR10 = numpy.dot(P.T, DTR)
    DTE10 = numpy.dot(P.T, DTE)

    
    Predictions, Scores = MVG.MultiV(DTR10, LTR, DTE10, 0.5)
    print("MVG Full")
    actDCF, MinDCF = me.printDCFsNoShuffle(DTE, LTE, Scores > 0, Scores, 0.5)
    print("ActDCF", actDCF)
    print("MinDCF", MinDCF)

    Predictions, Scores = qr.QuadraticRegression(DTR10, LTR, DTE10, 0.0207, 0.5)
    print("Quadratic Regression")
    actDCF, MinDCF = me.printDCFsNoShuffle(DTE, LTE, Scores > 0, Scores, 0.5)
    print("ActDCF", actDCF)
    print("MinDCF", MinDCF)

    Predictions, Scores = SVM.SVMPoly(DTR10, LTR, DTE10, LTR, 10.0, 0.01, 3.0, 1.0, 0.5)
    print("SVM Poly 0.5")
    actDCF, MinDCF = me.printDCFsNoShuffle(DTE, LTE, Scores > 0, Scores, 0.5)
    print("ActDCF", actDCF)
    print("MinDCF", MinDCF)

    Predictions, Scores = SVM.SVMPoly(DTR10, LTR, DTE10, LTR, 10.0, 0.01, 3.0, 1.0, 0.9)
    print("SVM Poly 0.9")
    actDCF, MinDCF = me.printDCFsNoShuffle(DTE, LTE, Scores > 0, Scores, 0.5)
    print("ActDCF", actDCF)
    print("MinDCF", MinDCF)

    Predictions, Scores = SVM.SVM_RBF(DTR10, LTR, DTE10, LTR, 1.0, 1.0, 0.1, 0.5)
    print("SVM RBF")
    actDCF, MinDCF = me.printDCFsNoShuffle(DTE, LTE, Scores > 0, Scores, 0.5)
    print("ActDCF", actDCF)
    print("MinDCF", MinDCF)

    GMM0 = GMM.GMM_LBG_Full(DTR10[:, LTR == 0], 4)
    GMM1 = GMM.GMM_LBG_Full(DTR10[:, LTR == 1], 4)
    Predictions, Scores = GMM.GMM_Scores(DTE10, GMM0, GMM1)

    print("GMM")
    actDCF, MinDCF = me.printDCFsNoShuffle(DTE, LTE, Scores > 0, Scores, 0.5)
    print("ActDCF", actDCF)
    print("MinDCF", MinDCF)


