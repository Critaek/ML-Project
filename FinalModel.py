from ast import Load
import numpy
import Load
import SVM
import ModelEvaluation as me
import Normalization
import DimReduction as dr

if __name__ == "__main__":
    DTR, LTR = Load.load()
    DTE, LTE = Load.load_test()

    DTR = Normalization.Normalization(DTR)
    DTE = Normalization.Normalization(DTE)

    DTR10 = dr.PCA(DTR, LTR, 10)
    DTE10 = dr.PCA(DTE, LTE, 10)

    

    Predictions, Scores = SVM.SVMPoly(DTR, LTR, DTE, LTR, 10.0, 0.01, 3.0, 1.0, 0.5)
    actDCF, MinDCF = me.printDCFsNoShuffle(DTE, LTE, Scores > 0, Scores, 0.5)

    print("ActDCF", actDCF)
    print("MinDCF", MinDCF)
