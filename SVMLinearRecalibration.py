import SVM
import DimReduction as dr
import ModelEvaluation as me
import ScoreCalibration as sc
import numpy
import Load as l

def mcol(vect):
    return vect.reshape(vect.size, 1)

def vrow(vect):
    return vect.reshape(1, vect.size)

def trainSVMLinear(D, L, NormD, K_Set, C_Set, prior_t): #K relativo al modello, non k_fold
    i = 5
    prior_tilde_set = [0.33, 0.5, 0.66]

    l_prova = numpy.logspace(-5, 2, num = 20)

    for l in l_prova:
        print("Lambda per calibration: ", l)
        for K in K_Set:
            for C in C_Set:
                PCA = dr.PCA(D, L, 5+i)
                Predictions, Scores = SVM.kFoldLinear(PCA, L, 5, K, C, prior_t)
                #Still called LLRs in the printDCFs function, but they are scores with no probabilistic interpretation
                #We use the same function for every model
                Scores = sc.calibrate_scores(vrow(Scores), L, prior_t, l)
                Scores = Scores.reshape(Scores.shape[1])
                for prior_tilde in prior_tilde_set:
                    ActDCF, minDCF = me.printDCFs(D, L, Scores > 0, Scores, prior_tilde) 
                    print(prior_t, "|" ,prior_tilde, "| SVM Linear | K =", K, "| C =", C, "| Raw | PCA =", 5+i,
                        "| ActDCF ={0:.3f}".format(ActDCF), "| MinDCF ={0:.3f}".format(minDCF))            

        for K in K_Set:
            for C in C_Set:
                PCA = dr.PCA(NormD, L, 5+i)
                Predictions, Scores = SVM.kFoldLinear(PCA, L, 5, K, C, prior_t)
                Scores = sc.calibrate_scores(vrow(Scores), L, prior_t, l)
                Scores = Scores.reshape(Scores.shape[1])
                for prior_tilde in prior_tilde_set:
                    ActDCF, minDCF = me.printDCFs(D, L, Scores > 0, Scores, prior_tilde)
                    print(prior_t, "|" ,prior_tilde, "| SVM Linear | K =", K, "| C =", C, "| Normalized | PCA =", 5+i,
                        "| ActDCF ={0:.3f}".format(ActDCF), "| MinDCF ={0:.3f}".format(minDCF))
    print("\n") 


if __name__ == "__main__":
    D, L = l.load()
    NormD = numpy.load("data/normalizedD.npy")

    different_Application=[0.5,0.33,0.66]
    for different_prior in different_Application:

        K_Set = numpy.array([0.0, 1.0, 10.0])
        C_Set = numpy.array([0.1, 1.0, 10.0])

        trainSVMLinear(D, L, NormD, K_Set, C_Set, different_prior)