import enum
from locale import normalize
import numpy
import Plot as plt

if __name__ == "__main__":
    f = open("data/RBF.txt")
    i_MinDCF = []
    lines = []
    gamma_Set = numpy.logspace(-3,-1, num = 3)

    #0.1 | 0.1 | SVM RBF | K = 0.0 | C = 0.01 | gamma = 0.001 | Raw | Uncalibrated | PCA = 10 | ActDCF =1.000 | MinDCF =1.000

    for i, line in enumerate(f):
        elements = line.split("|")
        elements =[elem.strip() for elem in elements]
        lines.append(elements)
        if (elements[7] == "Uncalibrated"):
            MinDCF = elements[10][8:]
            i_MinDCF.append((i, float(elements[0]), MinDCF)) #(indice, prior, mindcf)
    
    i_MinDCF05 = filter(lambda x: x[1] == 0.5, i_MinDCF)
    MinDCF = min(i_MinDCF05, key = lambda x: x[2])
    print(MinDCF)
    index = MinDCF[0]
    print(lines[index])

    Best_K = lines[index][3]
    raw_gamma_0 = []
    raw_gamma_1 = []
    raw_gamma_2 = []
    normalized_gamma_0 = []
    normalized_gamma_1 = []
    normalized_gamma_2 = []


    for line in lines:
        DataType = line[6]
        Cal = line[7]
        prior_t = float(line[0])
        pi_tilde = float(line[1])
        K = line[3]
        C = line[4]
        gamma = float(line[5][8:])
        minDCF = float(line[10][8:])

        if (prior_t == 0.5 and Cal == "Uncalibrated" and pi_tilde == 0.5):
            if (K == Best_K):
                if(DataType == "Raw"):
                    if(gamma == gamma_Set[0]):
                        raw_gamma_0.append(minDCF)
                    if(gamma == gamma_Set[1]):
                        raw_gamma_1.append(minDCF)
                    if(gamma == gamma_Set[2]):
                        raw_gamma_2.append(minDCF)

                if(DataType == "Normalized"):
                    if(gamma == gamma_Set[0]):
                        normalized_gamma_0.append(minDCF)
                    if(gamma == gamma_Set[1]):
                        normalized_gamma_1.append(minDCF)
                    if(gamma == gamma_Set[2]):
                        normalized_gamma_2.append(minDCF)

    C_Set = numpy.logspace(-2,0, num = 10)

    plt.plotThreeDCFsRBF(C_Set, raw_gamma_0, raw_gamma_1, raw_gamma_2, "C", "Raw")
    plt.plotThreeDCFsRBF(C_Set, normalized_gamma_0, normalized_gamma_1, normalized_gamma_2, "C", "Normalized")