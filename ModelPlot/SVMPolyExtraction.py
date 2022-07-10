import enum
from locale import normalize
import numpy
import Plot as plt

if __name__ == "__main__":
    f = open("data/Poly.txt")
    i_MinDCF = []
    lines = []

    #0.1 | 0.1 | SVM Poly | K = 0.0 | C = 0.01 | d = 2.0 | c = 0.0 | Raw | Uncalibrated | PCA = 10 | ActDCF =1.737 | MinDCF =0.997

    for i, line in enumerate(f):
        elements = line.split("|")
        elements =[elem.strip() for elem in elements]
        lines.append(elements)
        MinDCF = elements[11][8:]
        i_MinDCF.append((i, float(elements[0]), MinDCF)) #(indice, prior, mindcf)
    
    i_MinDCF05 = filter(lambda x: x[1] == 0.5, i_MinDCF)
    MinDCF = min(i_MinDCF05, key = lambda x: x[2])
    print(MinDCF)
    index = MinDCF[0]
    print(lines[index])

    Best_K = lines[index][3]
    Best_d = lines[index][5]
    Best_c = lines[index][6]
    raw05 = []
    raw01 = []
    raw09 = []
    normalized05 = []
    normalized01 = []
    normalized09 = []


    for line in lines:
        DataType = line[7]
        Cal = line[8]
        prior_t = float(line[0])
        pi_tilde = float(line[1])
        K = line[3]
        d = line[5]
        c = line[6]
        minDCF = float(line[11][8:])

        if (prior_t == 0.5 and Cal == "Uncalibrated"):
            if (K == Best_K and d == Best_d and c == Best_c):
                if(DataType == "Raw"):
                    if(pi_tilde == 0.5):
                        raw05.append(minDCF)
                    if(pi_tilde == 0.1):
                        raw01.append(minDCF)
                    if(pi_tilde == 0.9):
                        raw09.append(minDCF)

                if(DataType == "Normalized"):
                    if(pi_tilde == 0.5):
                        normalized05.append(minDCF)
                    if(pi_tilde == 0.1):
                        normalized01.append(minDCF)
                    if(pi_tilde == 0.9):
                        normalized09.append(minDCF)

    C_Set = numpy.logspace(-2,0, num = 10)
    plt.plotThreeDCFs(C_Set, raw05, raw09, raw01, "C", "Raw")
    plt.plotThreeDCFs(C_Set, normalized05, normalized09, normalized01, "C", "Normalized")
                
                    