import numpy
import Plot as plt

if __name__ == "__main__":
    lines = []
    i_MinDCF = []
    lines = []

    f = open("data/GMMResult.txt", "r")
    for i, line in enumerate(f):
        elements = line.split("|")
        elements =[elem.strip() for elem in elements] 
        lines.append(elements)
        if (float(elements[0]) == 0.5 and elements[4] == "Uncalibrated"):
            MinDCF = float(elements[7][8:])
            i_MinDCF.append((i, MinDCF))

    MinDCF = min(i_MinDCF, key = lambda x: x[1])
    print(MinDCF)
    print(lines[MinDCF[0]])

    nComponents = numpy.array([0,1,2,3,4,5])

    #0.1 | GMM Full | nComponents = 1 | Raw | Uncalibrated | PCA = 10 | ActDCF =1.785 | MinDCF =0.813

    rawFull = []
    normalizedFull = []
    rawDiagonal = []
    normalizedDiagonal = []
    rawTied = []
    normalizedTied = []

    for line in lines:
        nC = line[2][14:]
        type = line[1][4:]
        pi = float(line[0])
        DataType = line[3]
        MinDCF = float(line[7][8:])
        Cal = line[4]
        if(pi == 0.5 and Cal == "Uncalibrated"):
            if(type == "Full"):
                if(DataType == "Raw"):
                    rawFull.append(MinDCF)
                else:
                    normalizedFull.append(MinDCF)
            if(type == "Diagonal"):
                if(DataType == "Raw"):
                    rawDiagonal.append(MinDCF)
                else:
                    normalizedDiagonal.append(MinDCF)
            if(type == "Tied"):
                if(DataType == "Raw"):
                    rawTied.append(MinDCF)
                else:
                    normalizedTied.append(MinDCF)

    plt.plotHistGMM(nComponents, rawFull, normalizedFull, "Full")
    plt.plotHistGMM(nComponents,rawDiagonal, normalizedDiagonal, "Diagonal")
    plt.plotHistGMM(nComponents, rawTied, normalizedTied, "Tied")


        