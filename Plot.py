import matplotlib.pyplot as plt
import numpy
import scipy.stats as st

def plotHist(D, L, string):
    #Ogni riga della matrice è una feature
    D0 = D[:, L == 0]
    D1 = D[:, L == 1]

    hFea = {
        0: 'Fixed Acidity',
        1: 'Volatile Acidity',
        2: 'Citric Acid',
        3: 'Residual Sugar',
        4: 'Chlorides',
        5: 'Free Sulfur Dioxide',
        6: 'Total Sulfur Dioxide',
        7: 'Density',
        8: 'pH',
        9: 'Sulphates',
        10: 'Alcohol',
    } 

    for i in range(D.shape[0]):
        plt.figure()
        plt.xlabel(hFea[i])
        plt.hist(D0[i, :], bins=100, density=True, alpha=0.4, label='Bad Wine', color='red')
        plt.hist(D1[i, :], bins=100, density=True, alpha=0.4, label='Good Wine', color='blue')
        plt.legend()
        plt.tight_layout()  # Use with non-default font size to keep axis label inside the figure
        plt.savefig(string + 'hist_%d.pdf' % i)
        plt.close()

def HeatMapPearson(D, string):
    corr = []
    for i in range(D.shape[0]):
        row = []
        for j in range(D.shape[0]):
            v = st.pearsonr(D[i],D[j])
            row.append(v[0])
        corr.append(row)

    corr = numpy.vstack(corr)

    plt.imshow(corr, cmap="Greys")
    plt.xlabel(string)
    plt.savefig(string + "HeatMap.pdf")
    plt.close()

def plotDCF(x, y):
    #x = numpy.linspace(min(x), max(x), 7)
    plt.plot(x, y, label = "DCF")
    plt.xscale("log")

    plt.savefig("Plot_LR.pdf")