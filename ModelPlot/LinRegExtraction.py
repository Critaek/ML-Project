import numpy
import Plot as plt 

if __name__ == "__main__":
    f = open("data/LinRegPlot.txt", "r")
    normalized=[]
    raw=[]
    lambdas = numpy.logspace(-5, 2, num = 20)

    #prior_t|p_tilde|Linear Regression|Lambda =1.00e-5|Raw|PCA = 10|ActDCF =0.552|MinDCF =0.517
    #Sempre 0.5 | varia | fisso | varia | cambia
   
    for line in f:
        elements = line.split("|")
        elements =[elem.strip() for elem in elements] 
        if(elements[4]=="Normalized"):
            normalized.append(( elements[1], float(elements[7][8:]) ))
        else:
            raw.append(( elements[1], float(elements[7][8:]) ))

    nor = numpy.array(normalized,dtype="float")
    raw = numpy.array(raw,dtype="float")
    
    #(pi_tilde, lambda, mindcf)
    normalized05=[]
    normalized09 = []
    normalized01 = []
    raw05=[]
    raw09 = []
    raw01 = []
    
    #x = lambda, y = mindcf
    
    for n in nor:
        if(float(n[0]) == 0.5):
            normalized05.append(n[1])
        if (float(n[0]) == 0.1):
            normalized01.append(n[1])
        if (float(n[0]) == 0.9):
            normalized09.append(n[1])
        
    for n in raw:
        if(float(n[0]) == 0.5):
            raw05.append(n[1])
        if (float(n[0]) == 0.1):
            raw01.append(n[1])
        if (float(n[0]) == 0.9):
            raw09.append(n[1])
            
     #Saving results       
    numpy.save("data_to_plot/LinRegnormalized05",normalized05)
    numpy.save("data_to_plot/LinRegnormalized01",normalized01)
    numpy.save("data_to_plot/LinRegnormalized09",normalized09)
    numpy.save("data_to_plot/LinRegraw05",raw05)
    numpy.save("data_to_plot/LinRegraw01",raw01)
    numpy.save("data_to_plot/LinRegraw09",raw09)

    raw05 = numpy.array(raw05)
    raw01 = numpy.array(raw01)
    raw09 = numpy.array(raw09)

    plt.plotThreeDCFs(lambdas, normalized05, normalized09, normalized01, "λ")
    plt.plotThreeDCFs(lambdas, raw05, raw09, raw01, "λ")
    #plt.plotDCF(lambdas, raw05)        