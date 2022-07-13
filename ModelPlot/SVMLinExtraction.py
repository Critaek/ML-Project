import numpy
import Plot as plt 

if __name__ == "__main__":
    f = open("data/SVMLinear.txt", "r")
    normalized=[]
    raw=[]

    best_mindcf_Norm=0
    previousbest_mindcf_Norm=0
    best_mindcf_Raw=0
    previousbest_mindcf_Raw=0

    K_Set = numpy.array([0.0, 1.0, 10.0])
    C_Set = numpy.logspace(-2,0, num = 10)

    #   0.1 | 0.1 | SVM Linear | K = 0.0 | C = 0.01 | Raw | Uncalibrated | PCA = 10 | ActDCF =1.000 | MinDCF =0.993

    #(prior , MinDCF , K , C)
    for line in f:
        elements = line.split("|")
        elements =[elem.strip() for elem in elements]
        if(float(elements[0]) == 0.5):
            if(elements[5]=="Normalized" and elements[6]=="Uncalibrated"):
                normalized.append(( float(elements[1]), float(elements[9][8:]), float(elements[3][4:]), float(elements[4][4:]) ))

            elif(elements[5]=="Raw" and elements[6]=="Uncalibrated"):
                raw.append(( float(elements[1]), float(elements[9][8:]), float(elements[3][4:]), float(elements[4][4:]) ))

    
    
    #(pi_tilde, lambda, mindcf)
    normalized05=[]
    normalized09 = []
    normalized01 = []
    raw05=[]
    raw09 = []
    raw01 = []
    
    #x = lambda, y = mindcf
    
   


    
    

    filtNorm05=filter(lambda x: x[0] == 0.5, normalized)
    bestNorm05=min(raw, key=lambda x: x[1])
    filtNorm05=filter(lambda x: x[2] == bestNorm05[2], filtNorm05)
    for i in filtNorm05:
        if(i[2]==bestNorm05[2]):
            normalized05.append(i[1])
    normalized05=numpy.array(normalized05)        
         
   
 

    filtNorm01=filter(lambda x: x[0] == 0.1, normalized)
    bestNorm01=min(raw, key=lambda x: x[1])
    filtNorm01=filter(lambda x: x[2] == bestNorm01[2], filtNorm01)
    for i in filtNorm01:
        if(i[2]==bestNorm01[2]):
            normalized01.append(i[1])
    normalized01=numpy.array(normalized01) 
    

    filtNorm09=filter(lambda x: x[0] == 0.9, normalized)
    bestNorm09=min(raw, key=lambda x: x[1])
    filtNorm09=filter(lambda x: x[2] == bestNorm09[2], filtNorm09)
    for i in filtNorm09:
        if(i[2]==bestNorm09[2]):
            normalized09.append(i[1])
    normalized09=numpy.array(normalized09) 
   

    filtRaw05=filter(lambda x: x[0] == 0.5, raw)
    bestRaw05=min(raw, key=lambda x: x[1])
    print(bestRaw05)
    filtRaw05=filter(lambda x: x[2] == bestRaw05[2], filtRaw05)
    for i in filtRaw05:
        if(i[2]==bestRaw05[2]):
            raw05.append(i[1])
    raw05=numpy.array(raw05) 

    filtRaw01=filter(lambda x: x[0] == 0.1, raw)
    bestRaw01=min(raw, key=lambda x: x[1])
    print(bestRaw01)
    filtRaw01=filter(lambda x: x[2] == bestRaw01[2], filtRaw01)
    for i in filtRaw01:
        if(i[2]==bestRaw01[2]):
            raw01.append(i[1])
    raw01=numpy.array(raw01) 

    filtRaw09=filter(lambda x: x[0] == 0.9, raw)
    bestRaw09=min(raw, key=lambda x: x[1])
    print(bestRaw09)
    filtRaw09=filter(lambda x: x[2] == bestRaw09[2], filtRaw09)
    for i in filtRaw09:
        if(i[2]==bestRaw09[2]):
            raw09.append(i[1])
    raw09=numpy.array(raw09)    

    plt.plotThreeDCFs(C_Set, normalized05, normalized09, normalized01,"C","Normalized")
    plt.plotThreeDCFs(C_Set,raw05,raw09,raw01,"C","Raw")
           