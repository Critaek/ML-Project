import numpy

def mcol(v):
    return v.reshape(v.size, 1)

def load():
    train = open("data/Train.txt")
    DList = []
    LabelsList = []
    for line in train:
        numbers = line.split(",")[0:-1]
        numbers = mcol(numpy.array([float(i) for i in numbers]))
        DList.append(numbers)
        LabelsList.append(line.split(",")[-1])
    
    D = numpy.hstack(DList) 
    L = numpy.array(LabelsList, dtype=numpy.int32)

    return D, L