import numpy as np

def my_comp(tupla):
    return tupla[1]

def dice(setA, setB):
    return ( 2 * len(setA.intersection(setB)) / ( len(setA) + len(setB) ) )

def cosseno(vector1, vector2):
    dot = np.dot(vector1, vector2)

    sum1 = np.sqrt( np.dot(vector1,vector1) )
    e_length1 = np.sqrt(sum1)

    sum2 = np.sqrt( np.dot(vector2,vector2) )
    e_length2 = np.sqrt(sum2)

    return dot/(sum1*sum2)