def my_comp(tupla):
    return tupla[1]

def dice(setA, setB):
    return ( 2 * len(setA.intersection(setB)) / ( len(setA) + len(setB) ) )