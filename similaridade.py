def dice(setA, setB):
    return ( 2 * len(setA.intersection(setB)) / ( len(setA) + len(setB) ) )