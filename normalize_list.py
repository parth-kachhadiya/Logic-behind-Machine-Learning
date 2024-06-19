import numpy as np

def normalize_list(data):
    newLs = []
    for value in data:
        newLs.append((value - min(data))/(max(data) - min(data)))
        
    return newLs
    
ls = [10,20,30,40,50]

print(normalize_list(ls))