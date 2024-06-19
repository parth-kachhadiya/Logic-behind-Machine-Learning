import numpy as np

def one_hot_encoding(data):
    final_category = [] # Identify total number of unique categories

    for value in data:
        if value not in final_category:
            final_category.append(value)

    number_of_rows = len(data) # Number of rows in final matrix
    number_of_columns = len(final_category) # Number of columns in final matrix

    matrix = np.zeros((number_of_rows,number_of_columns))
    
    index = []
    for val in (data): # Final matrix operation
        for ct in final_category:
            if ct == val:
                index.append(final_category.index(val))

    for i, mat in zip(index, matrix):
        mat[i] = 1

    return matrix
category = ['cat','dog','dog','dog','cat','cat','parrot','cat','parrot','cat','dog']
print(one_hot_encoding(category))