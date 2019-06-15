import numpy as np


def find_most_compatible_match(candidate):
    """This method loop through candidate to find matches which has most compatible number"""
    best_matchIdx = -1
    best_matchVal = 0
    len_of_match = 5
    if not any(candidate):
        return -1
    for i in candidate:
        if W[len_of_match][i] > best_matchVal:
            best_matchVal = W[len_of_match][i]
            best_matchIdx = i
    return best_matchIdx
W = np.array([
              [0,0,0,1,1],
              [0,0,1,1,1],
              [0,1,0,1,0],
              [1,1,1,0,0],
              [1,1,0,0,0],
              [2,3,2,3,2]])
candidate = np.arange(5)

result= []
while True:
    index = find_most_compatible_match(candidate)
    if index == -1:
        break
    result.append(index)
    # candidate = np.delete(candidate, np.argwhere(candidate == index), axis=0)
    for i in candidate:
        if W[index, i] == 0:
            candidate = np.delete(candidate, np.argwhere(candidate == i))
print(result)
