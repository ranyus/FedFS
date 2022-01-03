"""
Porting from MIToolbox of the functions to Merge Multidimensional arrays into one. 
"""
import numpy as np
import pymit as mit


def normaliseArray(inVec):
    """
        normaliseArray takes an input vector and return an output vector 
        which is a normalised version of the input, and returns the number of states.
        A normalised array has min value = 0, max value = number of states and all values are integers
    """
    minVal = np.uint32(inVec.min())

    return [np.uint32(inVec)-minVal,np.uint32(inVec.max()-minVal+1)]



def merge_arrays_fast(firstVec,numStateFirst,secondVec, numStateSecond):
    """
    Fast version of merge arrays 
    """
    baseidx=firstVec+(secondVec*numStateFirst)
    upos= np.sort(np.unique(baseidx,return_index=True)[1])
    dic=dict(zip(baseidx[upos],np.arange(len(upos))+1))

    return [np.array([dic.get(baseidx[id]) for id in np.arange(len(baseidx))]),len(upos)+1]

def merge_arrays(firstVec,numStateFirst,secondVec, numStateSecond):
    """
    Merges two arrays into a joint state spase. Returns the joint vector and the number of states
    """

    stateMap = np.zeros(numStateFirst*numStateSecond,dtype=np.uint32)

    stateCount = 1

    outVec = np.zeros_like(firstVec,dtype=np.uint32)

    for i in np.arange(firstVec.shape[0]):
        curIdx = np.uint32(firstVec[i] + (secondVec[i] * numStateFirst))
        outVec[i] = stateMap[curIdx]
    
    del stateMap
    
    return [outVec,stateCount]



def disc_and_merge_arrays(firstVec,secondVec):
    """ 
    Merges two non nonrmalised arrays
    """
    firstNormVec,numStateFirst = normaliseArray(firstVec)
    secondNormVec,numStateSecond = normaliseArray(secondVec)

    outVec,stateCount = merge_arrays(firstNormVec,numStateFirst,secondNormVec,numStateSecond)

    del(firstNormVec)
    del(secondNormVec)
    return [outVec,stateCount]



def merge_multiple_arrays(inMat:np.array):
    """ 
    Computes the joint state space vector of an arbitrary number column vectors.  
    """

    curNumStates=0
    vecLength, numFeat = inMat.shape

    

    if (numFeat>2):
        outVec,curNumStates = disc_and_merge_arrays(inMat[:,0],inMat[:,1])
        
        for i in np.arange(2,numFeat):
            normVec,secNumStates=normaliseArray(inMat[:,i])
            outVec,curNumStates = merge_arrays_fast(outVec,curNumStates,normVec,secNumStates)

            del normVec
    elif (numFeat==2):
        outVec,curNumStates = disc_and_merge_arrays(inMat[:,0],inMat[:,1])
    else:
        outVec,curNumStates = normaliseArray(inMat[:,0])
        


    return [outVec,curNumStates]


def condH(X, Y):
    """
    Computes the conditional entropy H(X|Y)
    """

    D, ds = merge_multiple_arrays(Y)
    condH = mit.H_cond(X, D, bins=[int(np.unique(X)[-1] + 1), ds])
    #     del(mergedFirst)
    #     del(mergedSecond)

    return condH
