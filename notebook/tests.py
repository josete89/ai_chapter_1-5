import numpy as np

def testSuccess():
    print('Test succeded!')

def testResult(result,expected,got):
    if result == False:
        print("Expected value '{0}' got '{1}' ".format(got,expected))
        assert result
    else:
        testSuccess()

def testReverseWords(reverseFunction):
    testInput = [5,6,7]
    translated = "and a of"
    result = reverseFunction(testInput)
    opRes = translated == result
    testResult(opRes,result,translated)


def testOneHot(oneHotFunction,maxPosition):
    arr = np.zeros((3,maxPosition))
    arr[0,3] = 1.
    arr[1,10] = 1.
    arr[2,20] = 1.
    res = oneHotFunction([3,10,20])
    opRes = np.array_equal(res,arr)
    testResult(opRes,arr,res)

def testPreparedData(prepareFunc,maxPosition):
    train,test = prepareFunc(np.array([3,10,20]),np.array([3,10,20]))
    arr = np.zeros((3, maxPosition))
    arr[0, 3] = 1.
    arr[1, 10] = 1.
    arr[2, 20] = 1.
    opRes = np.array_equal(train, arr) and np.array_equal(test, arr)
    fail = []
    if opRes == False:
        fail = train if np.array_equal(test, arr) else test
    testResult(opRes, arr, fail)

def testPrepareTargetData(targetDataFunc):
    testInput = [1,0,1,0,0]
    train,test = targetDataFunc(testInput,testInput)
    if type(train) is np.ndarray and type(test) is np.ndarray:
        if train.dtype == np.dtype(np.float32) and test.dtype == np.dtype(np.float32):
            testResult(True,None,None)
        else:
            fail = test if train.dtype == np.dtype(np.float32) else train
            testResult(False, fail,"Type is not float32")
    else:
        fail = test if type(train) is np.ndarray else train
        testResult(False,fail,"NUMPY ARRAY")

def testPredictionFormat(predictionFormatFunc,maxPosition):
    testInput = 'movie test'
    arr = np.zeros(maxPosition)
    arr[17] = 1.
    arr[2178] = 1.
    res = predictionFormatFunc(testInput)
    opRes = np.array_equal([arr],res)
    testResult(opRes,arr,res)

def testModelStructure(modelFunc):
    modelResult = modelFunc()
    expected = len(modelResult.layers) == 3
    numLayers = len(modelResult.layers)
    testResult(expected,numLayers,"Expected 3 layers")
