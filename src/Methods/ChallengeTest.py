__author__ = 'dengzhihong'


from src.Methods.TestMethods import *

def testWithChallenge(train_vectors, train_labels, test_vectors, test_labels, trainset, PCA_K, C, GAMMA, method):
    TrainVectors = array(toFloatList(train_vectors)).reshape(4000, 784)
    TrainLabel = array(toFloatList(train_labels))
    TestVectors = array(toFloatList(test_vectors)).reshape(50, 784)
    TestLabel = array(toFloatList(test_labels))

    TrainNum = len(trainset)
    train = array(toFloatList(trainset)).reshape(TrainNum/2,2) - 1

    DataVectors = vstack((TrainVectors, TestVectors))
    DataVectors = normalization(DataVectors)

    if(PCA_K != 0):
        k = PCA_K
        pca = PCA(n_components=k)
        DataVectors = pca.fit_transform(DataVectors)
        sum = 0
        for i in range(k):
            sum += pca.explained_variance_ratio_[i]
        RetainRate = sum * 100
        #print "retain " + str(RetainRate) +"% of the variance"

    TrainData = DataVectors[0:4000]
    TestData = DataVectors[4000:4050]

    #TrainData, TrainLabel = prepareData(TrainData, TrainLabel, train)

    #outputTrainingData(TrainData, TrainLabel, PCA_K)
    clf = 0
    ClfSet = []
    #print 'Method: ', method
    if(method == 'svm_linear'):
        ClfSet = prepareSvmClassifier(TrainData, TrainLabel, 10, 'linear')
    elif(method == 'svm_poly'):
        ClfSet = prepareSvmClassifier(TrainData, TrainLabel, 10, 'poly')
    elif(method == 'svm_rbf'):
        ClfSet = prepareSvmClassifier(TrainData, TrainLabel, 10, 'rbf', c=C, Gamma=GAMMA)
    elif(method == 'lr'):
        clf = LogisticRegression()
        clf.fit(TrainData, TrainLabel)
    elif(method == '1nn'):
        clf = KNeighborsClassifier(n_neighbors=1)
        clf.fit(TrainData, TrainLabel)
    else:
        print "Please Choose a kernel"
        exit()

    N = TestData.shape[0]
    #print "Totoal ", N, " test data"
    correct = 0.0
    CorrectNum = []
    for i in range(10):
        CorrectNum.append(0)

    if(method[0:3] == 'svm'):
        for i in range(N):
            confidence = -999
            classification = -1
            for j in range(10):
                temp = ClfSet[j].decision_function(TestData[i])
                if(confidence < temp):
                    confidence = temp
                    classification = j
            if(classification ==  TestLabel[i]):
                CorrectNum[classification] += 1
                correct += 1
    else:
        result = clf.predict(TestData)
        for i in range(N):
            if(result[i] == TestLabel[i]):
                CorrectNum[int(result[i])] += 1
                correct += 1

    Correctness = correct/N * 100
    '''
    print "Accuracy: ", Correctness, "%"
    for i in range(10):
        print "digit ", i , ": ", CorrectNum[i], "/10"
    '''
    return Correctness

