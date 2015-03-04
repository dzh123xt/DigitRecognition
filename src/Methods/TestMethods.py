__author__ = 'dengzhihong'

from numpy import *
import numpy as np
from sklearn.decomposition import *
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
import pylab
import matplotlib.cm as cm
import matplotlib.pyplot as plt

def randomTrail(vectors, labels):
    train = open("./Output/random_train.txt","w")
    test = open("./Output/random_test.txt","w")
    TrainIndice = []
    TestIndice = []
    for i in range(2000):
        rand = random.randint(0, 3999)
        while(TrainIndice.count(rand) != 0):
            rand = random.randint(0, 3999)
        TrainIndice.append(rand)
    TrainIndice.sort()
    for i in range(4000):
        if(TrainIndice.count(i) == 0):
            TestIndice.append(i)
    for i in range(2000):
        train.write(str(TrainIndice[i]) + " " + str(TrainIndice[i]) + "\n")
        test.write(str(TestIndice[i]) + " " + str(TestIndice[i]) + "\n")
    train.close()
    test.close()
    return 0


def toFloatList(stringlist):
    floatlist = []
    for i in range(0, len(stringlist)):
         floatlist.append(float(stringlist[i]))
    return floatlist

def toStrList(floatlist):
    strlist = []
    for i in range(len(floatlist)):
        strlist.append(str(floatlist[i]))
    return strlist

def prepareData(Data, Label, index):
    #process the data
    Num = index.shape[0]
    k = Data.shape[1]
    OutData = zeros((Num, k))
    OutLabel = zeros(Num)

    for i in range(Num):
        DataIndex = index[i][0]
        LabelIndex = index[i][1]
        OutData[i] = Data[DataIndex]
        OutLabel[i] = Label[LabelIndex]

    return OutData, OutLabel

def prepareSvmClassifier(TrainData, TrainLabel, N, Kernel, c=1.0, Gamma=0.0):
    ClfSet = []
    Num = TrainData.shape[0]
    for i in range(N):
        #print i , "++++++++++++"
        TempLabel = TrainLabel.copy()
        # Generate Label, once only a number's label will be +1 others will be -1
        for j in range(Num):
            if(TrainLabel[j] == i):
                TempLabel[j] = 1
            else:
                TempLabel[j] = -1
        #outputLabelList(TempLabel, "TrainLabel" + str(i), "Train Label For Classifier" + str(i))
        if(Kernel == 'linear'):
            clf = SVC(kernel='linear', C=c)
        elif(Kernel == 'poly'):
            clf = SVC(kernel='poly', C=c, gamma=Gamma)
        elif(Kernel == 'rbf'):
            clf = SVC(kernel='rbf', C=c, gamma=Gamma)
        #print '----------------------------------'
        #print "Fit classifier " , i
        clf.fit(TrainData, TempLabel)
        #print clf.support_vectors_.shape
        #print '----------------------------------'
        ClfSet.append(clf)
    return ClfSet


def prepareLrClassifier(TrainData, TrainLabel, N):
    ClfSet = []
    Num = TrainData.shape[0]
    for i in range(N):
        #print i , "++++++++++++"
        TempLabel = TrainLabel.copy()
        # Generate Label, once only a number's label will be +1 others will be -1
        for j in range(Num):
            if(TrainLabel[j] == i):
                TempLabel[j] = 1
            else:
                TempLabel[j] = -1
        #outputLabelList(TempLabel, "TrainLabel" + str(i), "Train Label For Classifier" + str(i))
        clf = LogisticRegression()
        #print '----------------------------------'
        #print "Fit classifier " , i
        clf.fit(TrainData, TempLabel)
        #print clf.support_vectors_.shape
        #print '----------------------------------'
        ClfSet.append(clf)
    return ClfSet

def processLabel(Label, target):
    ResultLabel =Label.copy()
    Num = Label.shape[0]
    # Generate Label, once only a number's label will be +1 others will be -1
    for j in range(Num):
        if(ResultLabel[j] == target):
            ResultLabel[j] = 1
        else:
            ResultLabel[j] = -1
    return ResultLabel

def showDigit(digit, title = "Digit"):
    fig = pylab.figure()
    pylab.title(title)
    fig.add_subplot(1,1,1)
    pylab.imshow(digit.reshape(28, 28).T, cmap = cm.Greys_r)
    pylab.show()

def outputTrainingData(TrainData, TrainLabel, PCA_K):
    output = open("./Output/Trial2_" + str(PCA_K) + ".txt", "w")
    N = TrainData.shape[0]
    D = TrainData.shape[1]
    for i in range(N):
        output.write(str(int(TrainLabel[i])) + " ")
        for j in range(D):
            output.write(str(j) + ":" + str(TrainData[i][j]) + " ")
        output.write("\n")
    output.close()

def normalization(data):
    return data/255.0 * 2 - 1

def testWithSVM(labels, vectors, testset, trainset, Kernel='linear', C=1.0, gamma=0.0, PCA_K=0):
    Label = array(toFloatList(labels))
    OriginData = array(toFloatList(vectors)).reshape(4000, 784)
    TestNum = len(testset)
    test = array(toFloatList(testset)).reshape(TestNum/2,2) - 1
    TrainNum = len(trainset)
    train = array(toFloatList(trainset)).reshape(TrainNum/2,2) - 1

    OriginData = normalization(OriginData)

    Data = OriginData
    RetainRate = 100
    if(PCA_K != 0):
        k = PCA_K
        pca = PCA(n_components=k)
        Data = pca.fit_transform(OriginData)
        sum = 0
        for i in range(k):
            sum += pca.explained_variance_ratio_[i]
        RetainRate = sum * 100
        print "retain " + str(RetainRate) +"% of the variance"

    TrainData, TrainLabel = prepareData(Data, Label, train)
    TestData, TestLabel = prepareData(Data, Label, test)

    #outputTrainingData(TrainData, TrainLabel, PCA_K)

    print 'SVM with ', Kernel, ' Kernel'
    if(Kernel == 'linear'):
        ClfSet = prepareSvmClassifier(TrainData, TrainLabel, 10, Kernel, C)
    elif(Kernel == 'poly'):
        ClfSet = prepareSvmClassifier(TrainData, TrainLabel, 10, Kernel, C, Gamma=gamma)
    elif(Kernel == 'rbf'):
        ClfSet = prepareSvmClassifier(TrainData, TrainLabel, 10, Kernel, C, Gamma=gamma)
    else:
        ClfSet = []
        print "Please Choose a kernel"
        exit()

    N = test.shape[0]
    correct = 0.0
    CorrectNum = []
    for i in range(10):
        CorrectNum.append(0)

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
    Correctness = correct/N * 100
    print "Accuracy: ", Correctness, "%"
    for i in range(10):
        print "digit ", i , ": ", CorrectNum[i], "/200"
    return RetainRate, Correctness, CorrectNum


def testWithLR(labels, vectors, testset, trainset, PCA_K=0):
    Label = array(toFloatList(labels))
    OriginData = array(toFloatList(vectors)).reshape(4000,784)
    TestNum = len(testset)
    test = array(toFloatList(testset)).reshape(TestNum/2,2) - 1

    TrainNum = len(trainset)
    train = array(toFloatList(trainset)).reshape(TrainNum/2,2) - 1
    OriginData = normalization(OriginData)
    Data = OriginData
    RetainRate = 100
    if(PCA_K != 0):
        k = PCA_K
        pca = PCA(n_components=k)
        Data = pca.fit_transform(OriginData)
        sum = 0
        for i in range(k):
            sum += pca.explained_variance_ratio_[i]
        RetainRate = sum * 100
        print "retain " + str(sum * 100) +"% of the variance"

    TrainData, TrainLabel = prepareData(Data, Label, train)
    TestData, TestLabel = prepareData(Data, Label, test)

    print 'Logistic Regression With Dimension Reduced to ', PCA_K
    clf = LogisticRegression()
    clf.fit(TrainData, TrainLabel)
    result = clf.predict(TestData)
    N = TestLabel.shape[0]
    correct = 0.0
    CorrectNum = []
    for i in range(10):
        CorrectNum.append(0)
    for i in range(N):
        #print result[i], " - ", TestLabel[i]
        if(result[i] == TestLabel[i]):
            CorrectNum[int(result[i])] += 1
            correct += 1
    Correctness = correct/N * 100
    print "Accuracy: ", Correctness
    for i in range(10):
        print "digit ", i , ": ", CorrectNum[i], "/200"
    return RetainRate, Correctness, CorrectNum

def testWithKNN(labels, vectors, testset, trainset, PCA_K=0):
    Label = array(toFloatList(labels))
    OriginData = array(toFloatList(vectors)).reshape(4000,784)
    TestNum = len(testset)
    test = array(toFloatList(testset)).reshape(TestNum/2,2) - 1

    TrainNum = len(trainset)
    train = array(toFloatList(trainset)).reshape(TrainNum/2,2) - 1
    OriginData = normalization(OriginData)
    Data = OriginData
    RetainRate = 100
    if(PCA_K != 0):
        k = PCA_K
        pca = PCA(n_components=k)
        Data = pca.fit_transform(OriginData)
        sum = 0
        for i in range(k):
            sum += pca.explained_variance_ratio_[i]
        RetainRate = sum * 100
        print "retain " + str(sum * 100) +"% of the variance"
    print '1NN With Dimension Reduced to ', PCA_K
    TrainData, TrainLabel = prepareData(Data, Label, train)
    TestData, TestLabel = prepareData(Data, Label, test)

    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(TrainData, TrainLabel)
    result = knn.predict(TestData)
    N = TestLabel.shape[0]
    correct = 0.0
    CorrectNum = []
    for i in range(10):
        CorrectNum.append(0)
    for i in range(N):
        #print result[i], " - ", TestLabel[i]
        if(result[i] == TestLabel[i]):
            CorrectNum[int(result[i])] += 1
            correct += 1
    Correctness = correct/N * 100
    print "Accuracy: ", Correctness
    for i in range(10):
        print "digit ", i , ": ", CorrectNum[i], "/200"
    return RetainRate, Correctness, CorrectNum


from test import *
from numpy import *

def outputDataForCrossValidation(labels, vectors, testset, trainset, PCA_K=0):
    Label = array(toFloatList(labels))
    OriginData = array(toFloatList(vectors)).reshape(4000,784)
    TestNum = len(testset)
    test = array(toFloatList(testset)).reshape(TestNum/2,2) - 1
    TrainNum = len(trainset)
    train = array(toFloatList(trainset)).reshape(TrainNum/2,2) - 1

    OriginData = normalization(OriginData)

    Data = OriginData
    RetainRate = 100
    if(PCA_K != 0):
        k = PCA_K
        pca = PCA(n_components=k)
        Data = pca.fit_transform(OriginData)
        sum = 0
        for i in range(k):
            sum += pca.explained_variance_ratio_[i]
        RetainRate = sum * 100
        print "retain " + str(RetainRate) +"% of the variance"

    TrainData, TrainLabel = prepareData(Data, Label, train)
    TestData, TestLabel = prepareData(Data, Label, test)

    outputTrainingData(TrainData, TrainLabel, PCA_K)

def accuracyAndDimension(labels, vectors, testset, trainset):
    TrialResult = zeros((5,1,6))
    # ------------------------------------------------------------------------------------------------
    # Trail 1
    # ------------------------------------------------------------------------------------------------
    print "Trial 1"
    TestC_1 = [2, 8, 2, 2, 2, 2]
    TestGamma_1 = [0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125]
    TestK = [20, 50, 84, 150, 300, 400]
    #drawCombinedDiagram(labels, vectors, testset, trainset, 'Trial 1', TestK, TestC, TestGamma)
    Result_1 = getTrialAccuracyResult(labels, vectors, testset, trainset, TestK, TestC_1, TestGamma_1)
    #print TrialResult
    #print "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    TrialResult += Result_1
    print TrialResult
    # ------------------------------------------------------------------------------------------------
    # Trail 2
    print "Trial 2"
    TestC_2 = [2, 8, 8, 2, 2, 2]
    TestGamma_2 = [0.03125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125]
    #drawCombinedDiagram(labels, vectors, trainset, testset, 'Trial 2', TestK, TestC, TestGamma)
    Result_2 = getTrialAccuracyResult(labels, vectors, trainset, testset, TestK, TestC_2, TestGamma_2)
    #print TrialResult
    #print "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    TrialResult += Result_2
    print TrialResult
    # ------------------------------------------------------------------------------------------------
    #Random Trial
    '''
    print "Trial 3"
    #testWithSVM(labels, vectors, random_testset, random_trainset, Kernel='linear', PCA_K=20)
    TestC_3 = [2, 8, 8, 2, 2, 2]
    TestGamma_3 = [0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125]
    #drawCombinedDiagram(labels, vectors, random_testset, random_trainset, 'Random Trial', TestK, TestC, TestGamma)
    Result_3 = getTrialResult(labels, vectors, random_testset, random_trainset, TestK, TestC_3, TestGamma_3)
    TrialResult += Result_3
    '''
    #print TrialResult
    #print "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    # ------------------------------------------------------------------------------------------------
    TrialResult = TrialResult/2.0
    print TrialResult
    print "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    C_SVM_LINEAR = TrialResult[0][0].tolist()
    C_SVM_POLY = TrialResult[1][0].tolist()
    C_SVM_RBF = TrialResult[2][0].tolist()
    C_LR = TrialResult[3][0].tolist()
    C_1NN = TrialResult[4][0].tolist()

    plt.figure(1)
    plt.title("Average Accuracy for all trials")
    plt.xlabel("dimension")
    plt.ylabel("accuracy(%)")
    plt.axis([0,450,83,97])
    plt.grid(True)
    plt.plot(TestK, C_SVM_LINEAR, 'ro--')
    plt.plot(TestK, C_SVM_POLY, 'b<--')
    plt.plot(TestK, C_SVM_RBF, 'gs--')
    plt.plot(TestK, C_LR, 'yd--')
    plt.plot(TestK, C_1NN, 'co--')
    P = 4
    plt.annotate("svm linear", xy=(TestK[P], C_SVM_LINEAR[P]),xytext=(TestK[P]-18, C_SVM_LINEAR[P]+0.2))
    plt.annotate("svm poly", xy=(TestK[P], C_SVM_POLY[P]),xytext=(TestK[P]-18, C_SVM_POLY[P]+0.5))
    plt.annotate("svm rbf", xy=(TestK[P], C_SVM_RBF[P]),xytext=(TestK[P]-18, C_SVM_RBF[P]+0.2))
    plt.annotate("lr", xy=(TestK[P], C_LR[P]),xytext=(TestK[P]-18, C_LR[P]+0.2))
    plt.annotate("1nn", xy=(TestK[P], C_1NN[P]),xytext=(TestK[P]-18, C_1NN[P]+0.2))
    plt.show()

def recognitionNumberAndDigit(labels, vectors, testset, trainset):
    TrialResult = zeros((5,1,10))
    Digit = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # ------------------------------------------------------------------------------------------------
    # Trail 1
    # ------------------------------------------------------------------------------------------------
    print "Trial 1"
    TestC_1 = [2, 8, 2, 2, 2, 2]
    TestGamma_1 = [0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125]
    TestK = [20, 50, 84, 150, 300, 400]
    #drawCombinedDiagram(labels, vectors, testset, trainset, 'Trial 1', TestK, TestC, TestGamma)
    Result_1 = getTrialDigitResult(labels, vectors, testset, trainset, TestK, TestC_1, TestGamma_1)
    #print TrialResult
    #print "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    TrialResult += Result_1
    print TrialResult
    # ------------------------------------------------------------------------------------------------
    # Trail 2

    print "Trial 2"
    TestC_2 = [2, 8, 8, 2, 2, 2]
    TestGamma_2 = [0.03125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125]
    #drawCombinedDiagram(labels, vectors, trainset, testset, 'Trial 2', TestK, TestC, TestGamma)
    Result_2 = getTrialDigitResult(labels, vectors, trainset, testset, TestK, TestC_2, TestGamma_2)
    #print TrialResult
    #print "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    TrialResult += Result_2
    print TrialResult
    # ------------------------------------------------------------------------------------------------

    TrialResult = TrialResult/2.0
    TrialResult = TrialResult.round(2)
    print TrialResult
    print "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    C_SVM_LINEAR = TrialResult[0][0].tolist()
    C_SVM_POLY = TrialResult[1][0].tolist()
    C_SVM_RBF = TrialResult[2][0].tolist()
    C_LR = TrialResult[3][0].tolist()
    C_1NN = TrialResult[4][0].tolist()

    plt.figure(1)
    plt.title("Average Recognition Number for all digits")
    plt.xlabel("digit")
    plt.ylabel("num of correctly recognition")
    plt.axis([-1,10,140,205])
    plt.grid(True)
    plt.plot(Digit, C_SVM_LINEAR, 'ro--')
    plt.plot(Digit, C_SVM_POLY, 'b<--')
    plt.plot(Digit, C_SVM_RBF, 'gs--')
    plt.plot(Digit, C_LR, 'yd--')
    plt.plot(Digit, C_1NN, 'co--')

    P = 9
    plt.annotate("svm linear", xy=(Digit[P], C_SVM_LINEAR[P]),xytext=(Digit[P]+0.1, C_SVM_LINEAR[P]))
    plt.annotate("svm poly", xy=(Digit[P], C_SVM_POLY[P]),xytext=(Digit[P]+0.1, C_SVM_POLY[P]))
    plt.annotate("svm rbf", xy=(Digit[P], C_SVM_RBF[P]),xytext=(Digit[P]+0.1, C_SVM_RBF[P]))
    plt.annotate("lr", xy=(Digit[P], C_LR[P]),xytext=(Digit[P]+0.1, C_LR[P]))
    plt.annotate("1nn", xy=(Digit[P], C_1NN[P]),xytext=(Digit[P]+0.1, C_1NN[P]))

    plt.show()

def getTestResult(vectors, labels, testset, trainset, PCA_K=0, C=1, GAMMA=0.0, method=''):
    Label = array(toFloatList(labels))
    OriginData = array(toFloatList(vectors)).reshape(4000,784)
    TestNum = len(testset)
    test = array(toFloatList(testset)).reshape(TestNum/2,2) - 1
    TrainNum = len(trainset)
    train = array(toFloatList(trainset)).reshape(TrainNum/2,2) - 1

    OriginData = normalization(OriginData)

    Data = OriginData
    RetainRate = 100
    if(PCA_K != 0):
        k = PCA_K
        pca = PCA(n_components=k)
        Data = pca.fit_transform(OriginData)
        sum = 0
        for i in range(k):
            sum += pca.explained_variance_ratio_[i]
        RetainRate = sum * 100
        #print "retain " + str(RetainRate) +"% of the variance"

    TrainData, TrainLabel = prepareData(Data, Label, train)
    TestData, TestLabel = prepareData(Data, Label, test)

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

def getTrialAccuracyResult(labels, vectors, testset, trainset, TestK, TestC, TestGamma):
    N = len(TestK)
    C_SVM_LINEAR = []
    C_SVM_POLY = []
    C_SVM_RBF = []
    C_LR = []
    C_1NN = []
    AccuracyResult = zeros((5,1,N))
    for i in range(N):
        print "Test ", i
        print "--------------------------------------------------------------------------------------------------"
        print "K = ", TestK[i]
        Rate, Correct, C_NUM = testWithSVM(labels, vectors, testset, trainset, Kernel='linear', PCA_K=TestK[i])
        C_SVM_LINEAR.append(Correct)
        print "--------------------------------------------------------------------------------------------------"
        Rate, Correct, C_NUM = testWithSVM(labels, vectors, testset, trainset, Kernel='poly', PCA_K=TestK[i])
        C_SVM_POLY.append(Correct)
        print "--------------------------------------------------------------------------------------------------"
        Rate, Correct, C_NUM = testWithSVM(labels, vectors, testset, trainset, Kernel='rbf', C=TestC[i], gamma=TestGamma[i], PCA_K=TestK[i])
        C_SVM_RBF.append(Correct)
        print "--------------------------------------------------------------------------------------------------"
        Rate, Correct, C_NUM = testWithLR(labels, vectors, testset, trainset, PCA_K=TestK[i])
        C_LR.append(Correct)
        print "--------------------------------------------------------------------------------------------------"
        Rate, Correct, C_NUM = testWithKNN(labels, vectors, testset, trainset, PCA_K=TestK[i])
        C_1NN.append(Correct)
        print "--------------------------------------------------------------------------------------------------"
    AccuracyResult[0][0] = array(C_SVM_LINEAR)
    AccuracyResult[1][0] = array(C_SVM_POLY)
    AccuracyResult[2][0] = array(C_SVM_RBF)
    AccuracyResult[3][0] = array(C_LR)
    AccuracyResult[4][0] = array(C_1NN)
    #print "Return:"
    #print AccuracyResult
    #print "\n\n"
    return AccuracyResult

def getTrialDigitResult(labels, vectors, testset, trainset, TestK, TestC, TestGamma):
    N = len(TestK)
    Num_linear = zeros((1, 10))
    Num_poly = zeros((1, 10))
    Num_rbf = zeros((1, 10))
    Num_lr = zeros((1, 10))
    Num_1nn = zeros((1, 10))
    RecognitionNum = zeros((5,1,10))
    for i in range(len(TestK)):
        print "Test ", i
        print "--------------------------------------------------------------------------------------------------"
        print "K = ", TestK[i]
        Rate, Correct, C_NUM = testWithSVM(labels, vectors, testset, trainset, Kernel='linear', PCA_K=TestK[i])
        Num_linear += array(C_NUM)
        print "--------------------------------------------------------------------------------------------------"
        Rate, Correct, C_NUM = testWithSVM(labels, vectors, testset, trainset, Kernel='poly', PCA_K=TestK[i])
        Num_poly += array(C_NUM)
        print "--------------------------------------------------------------------------------------------------"
        Rate, Correct, C_NUM = testWithSVM(labels, vectors, testset, trainset, Kernel='rbf', C=TestC[i], gamma=TestGamma[i], PCA_K=TestK[i])
        Num_rbf += array(C_NUM)
        print "--------------------------------------------------------------------------------------------------"
        Rate, Correct, C_NUM = testWithLR(labels, vectors, testset, trainset, PCA_K=TestK[i])
        Num_lr += array(C_NUM)
        print "--------------------------------------------------------------------------------------------------"
        Rate, Correct, C_NUM = testWithKNN(labels, vectors, testset, trainset, PCA_K=TestK[i])
        Num_1nn += array(C_NUM)
    RecognitionNum[0][0] = array(Num_linear)
    RecognitionNum[1][0] = array(Num_poly)
    RecognitionNum[2][0] = array(Num_rbf)
    RecognitionNum[3][0] = array(Num_lr)
    RecognitionNum[4][0] = array(Num_1nn)
    return RecognitionNum/N