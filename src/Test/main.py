__author__ = 'dengzhihong'

from src.Methods.TestMethods import *

if __name__ == '__main__':
    print "Read Data From txt"
    labels = str(open('../Data/digits4000_digits_labels.txt', 'r').read()).split()
    vectors = str(open('../Data/digits4000_digits_vec.txt', 'r').read()).split()
    testset = str(open('../Data/digits4000_testset.txt', 'r').read()).split()
    trainset = str(open('../Data/digits4000_trainset.txt', 'r').read()).split()

    #outputDataForCrossValidation(labels, vectors, testset, trainset, PCA_K=200)

    #accuracyAndDimension(labels, vectors, testset, trainset)

    #recognitionNumberAndDigit(labels, vectors, testset, trainset)

    #testWithSVM(labels, vectors, testset, trainset, Kernel='rbf', C=8, gamma=0.0078125, PCA_K=50)

    TestK = [20, 50, 84, 150, 300, 400]
    TestC = [2, 8, 2, 2, 2, 2]
    TestGamma = [0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125]
    TestC_2 = [2, 8, 8, 2, 2, 2]
    TestGamma_2 = [0.03125, 0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125]
    Method = ['svm_linear', 'svm_poly', 'svm_rbf', 'lr', '1nn']
    #for i in range(len(Method)):
    print Method
    for i in range(len(TestK)):
        print "Dimension: ", TestK[i], "   ",
        for j in range(len(Method)):
            #print "Method: ", Method[j]
            #trial 1
            accuracy = testWithChallenge(vectors, labels, challenge_vectors, challenge_labels, trainset, TestK[i], TestC[i], TestGamma[i], Method[j])
            #trial 2
            #accuracy = testWithChallenge(vectors, labels, challenge_vectors, challenge_labels, trainset,
             #                           PCA_K=TestK[i], C=TestC[i], GAMMA=TestGamma[i], method=Method[j])

            #accuracy = getTestResult(vectors, labels, testset, trainset,PCA_K=TestK[i], C=TestC[i], GAMMA=TestGamma[i], method=Method[j])
            #accuracy = getTestResult(vectors, labels, trainset, testset,PCA_K=TestK[i], C=TestC_2[i], GAMMA=TestGamma_2[i], method=Method[j])
            print accuracy, '%  ',
        print "\n"
