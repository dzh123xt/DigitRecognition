__author__ = 'dengzhihong'


from numpy import *
import numpy as np
import src.Methods.TestMethods as test_methods
import random
import matplotlib.pyplot as plt

def drawDiagramOfSvmRbf(labels, vectors, testset, trainset):
    TestC = [2, 8, 2, 2, 2]
    TestGamma = [0.0078125, 0.0078125, 0.0078125, 0.0078125, 0.0078125]
    TestK = [20, 50, 84, 150, 400]
    Correctness = []
    RetainRate = []
    for i in range(len(TestK)):
        print "Test ", i
        print "--------------------------------------------------------------------------------------------------"
        print "K = ", TestK[i], "  Gamma = ", TestGamma[i], "  C = ", TestC[i]
        Rate, Correct = test_methods.testWithSVM(labels, vectors, testset, trainset, Kernel='rbf', C=TestC[i], gamma=TestGamma[i], PCA_K=TestK[i])
        RetainRate.append(round(Rate,2))
        Correctness.append(Correct)
        print "--------------------------------------------------------------------------------------------------"
    plt.figure(1)
    plt.title("Relation between accuracy and dimension, kernel = rbf")
    plt.xlabel("dimension")
    plt.ylabel("accuracy(%)")
    plt.axis([0,500,95,97])
    plt.grid(True)
    plt.plot(TestK, Correctness, 'bo--')
    for i in range(len(TestK)):
        plt.annotate("Retain " + str(RetainRate[i]) + "%", xy=(TestK[i],Correctness[i]),xytext=(TestK[i]-10,Correctness[i]+0.1))
    plt.show()

def drawDiagramOfSvmLinear(labels, vectors, testset, trainset):
    TestK = [20, 50, 84, 150, 400]
    Correctness = []
    RetainRate = []
    for i in range(len(TestK)):
        print "Test ", i
        print "--------------------------------------------------------------------------------------------------"
        print "K = ", TestK[i]
        Rate, Correct = test_methods.testWithSVM(labels, vectors, testset, trainset, Kernel='linear', PCA_K=TestK[i])
        RetainRate.append(round(Rate,2))
        Correctness.append(Correct)
        print "--------------------------------------------------------------------------------------------------"
    plt.figure(1)
    plt.title("Relation between accuracy and dimension, kernel = linear")
    plt.xlabel("dimension")
    plt.ylabel("accuracy(%)")
    plt.axis([0,450,84,90])
    plt.grid(True)
    plt.plot(TestK, Correctness, 'ro--')
    for i in range(len(TestK)):
        plt.annotate("Retain " + str(RetainRate[i]) + "%", xy=(TestK[i],Correctness[i]),xytext=(TestK[i]-10,Correctness[i]+0.1))
    plt.show()

def drawDiagramOfSvmPoly(labels, vectors, testset, trainset):
    TestK = [20, 50, 84, 150, 400]
    Correctness = []
    RetainRate = []
    for i in range(len(TestK)):
        print "Test ", i
        print "--------------------------------------------------------------------------------------------------"
        print "K = ", TestK[i]
        Rate, Correct = test_methods.testWithSVM(labels, vectors, testset, trainset, Kernel='poly', PCA_K=TestK[i])
        RetainRate.append(round(Rate,2))
        Correctness.append(Correct)
        print "--------------------------------------------------------------------------------------------------"
    plt.figure(1)
    plt.title("Relation between accuracy and dimension, kernel = poly")
    plt.xlabel("dimension(%)")
    plt.ylabel("accuracy(%)")
    plt.axis([0,500,91,97])
    plt.grid(True)
    plt.plot(TestK, Correctness, 'ro--')
    for i in range(len(TestK)):
        plt.annotate("Retain " + str(RetainRate[i]) + "%", xy=(TestK[i],Correctness[i]),xytext=(TestK[i]-16,Correctness[i]+0.2))
    plt.show()

def drawDiagramOfKNN(labels, vectors, testset, trainset):
    TestK = [20, 50, 84, 150, 400]
    Correctness = []
    RetainRate = []
    for i in range(len(TestK)):
        print "Test ", i
        print "--------------------------------------------------------------------------------------------------"
        print "K = ", TestK[i]
        Rate, Correct = test_methods.testWithKNN(labels, vectors, testset, trainset, PCA_K=TestK[i])
        RetainRate.append(round(Rate,2))
        Correctness.append(Correct)
        print "--------------------------------------------------------------------------------------------------"
    plt.figure(1)
    plt.title("Relation between accuracy and dimension, 1NN")
    plt.xlabel("dimension")
    plt.ylabel("accuracy(%)")
    plt.axis([0,500,91,93])
    plt.grid(True)
    plt.plot(TestK, Correctness, 'ro--')
    for i in range(len(TestK)):
        plt.annotate("Retain " + str(RetainRate[i]) + "%", xy=(TestK[i],Correctness[i]),xytext=(TestK[i]-16,Correctness[i]+0.05))
    plt.show()

def drawDiagramOfLr(labels, vectors, testset, trainset):
    TestK = [20, 50, 84, 150, 400]
    Correctness = []
    RetainRate = []
    for i in range(len(TestK)):
        print "Test ", i
        print "--------------------------------------------------------------------------------------------------"
        print "K = ", TestK[i]
        Rate, Correct = test_methods.testWithLR(labels, vectors, testset, trainset, PCA_K=TestK[i])
        RetainRate.append(round(Rate,2))
        Correctness.append(Correct)
        print "--------------------------------------------------------------------------------------------------"
    plt.figure(1)
    plt.title("Relation between accuracy and dimension, Logistic Regression")
    plt.xlabel("dimension")
    plt.ylabel("accuracy(%)")
    plt.axis([0,500, 85, 89])
    plt.grid(True)
    plt.plot(TestK, Correctness, 'ro--')
    for i in range(len(TestK)):
        plt.annotate("Retain " + str(RetainRate[i]) + "%", xy=(TestK[i],Correctness[i]),xytext=(TestK[i]-16,Correctness[i]+0.05))
    plt.show()

def drawCombinedDiagram(labels, vectors, testset, trainset, title, TestK, TestC, TestGamma):
    N = len(TestK)
    C_SVM_LINEAR = []
    C_SVM_POLY = []
    C_SVM_RBF = []
    C_LR = []
    C_1NN = []
    RetainRate = zeros((1, 10))
    Num_linear = zeros((1, 10))
    Num_poly = zeros((1, 10))
    Num_rbf = zeros((1, 10))
    Num_lr = zeros((1, 10))
    Num_1nn = zeros((1, 10))
    for i in range(len(TestK)):
        print "Test ", i
        print "--------------------------------------------------------------------------------------------------"
        print "K = ", TestK[i]
        Rate, Correct, C_NUM = test_methods.testWithSVM(labels, vectors, testset, trainset, Kernel='linear', PCA_K=TestK[i])
        C_SVM_LINEAR.append(Correct)
        Num_linear += array(C_NUM)
        print "--------------------------------------------------------------------------------------------------"
        Rate, Correct, C_NUM = test_methods.testWithSVM(labels, vectors, testset, trainset, Kernel='poly', PCA_K=TestK[i])
        C_SVM_POLY.append(Correct)
        Num_poly += array(C_NUM)
        print "--------------------------------------------------------------------------------------------------"
        Rate, Correct, C_NUM = test_methods.testWithSVM(labels, vectors, testset, trainset, Kernel='rbf', C=TestC[i], gamma=TestGamma[i], PCA_K=TestK[i])
        C_SVM_RBF.append(Correct)
        Num_rbf += array(C_NUM)
        print "--------------------------------------------------------------------------------------------------"
        Rate, Correct, C_NUM = test_methods.testWithLR(labels, vectors, testset, trainset, PCA_K=TestK[i])
        C_LR.append(Correct)
        Num_lr += array(C_NUM)
        print "--------------------------------------------------------------------------------------------------"
        Rate, Correct, C_NUM = test_methods.testWithKNN(labels, vectors, testset, trainset, PCA_K=TestK[i])
        C_1NN.append(Correct)
        Num_1nn += array(C_NUM)
        print "--------------------------------------------------------------------------------------------------"
    print "Average Correct Num of different Digit"
    print "SVM Linear: ", (Num_linear/N).round(2)
    print "SVM Poly: ", (Num_poly/N).round(2)
    print "SVM Rbf: ", (Num_rbf/N).round(2)
    print "LR: ", (Num_lr/N).round(2)
    print "1NN: ", (Num_1nn/N).round(2)
    print "--------------------------------------------"
    plt.figure(1)
    plt.title(title)
    plt.xlabel("dimension")
    plt.ylabel("accuracy(%)")
    plt.axis([0,450,83,97])
    plt.grid(True)
    plt.plot(TestK, C_SVM_LINEAR, 'ro--')
    plt.plot(TestK, C_SVM_POLY, 'b<--')
    plt.plot(TestK, C_SVM_RBF, 'gs--')
    plt.plot(TestK, C_LR, 'yd--')
    plt.plot(TestK, C_1NN, 'co--')
    P = 5
    plt.annotate("svm linear", xy=(TestK[P], C_SVM_LINEAR[P]),xytext=(TestK[P]-18, C_SVM_LINEAR[P]+0.2))
    plt.annotate("svm poly", xy=(TestK[P], C_SVM_POLY[P]),xytext=(TestK[P]-18, C_SVM_POLY[P]+0.5))
    plt.annotate("svm rbf", xy=(TestK[P], C_SVM_RBF[P]),xytext=(TestK[P]-18, C_SVM_RBF[P]+0.2))
    plt.annotate("lr", xy=(TestK[P], C_LR[P]),xytext=(TestK[P]-18, C_LR[P]+0.2))
    plt.annotate("1nn", xy=(TestK[P], C_1NN[P]),xytext=(TestK[P]-18, C_1NN[P]+0.2))
    plt.show()

