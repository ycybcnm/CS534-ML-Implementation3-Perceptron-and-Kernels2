# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 15:11:32 2020

@author: Chenyun Yu

"""

import csv
import numpy as np
import matplotlib.pyplot as plt
#import time as time

class  KernelizedPerceptron:
    
    def __init__(self, method = "holdout", p = 2):
        """
        init func.
        para.: 
            method: the method to manage data
            p: the k -polynomial for kernel
        ret.:
        """
        # method for data
        self.data_method = method
        # the final accuracy for validation and prediction
        self.correct_rate = 0
        # k -polynomial
        self.poly = p
        # max training accuracy
        self.maxTrain = 0
        # max validation accuracy
        self.maxval = 0
        # max iteration, default is 100
        self.maxIter = 100
        
        if self.data_method == "holdout":
            """data use for tain"""
            self.train_x = list()
            self.train_y = list()
        
            """data use for validation"""
            self.val_x = list()
            self.val_y = list()
                
            """alpha"""
            self.alpha = np.zeros(0)
            self.alpha_recorder = list()
            
    def setMaxiter(self, i):
        """
        set the iteration time
        para.:
            i : itertation time
        ret:.:
            None
        """
        self.maxIter = i 
            
    def loadTrainDataFrom(self, fname_x  = "pa3_train_X.csv", fname_y = "pa3_train_y.csv"):
        """
        load the training data from file
        para.: 
            fanme_x: file where stores the X(condition)
            fanme_y: file where stores the y(result)
        ret.: 
            None
        """
        # load data from file
        dataX = []
        dataY = []
        with open(fname_x) as csvfile:
            trainData_reader  = csv.reader(csvfile, delimiter=" ")
            line_count = 0
            for row in trainData_reader:
                if line_count != 0:
                    line = ",".join(row).split(",")
                    dataX.append(list(map(float, line)))
                line_count += 1
            
        with open(fname_y) as csvfile:
            trainData_reader  = csv.reader(csvfile, delimiter=" ")
            line_count = 0
            for row in trainData_reader:
                if line_count != 0:
                    line = ",".join(row).split(",")
                    dataY.append(list(map(float, line)))
                line_count += 1
                
        self.train_x = np.array(dataX)
        self.train_y = np.array(dataY).reshape( self.train_x.shape[0],)
        
    def loadValDataFrom(self, fname_x  = "pa3_dev_X.csv", fname_y = "pa3_dev_y.csv"):
        """
        load the validation data from file
        para.: 
            fanme_x: file where stores the X(condition)
            fanme_y: file where stores the y(result)
        ret.: 
            None
        """
        # load data from file
        dataX = []
        dataY = []
        with open(fname_x) as csvfile:
            trainData_reader  = csv.reader(csvfile, delimiter=" ")
            line_count = 0
            for row in trainData_reader:
                if line_count != 0:
                    line = ",".join(row).split(",")
                    dataX.append(list(map(float, line)))
                line_count += 1
            
        with open(fname_y) as csvfile:
            trainData_reader  = csv.reader(csvfile, delimiter=" ")
            line_count = 0
            for row in trainData_reader:
                if line_count != 0:
                    line = ",".join(row).split(",")
                    dataY.append(list(map(float, line)))
                line_count += 1
                
        self.val_x = np.array(dataX)
        self.val_y = np.array(dataY).reshape( self.val_x.shape[0],)
        
        
    def __Kernelized(self, x, y):
        """
        helper function for computing the kernel result
        para.: 
            x: xth 
            y: yth
        ret.:
            None
        """
        return np.power((np.matmul(x, y.T) + 1), self.poly)
        
    def training(self):
        """
        the traing process of kernelized perceptron
        para.: 
            None
        ret.: 
            None
        """
        gram_matrix = self.__Kernelized(self.train_x, self.train_x)
        N = len(self.train_x)
        self.alpha = np.zeros(N)
     
        for _ in range(self.maxIter):
            for i in range(N):
                u = np.dot(gram_matrix[i], self.alpha * self.train_y)
                if u * self.train_y[i] <= 0:
                    self.alpha[i] += 1
                    
            self.alpha_recorder.append(self.alpha.copy())
            
    def trainingBatch(self, learning_rate):
        """
        the traing process of batch kernelized perceptron
        para.: 
            None
        ret.: 
            None
        """
        gram_matrix = self.__Kernelized(self.train_x, self.train_x)
        N = len(self.train_x)
        self.alpha = np.zeros(N)
     
        for _ in range(self.maxIter):
            u = np.dot(gram_matrix, self.alpha * self.train_y)
            # (?) I don't think learning rate does any effort, 
            # because other computations are only about sign, not value 
            # but the learning rate is always a positive num, 
            # it will not change the sign.
            u *= learning_rate
            for i in range(N):
                if u[i] * self.train_y[i] <= 0:
                    self.alpha[i] += 1
                    
            self.alpha_recorder.append(self.alpha.copy())
                                             
    def __predictCorrectRate(self, alpha, data_x, data_y):
        """
        helper function for computing the prediction result
        para.: 
            alpha: alpha
            data_x: conditon
            data_y; result
        ret.: 
            accuracy
        """
        c = 0
        w = 0 
        gram_matrix = self.__Kernelized(data_x, self.train_x)
        prediedted_y = np.sign(np.dot(gram_matrix, alpha * self.train_y))
        for idx, _ in enumerate(data_x):
            if prediedted_y[idx] == data_y[idx]:
                c += 1
            else:
                w += 1
                
        return c / (c + w)

        
    def correctRate(self):
        """
        update and return the final accuracy
        para.: 
            Mone
        ret.: 
            final accuracy
        """
        self.correct_rate = self.__predictCorrectRate(self.alpha,self.val_x, self.val_y)
        return self.correct_rate
    
        
    def kernelizedPlot(self):
        """
        plot function
        para.:
            None
        """
        valRates = list()
        trainRates = list()
        for val in self.alpha_recorder:
            trainRate = self.__predictCorrectRate(val, self.train_x, self.train_y)
            valRate = self.__predictCorrectRate(val, self.val_x, self.val_y)
            trainRates.append(trainRate)
            valRates.append(valRate)

        plt.xlabel("iteration")
        plt.ylabel("accuracy")
        plt.scatter(list(range(100)), trainRates, c = 'red', label="train")
        plt.scatter(list(range(100)), valRates, c = 'blue', label="val")
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.title("average")
        plt.show()
        self.maxTrain = max(trainRates)
        self.maxVal = max(valRates)
        
        
    def doit(self):
        self.loadTrainDataFrom()
        self.loadValDataFrom()
        self.training()
        #self.trainingBatch(learning_rate=0.1)
        #self.correctRate()
        self.kernelizedPlot()
        

if __name__ == "__main__":
    
    ap = KernelizedPerceptron(p=1)
    ap.doit()
    
    #timer = list()
    #for i in range(1,21):
        #ts = time.perf_counter()
        #ap = KernelizedPerceptron(p=1)
        #ap.setMaxiter(i * 10)
        #ap.doit()
        #te = time.perf_counter()
        
        #timer.append(te -ts)
    
    