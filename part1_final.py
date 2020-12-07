# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 15:11:32 2020

@author: Chenyun Yu

"""

import csv
import numpy as np
import matplotlib.pyplot as plt

class AveragePerceptron:
    
    def __init__(self, method = "holdout"):
        """
        init func.
        para.: 
            method: the method to manage data
        ret.:
        """
        self.data_method = method
        self.average_correct_rate = 0
        self.online_correct_rate = 0
        
        if self.data_method is "holdout":
            #data use for tain
            self.train_x = list()
            self.train_y = list()
        
            #data use for validation
            self.val_x = list()
            self.val_y = list()
            
            #data used for test
            self.test_x = list()
            self.test_y = list()
            
            #online weight
            self.online_weights = np.zeros(0)
            self.online_weights_recorder = list()
            
            #average weight
            self.average_weights = np.zeros(0)
            self.average_weights_recorder = list()
            
    
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
        
    def training(self, maxIter = 100):
        """
        the traing process of kernelized perceptron
        para.: 
            maxIter: max iteration
        ret.: 
            None
        """
    
        self.online_weights = np.zeros(self.train_x.shape[1])
        self.average_weights = np.zeros(self.train_x.shape[1])
        example_counter = 1
     
        for _ in range(maxIter):
            for idx, xi in enumerate(self.train_x):
                if self.train_y[idx] * np.dot(xi,self.online_weights) <= 0:
                    self.online_weights += self.train_y[idx] * xi
                
                self.average_weights = (example_counter * self.average_weights + self.online_weights)/(example_counter + 1)
                example_counter = example_counter + 1
            
            #print(self.online_weights)  
            self.average_weights_recorder.append(self.average_weights.copy())
            self.online_weights_recorder.append(self.online_weights.copy())
                
                
                
                
    def __predictCorrectRate(self, weight, data_x, data_y):
        
        """
        helper function for computing the prediction result
        para.: 
            weight: weights
            data_x: conditon
            data_y; result
        ret.: 
            accuracy
        """
        c = 0
        w = 0
        for idx, xi in enumerate(data_x):
            if data_y[idx] * np.dot(xi,weight) > 0:
                c += 1
            else:
                w += 1
            
        return c / (c + w)
        
    def averageCorrectRate(self):
        self.average_correct_rate = self.__predictCorrectRate(self.average_weights, self.val_x, self.val_y)
        return self.average_correct_rate
        
    def onlineCorrectRate(self):
        self.online_correct_rate = self.__predictCorrectRate(self.online_weights, self.val_x, self.val_y)
        return self.online_correct_rate
    
    def onlinePlot(self):
        valRates = list()
        trainRates = list()
        for val in self.online_weights_recorder:
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
        plt.title("online")
        plt.show()
        
    def averagePlot(self):
        valRates = list()
        trainRates = list()
        for val in self.average_weights_recorder:
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
        
        
    def doit(self):
        self.loadTrainDataFrom()
        self.loadValDataFrom()
        self.training()
        #self.onlineCorrectRate()
        #self.averageCorrectRate()
        self.onlinePlot()
        self.averagePlot()
        

if __name__ == "__main__":
    ap = AveragePerceptron()
    ap.doit()
    
    
    