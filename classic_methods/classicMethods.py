import numpy as np
import cv2
# from skimage.segmentation import *
from sklearn import *
import copy
from joblib import Parallel, delayed
#from sklearn.naive_bayes import MultinomialNB, ComplementNB
from timeit import default_timer as timer

class nonNNAlgo:
    
    def __init__(self, train_X, train_Y, test_X, test_Y, p1, p2, segments_list = None):
        self.train_X = train_X
        if train_Y is not None:
            self.train_Y = train_Y.ravel()
        self.test_X = test_X
        if test_Y is not None:
            self.test_Y = test_Y.ravel()
        self.segments_list = segments_list

        self.bayes = naive_bayes.GaussianNB()
        if p1 is not None:
            self.gmm0 = mixture.GaussianMixture(n_components=p1[0], covariance_type='full')
            self.gmm1 = mixture.GaussianMixture(n_components=p1[1], covariance_type='full')
        if p2 is not None:
            self.clf = ensemble.RandomForestClassifier(n_estimators=p2[0], max_depth=p2[1])
        
    # random forest
    def randomForestSeg(self, train_idx = False, test_idx = False, clf = None, pMetrics=False):
        if train_idx:
            
            self.clf.fit(self.train_X, self.train_Y)
            
        if test_idx:  
            start = timer() 
            test_Y_pred = self.clf.predict(self.test_X)
            test_time = timer() - start
            #print('[rf]', )
            
            if pMetrics is True:
                iou, dice = self.calMetric(test_Y_pred,self.test_Y)
                #print('iou', iou, 'dice', dice)
            
                return test_Y_pred, iou, dice, test_time
            return test_Y_pred
    
    # GMM
    def GaussianMixtureModelSeg(self, train_idx = False, test_idx = False, pMetrics=False):
        
        if train_idx:

            idx0 = np.argwhere(self.train_Y == 0) # background
            idx1 = np.argwhere(self.train_Y == 1) # tool

            self.gmm0.fit(np.squeeze(self.train_X[idx0]))
            self.gmm1.fit(np.squeeze(self.train_X[idx1]))
        
        if test_idx:
            # previous code: 0.9 * gmm1 results
            start = timer()
            test_Y_pred = 1.0 * (self.gmm0.score_samples(self.test_X) < self.gmm1.score_samples(self.test_X))
            test_time = timer() - start
            #print('[gmm]', timer() - start)

            if pMetrics is True:
                iou, dice = self.calMetric(test_Y_pred,self.test_Y)
                #print('iou', iou, 'dice', dice)
            
                return test_Y_pred, iou, dice, test_time
            return test_Y_pred
    
    # naive Bayesian (Gaussian)
    def naiveBayesianSeg(self, train_idx = False, cv_idx = False, test_idx = False, pMetrics=False):
        if train_idx:
            self.bayes.fit(self.train_X, self.train_Y) # classes=np.unique(self.train_Y)
        
        if test_idx:
            start = timer()
            #test_Y_pred = self.bayes.predict(self.test_X)
            test_Y_pred_temp = self.bayes.predict_proba(self.test_X)
            test_Y_pred = 1 * (test_Y_pred_temp[:,1]>0.2)
            #test_Y_pred2 = 1 * (test_Y_pred_temp[:,1]<1)
#            test_Y_pred = test_Y_pred1 * test_Y_pred2
            test_time = timer() - start
            #print('[bayes]', timer() - start)
            
            if pMetrics is True:
                iou, dice = self.calMetric(test_Y_pred,self.test_Y)
                #print('iou', iou, 'dice', dice)
            
                return test_Y_pred, iou, dice, test_time
            return test_Y_pred
    
    def calMetric(self,Y1,Y2):

        width = 520
        height = 475
        num_pixel = (width-7)*(height-7)
        num_image = int(len(Y1)/num_pixel)

        metrics = Parallel(n_jobs=-1)(delayed(self.metricLoop)(Y1[i*num_pixel:(i+1)*num_pixel],\
Y2[i*num_pixel:(i+1)*num_pixel]) for i in range(0,num_image))
        metrics = np.concatenate(metrics,axis=0)
        
        return np.mean(metrics[:,0]), np.mean(metrics[:,1])

    def metricLoop(self,y1,y2):
        
        overlap = np.sum( y1 * y2 )
        y1 = np.sum(y1)
        y2 = np.sum(y2)
        union = y1+y2-overlap
        
        return np.array([[overlap/union, 2*overlap/(y1+y2)]])
