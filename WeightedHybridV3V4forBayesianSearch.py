# import stuff
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
#-----QUI PASSO I RECOMMENDERS GIA' FITTATI NELL'INIZIALIZZAZIONE---------
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
import sys

import os

sys.path.insert(0, '../Lab/')

from Base.BaseRecommender import BaseRecommender
from operator import add
from Base.DataIO import DataIO
import numpy as np
from Base.NonPersonalizedRecommender import TopPop


from Base.Recommender_utils import check_matrix, similarityMatrixTopK
from Base.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender



#------STUFF FOR SUBSTITUTE INITIALIZATION AND FIT PARAMETERS

def createWeights(w0,w1,w2,w3,w4,w5,w6,w7):
    weights=[]
    weights.append(w0)
    weights.append(w1)
    weights.append(w2)
    weights.append(w3)
    weights.append(w4)
    weights.append(w5)
    weights.append(w6)
    weights.append(w7)
    return weights
#-----------------------
class WeightedHybridV3V4forBayesianSearch(BaseRecommender):
    RECOMMENDER_NAME = "WeightedHybridV3V4forBayesianSearch"

    # initialization
    # recs = [rec1, rec2, ...]
    # inits = [recsatt1, recsatt2, ..]
    # recsatt = {
    #     paramname = "paramvalue"
    # }
    def __init__(self, URM_train,ICM_all,recs):

        super(WeightedHybridV3V4forBayesianSearch, self).__init__(URM_train)
        
        self.recs = recs
        #self.top = TopPop(URM_train)
        self.URM_train = URM_train
        
        self.weights = []

       

    def fit(self,w0,w1,w2,w3,w4,w5,w6,w7):
        print("--------FITTING...-------")
        #self.top.fit()
        self.weights=createWeights(w0,w1,w2,w3,w4,w5,w6,w7)

    # qui calcolo score per ogni metodo e sommo e tutte quelle belle cose
    # questa funzione è chiamat dentro reccommend e ritorna lo score degli items
    def _compute_item_score(self, user_id_array, items_to_compute=None):
        scores = [self.recs[i]._compute_item_score(user_id_array) for i in range(8) ]
        
#         item_weights_0 = self.recs[0]._compute_item_score(user_id_array)
#         item_weights_1 = self.recs[1]._compute_item_score(user_id_array)
#         item_weights_2 = self.recs[2]._compute_item_score(user_id_array)
#         item_weights_3 = self.recs[3]._compute_item_score(user_id_array)
#         item_weights_4 = self.recs[4]._compute_item_score(user_id_array)
#         item_weights_5 = self.recs[5]._compute_item_score(user_id_array)
#         item_weights_4 = self.recs[6]._compute_item_score(user_id_array)
#         item_weights_5 = self.recs[7]._compute_item_score(user_id_array)
        result = 0
        for i, s in enumerate(scores):
            result += s * self.weights[i]
#         item_weights = item_weights_0*self.weights[0] + item_weights_1*self.weights[1] + item_weights_2*self.weights[2] + item_weights_3*self.weights[3] + item_weights_4*self.weights[4] + item_weights_5*self.weights[5] 

        return result

