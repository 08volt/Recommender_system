# import stuff

#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
#-----QUI PASSO I RECOMMENDERS GIA' FITTATI NELL'INIZIALIZZAZIONE---------
#----- semplicemnte in init richiede una variabile recs, che coontiene ---
#----- la lista recs=[UserKNNCF, P3Alpha,...] tutti fittati  -------------
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

from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from KNN.UserKNNCFRecommender import UserKNNCFRecommender
from myRecommenders.UserIcmKNNCFRecommender import UserIcmKNNCFRecommender
from myRecommenders.ItemIcmKNNCFRecommender import ItemIcmKNNCFRecommender
from myRecommenders.WeightedHybrid import WeightedHybrid
from myRecommenders.WeightedHybridV2 import WeightedHybridScoreRecommender
from myRecommenders.WeightedListHybrid import WeightedListHybrid
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from GraphBased.P3alphaRecommender import P3alphaRecommender
from GraphBased.RP3betaRecommender import RP3betaRecommender


def sumScores(ss):
    res = ss[0]

    for s in ss[1:]:
        res = list(map(add, res, s))

    return res


def sumScoresWeights(ss, ww):
    res = ss[0] * ww[0]

    for s,w in zip(ss[1:], ww[1:]):
        res = list(map(add, res, s * w))

    return res
   

def createWeights(w0,w1,w2,w3,w4,w5,w6,w7):
    weights=[]
    weights.append(w0)
    weights.append(w1)
    weights.append(w2)
    weights.append(w3)
    weights.append(w4)
    weights.append(w5)
    weights.append(w6)
    return weights
#-----------------------

class WeightedHybridV3ScaledForBayesianSearch(BaseRecommender):
    RECOMMENDER_NAME = "WeightedHybridV3ScaledForBayesianSearch"

    def __init__(self, URM_train,ICM_all,recs):

        super(WeightedHybridV3ScaledForBayesianSearch, self).__init__(URM_train)
        self.recs = recs
        self.top = TopPop(URM_train)
        self.URM_train = URM_train
        self.means = []
        self.stds = []
       
        self.weights = []

       
            

    def fit(self, w0,w1,w2,w3,w4,w5,w6,w7):
        print("--------FITTING START-------")
        self.top.fit()
        
        self.weights =createWeights(w0,w1,w2,w3,w4,w5,w6)
        for rec in self.recs:
            s = rec._compute_item_score(np.array(range(self.URM_train.shape[0])), np.array(range(self.URM_train.shape[1])))
            self.means.append(np.mean(s))
            self.stds.append(np.std(s))
        print("------FITTING END------")

    # qui calcolo score per ogni metodo e sommo e tutte quelle belle cose
    # questa funzione è chiamat dentro reccommend e ritorna lo score degli items
    def _compute_item_score(self, user_id_array, items_to_compute=None):
        if isinstance(user_id_array, int):
            scores = []
            user_id = int(user_id_array)
            user_profile_length = self.URM_train[user_id].getnnz(1)
        
            if user_profile_length <= 0:
                #cold user top pop
                return self.top._compute_item_score([int(user_id)], items_to_compute)
                

            for rec,m,std in zip(self.recs,self.means,self.stds):
                sc = rec._compute_item_score(int(user_id), items_to_compute)
                sc = (sc - m) / std
                scores.append(sc)

            return np.array(sumScoresWeights(scores, self.weights))

            
        item_weights = np.zeros((len(user_id_array), self.URM_train.shape[1]), dtype=np.float32)
        

        for i, user_id in enumerate(user_id_array):
            # user_profile_length = self.URM_train[user_id].getnnz(1)

            scores = []
            user_profile_length = self.URM_train[user_id].getnnz(1)
    
            if user_profile_length <= 0:
                #cold user top pop
                item_weights[i]=self.top._compute_item_score([int(user_id)], items_to_compute)
                i += 1
                continue

            for rec,m,std in zip(self.recs,self.means,self.stds):
                sc = rec._compute_item_score(int(user_id), items_to_compute)
                sc = (sc - m) / std
                scores.append(sc)

            item_weights[i] = np.array(
                sumScoresWeights(scores, self.weights))

            # print(item_weights[i])
            i += 1
        return item_weights
