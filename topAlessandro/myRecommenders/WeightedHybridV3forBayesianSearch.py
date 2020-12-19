# import stuff
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
#-----QUI PASSO I RECOMMENDERS GIA' FITTATI NELL'INIZIALIZZAZIONE---------
#----- semplicemnte in init richiede una variabile recs, che coontiene
#----- la lista recs=[UserKNNCF, P3Alpha,...] tutti fittati
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


# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')


# Restore
def enablePrint():
    sys.stdout = sys.__stdout__
    
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
class WeightedHybridV3forBayesianSearch(BaseRecommender):
    RECOMMENDER_NAME = "WeightedHybridV3forBayesianSearch"

    # initialization
    # recs = [rec1, rec2, ...]
    # inits = [recsatt1, recsatt2, ..]
    # recsatt = {
    #     paramname = "paramvalue"
    # }
    def __init__(self, URM_train,ICM_all,recs):

        super(WeightedHybridV3forBayesianSearch, self).__init__(URM_train)
        
        self.recs = recs
        self.top = TopPop(URM_train)
        self.URM_train = URM_train
        
        self.weights = []

       

    def fit(self,w0,w1,w2,w3,w4,w5,w6,w7):
        print("--------FITTING...-------")
        self.top.fit()
        self.weights=createWeights(w0,w1,w2,w3,w4,w5,w6,w7)

    # qui calcolo score per ogni metodo e sommo e tutte quelle belle cose
    # questa funzione è chiamat dentro reccommend e ritorna lo score degli items
    def _compute_item_score(self, user_id_array, items_to_compute=None):
        item_weights = np.zeros((len(user_id_array), self.URM_train.shape[1]), dtype=np.float32)
        print("--------Computing scores (recommending)...-------")
        for i, user_id in enumerate(user_id_array):
            # user_profile_length = self.URM_train[user_id].getnnz(1)

            scores = []
            user_profile_length = self.URM_train[user_id].getnnz(1)
    
            if user_profile_length==0:
                #cold user top pop
                item_weights[i]=self.top._compute_item_score([int(user_id)], items_to_compute)
                i += 1
                continue

            for rec in self.recs:
                scores.append(rec._compute_item_score(int(user_id), items_to_compute))

            item_weights[i] = np.array(sumScoresWeights(scores, self.weights))

            # print(item_weights[i])
            i += 1
        return item_weights
