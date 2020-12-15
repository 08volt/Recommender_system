# import stuff
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
from MatrixFactorization.IALSRecommender import IALSRecommender
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
def obtainFits():    
    userknn = {}
    userknn["topK"] = 102
    userknn["shrink"] = 1

    itemknn = {}
    itemknn["topK"] = 141
    itemknn["shrink"] = 47

    cy = {}
    cy["epochs"] = 254
    cy["topK"] = 881
    cy["positive_threshold_BPR"] = 0.9765
    cy["learning_rate"] = 0.0002
    cy["batch_size"] = 10
    cy["sgd_mode"] = "sdg"

    uicm = {}
    uicm["topK"] = 181
    uicm["shrink"] = 0.1
    uicm["normalize"] = True

    iicm = {}
    iicm["topK"] = 893
    iicm["shrink"] = 2
    iicm["normalize"] = True

    alpha = {}
    alpha["alpha"] = 0.547615508822564
    alpha["topK"] = 500

    beta= {}
    beta["topK"]=500
    beta["alpha"]=0.3784740936494376
    beta["beta"]=0.1
    beta["implicit"]=False
    beta["normalize_similarity"]=False

    ials={}
    ials["epochs"]=45
    ials["num_factors"]=50
    ials["alpha"]=10.0
    ials["epsilon"]=10.0
    ials["reg"]=1e-07
    ials["init_mean"]=1.0
    ials["init_std"]=0.6773548197826565

    #ORIDNE RECOMMENDERS:
    #fits = [userknn,itemknn, cy, uicm, iicm, alpha]
    fits = [alpha,userknn,uicm,beta,ials,cy]
    return fits
    
def obtainRecs():
    recs = [
    P3alphaRecommender,
    UserKNNCFRecommender,
    UserIcmKNNCFRecommender,
    RP3betaRecommender,
    IALSRecommender,
    SLIM_BPR_Cython
    ]
    return recs
def obtainInits(URM_train,ICM_all):    
    only_URM = {
        "URM_train" : URM_train
    }

    cython = {
        "URM_train" : URM_train,
        "recompile_cython" : False,
        "verbose" : False
    }

    also_ICM = {
        "URM_train" : URM_train,
        "ICM" : ICM_all
    }


    inits = [
        only_URM,only_URM,also_ICM,only_URM,only_URM ,cython
    ]
    return inits
def createWeights(w0,w1,w2,w3,w4,w5):
    weights=[]
    weights.append(w0)
    weights.append(w1)
    weights.append(w2)
    weights.append(w3)
    weights.append(w4)
    weights.append(w5)
    
    return weights
#-----------------------
class WeightedHybridV22forBayesianSearch(BaseRecommender):
    RECOMMENDER_NAME = "WeightedHybridV22forBayesianSearch"

    # initialization
    # recs = [rec1, rec2, ...]
    # inits = [recsatt1, recsatt2, ..]
    # recsatt = {
    #     paramname = "paramvalue"
    # }
    def __init__(self, URM_train,ICM_all):

        super(WeightedHybridV22forBayesianSearch, self).__init__(URM_train)
        
        recs=obtainRecs()
        inits=obtainInits(URM_train,ICM_all)

        self.recs = []
        self.top = TopPop(URM_train)
        self.URM_train = URM_train
        
        self.weights = [1 for rec in recs]

        for rec, init in zip(recs, inits):
            self.recs.append(rec(**init))

    def fit(self,w0,w1,w2,w3,w4,w5):
        print("--------FITTING IN PROGRESS...-------")
        self.top.fit()
        #fits sono fixate per ricerca bayesiana
        fits=obtainFits()
        self.weights=createWeights(w0,w1,w2,w3,w4,w5)
        for rec, fit in zip(self.recs, fits):
            rec.fit(**fit)
        print("------FITTING END, SIAMO GROSSISSIMI ------")

    # qui calcolo score per ogni metodo e sommo e tutte quelle belle cose
    # questa funzione è chiamat dentro reccommend e ritorna lo score degli items
    def _compute_item_score(self, user_id_array, items_to_compute=None):
        item_weights = np.zeros((len(user_id_array), self.URM_train.shape[1]), dtype=np.float32)

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

            item_weights[i] = np.array(
                sumScoresWeights(scores, self.weights))

            # print(item_weights[i])
            i += 1
        return item_weights
