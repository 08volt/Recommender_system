# import stuff
import sys

import os

sys.path.insert(0, '../Lab/')
from Base.BaseRecommender import BaseRecommender
from operator import add
from Base.DataIO import DataIO
import numpy as np
from Base.NonPersonalizedRecommender import TopPop




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


class WeightedHybridScoreRecommender(BaseRecommender):
    RECOMMENDER_NAME = "WeightedHybridScoreRecommender"

    # initialization
    # recs = [rec1, rec2, ...]
    # inits = [recsatt1, recsatt2, ..]
    # recsatt = {
    #     paramname = "paramvalue"
    # }
    def __init__(self, URM_train, recs, inits):

        super(WeightedHybridScoreRecommender, self).__init__(URM_train)
        self.recs = []
        self.top = TopPop(URM_train)
        self.URM_train = URM_train
        self.means = []
        self.stds = []


        self.weights = [1 for rec in recs]

        for rec, init in zip(recs, inits):
            self.recs.append(rec(**init))
            
        
            

    def fit(self, fits, weights):
        print("--------FITTING START-------")
        self.top.fit()

        for rec, fit in zip(self.recs, fits):
            print(f"--------FITTING IN PROGRESS: {rec.RECOMMENDER_NAME}-------")
            rec.fit(**fit)
        self.weights = weights
        for rec in self.recs:
            s = rec._compute_item_score(range(URM_train.shape[0]), range(URM_train.shape[1]))
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
