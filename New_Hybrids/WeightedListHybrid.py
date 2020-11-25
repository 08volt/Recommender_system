#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, '../Lab/')
import numpy as np
from Base.BaseRecommender import BaseRecommender

from Base.BaseRecommender import BaseRecommender
class WeightedListHybrid(BaseRecommender):
    def __init__(self,URM_train,recommenders,weights, verbose=True):
        super(WeightedListHybrid, self).__init__(URM_train, verbose = verbose)
        self.recommenders = recommenders
        self.weights = weights
        
    def recommend(self, user_id_array, cutoff = None, remove_seen_flag=True, items_to_compute = None,
                  remove_top_pop_flag = False, remove_custom_items_flag = False, return_scores = False):
        
        rank = []   
        
        n_r = len(self.recommenders)
        
        for user_id in user_id_array:
            recs = []
            for rrr in range(n_r):
                recs.append(self.recommenders[rrr].recommend(int(user_id)))

            i = [0 for rrrrr in range(n_r)]
            
            final_rec = []
            r = 0
            while(r<cutoff):
                rec_index = np.argmin(i)
                recommandation = recs[rec_index][i[rec_index]]
                if not recommandation in final_rec:
                    final_rec.append(recommandation)
                    r +=1
                i[rec_index] += 1
            
            rank.append(final_rec)
        
        scores = np.full((len(rank), 25975), 0.0)

          
        rank = np.array(rank)
        
        if return_scores:
            return np.array(rank), np.array(scores)
        else:
            return np.array(rank)

