#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, '../Lab/')
import numpy as np
from Base.BaseRecommender import BaseRecommender

class WeightedHybrid(BaseRecommender):
    def __init__(self,URM_train,recommenders,weights, verbose=True):
        super(WeightedHybrid, self).__init__(URM_train, verbose = verbose)
        self.recommenders = recommenders
        self.weights =  [w /sum(weights) for w in weights]

    def recommend(self, user_id_array,weights=None, cutoff = None, remove_seen_flag=True, items_to_compute = None,
                  remove_top_pop_flag = False, remove_custom_items_flag = False, return_scores = False):

        rank = [[] for u in user_id_array]
        scores = [[] for u in user_id_array]
        if weights != None:
            self.weights = weights


        for r,w in zip(self.recommenders, self.weights):
            print(f"recommender {r}")
            ranking_list, scores_batch = r.recommend(user_id_array, cutoff = cutoff,return_scores=True)

            scores_sorted = []
            for i in range(len(user_id_array)):
                scores_sorted.append(sorted(scores_batch[i], reverse=True)[:cutoff])


            for i in range(len(user_id_array)):

                for item,score in zip(ranking_list[i],scores_sorted[i]):
                    if not item in rank[i]:
                        rank[i].append(item)
                        scores[i].append(score*w)
                    else:
                        scores[i][rank[i].index(item)] += score*w
                rank[i] = [item for _,item in sorted(zip(scores[i],rank[i]),reverse=True)]
                scores[i] = [item for item,_ in sorted(zip(scores[i],rank[i]),reverse=True)]

        tmp_scores = np.full((len(rank), 25975), 0.0)
        for i in range(len(user_id_array)):
            rank[i] = rank[i][:cutoff]
            for c in range(cutoff):
                tmp_scores[i,c] = scores[i][c]

        scores = tmp_scores

        rank = np.array(rank)

        if return_scores:
            return np.array(rank), np.array(scores)
        else:
            return np.array(rank)
