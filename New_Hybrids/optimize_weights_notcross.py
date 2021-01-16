import math
import sys

sys.path.insert(0, '../Lab/')
import numpy as np
from WeightedHybridV10 import WeightedHybridScoreRecommender
from Base.Evaluation.Evaluator import EvaluatorHoldout
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample

from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from KNN.UserKNNCFRecommender import UserKNNCFRecommender
from UserIcmKNNCFRecommender import UserIcmKNNCFRecommender
from ItemIcmKNNCFRecommender import ItemIcmKNNCFRecommender

def optimize_weights(URM_train,recs, inits, fits,levels = 6,weights = None, index = 0, normalize_scores = False):
    if index == len(recs):
        return weights
    
    if index == 0:
        weights = [1 for i in range(len(recs))]
        return optimize_weights(URM_train,recs, inits, fits,levels, weights, 1)

    trials = [0.0001,0.001,0.01,0.1,1,10,100]
    print(f"optimize {index}")
    URM_train1,evaluator_validation,inits1 = rinnova(URM_train, inits[:index+1])
    recommender = WeightedHybridScoreRecommender(URM_train1, recs[:index+1], inits1, normalize_scores)
    recommender.fit(fits[:index+1],weights[:index+1])
    
    weights[:index+1] = optimize_process(recommender,evaluator_validation,weights[:index+1], trials, 0, levels)
    return optimize_weights(URM_train,recs, inits, fits,levels,weights, index+1)

def optimize_process(recommender,evaluator_validation, weights, trials, l=0, levels=6):

    MAPS = []

    

    print(f"trials {trials}")

    for i,t in enumerate(trials):

        weights[len(weights) -1] = t
        print(f"weights : {weights}")
        recommender.setWeights(weights)
        

        result_dict, _ = evaluator_validation.evaluateRecommender(recommender)
        print(f"MAP {result_dict[10]['MAP']} WITH WEIGHTS {weights}")
        MAPS.append(result_dict[10]["MAP"])
    
    best_t = trials[np.argmax(MAPS)]
    best_MAP = np.max(MAPS)
    print(f"BEST MAP {best_MAP} WITH T {best_t}")
    if (l + 1 == levels):
        print("-"*100 + "\n" + "-"*100+ "\n" + "-"*100)
        print(f"BEST T:{best_t} -> BEST MAP:{best_MAP}")
        weights[len(weights) -1] = best_t
        return weights

    diff_min = math.pow(10,-l-1)* best_t
    diff_magg = math.pow(10,-l)*best_t
    trials = [best_t - diff_magg*5, 
              best_t - diff_magg*2, 
              best_t - diff_magg*1.5, 
              best_t - diff_magg*1,
              best_t - diff_min*8, 
              best_t - diff_min*5, 
              best_t - diff_min*2,
              best_t - diff_min*1,
              best_t - diff_min*0.5,
              best_t - diff_min*0.2,
              best_t, 
              best_t + diff_magg*0.2, 
              best_t + diff_magg*0.5, 
              best_t + diff_magg*1, 
              best_t + diff_magg*2, 
              best_t + diff_magg*5, 
              best_t + diff_magg*8]
    print(trials)
    return optimize_process(recommender,evaluator_validation, weights, trials, l+1, levels)



def rinnova(URM_all, inits):
    URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_all, train_percentage = 0.8)
    evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10])

    for i in range(len(inits)):
        inits[i]["URM_train"] = URM_train
        first = True
        if "S_matrix_target" in inits[i]:
            if first:
                first = False
                itemknn = {}
                itemknn["topK"] = 141
                itemknn["shrink"] = 47
                iknn = ItemKNNCFRecommender(URM_train)
                iknn.fit(**itemknn)
                inits[i]["S_matrix_target"] = iknn.W_sparse
            else:
                iicm = {}
                iicm["topK"] = 893
                iicm["shrink"] = 2
                iicm["normalize"] = True
                iknn = ItemIcmKNNCFRecommender(URM_train, ICM_all)
                iknn.fit(**iicm)
                inits[i]["S_matrix_target"] = iknn.W_sparse
                
            

    return URM_train,evaluator_validation,inits
