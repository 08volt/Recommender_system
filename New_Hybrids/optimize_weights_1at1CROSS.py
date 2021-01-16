import math
import sys

sys.path.insert(0, '../Lab/')
import numpy as np
from WeightedHybridV10 import WeightedHybridScoreRecommender
from Base.Evaluation.Evaluator import EvaluatorHoldout
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
import random
import gc
from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from KNN.UserKNNCFRecommender import UserKNNCFRecommender
from UserIcmKNNCFRecommender import UserIcmKNNCFRecommender
from ItemIcmKNNCFRecommender import ItemIcmKNNCFRecommender

def optimize_weights_1at1CROSS(URM_train,recs, inits, fits,levels = 2,weights = None, normalize_scores = False,validations = 4):
    
    if weights == None:
        weights = [1 for i in range(len(recs))]
    optimization_order = list(range(len(recs)))
    
    random.shuffle(optimization_order)
    for opt in optimization_order:
        trials = np.linspace(0.0001, 1, num=31, endpoint=True)
        #trials = [0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,50,80]
        print(f"optimize {opt}")
#         if rinnova_ds:
#             URM_train1,evaluator_validation,inits1 = rinnova(URM_train, inits, split)
#             recommender = WeightedHybridScoreRecommender(URM_train1, recs, inits1, normalize_scores)
#             recommender.fit(fits,weights)
   
        weights = optimize_process(URM_train,weights,opt, trials,recs, inits, fits,validations, 0, levels)
    return weights

def optimize_process(URM_train, weights,index, trials,recs, inits, fits,validations = 3, l=0, levels=2):

    MAPS = [[] for t in trials]
    
    for v in range(validations):
        print(f"TRIALS: {trials}")
        URM_train1 = None
        recommender = None
        gc.collect()
        URM_train1,evaluator_validation,inits1 = rinnova(URM_train, inits, 0.8)
        recommender = WeightedHybridScoreRecommender(URM_train1, recs, inits)
        recommender.fit(fits,weights)
        for i,t in enumerate(trials):

            weights[index] = t
            recommender.setWeights(weights)

            result_dict, _ = evaluator_validation.evaluateRecommender(recommender)
            print(f"MAP {result_dict[10]['MAP']} WITH WEIGHTS {weights}")
            MAPS[i].append(result_dict[10]["MAP"])
    
    means = [np.mean(m) for m in MAPS]
    best_t = trials[np.argmax(means)]
    best_MAP = np.max(means)
    print(f"MAPS: {means}")
    print(f"BEST MAP {best_MAP} WITH T {best_t}")
    if (l + 1 == levels):
        print("-"*100 + "\n" + "-"*100+ "\n" + "-"*100)
        print(f"BEST T:{best_t} -> BEST MAP:{best_MAP}")
        if best_t <= 0.0001:
            best_t = 0
        weights[index] = best_t
        return weights
    
    if best_t <= 0.0001:
        weights[index] = 0
        return weights
    elif best_t > 1:
        weights[index] = 1
        return weights
    else:
        left =  trials[0]/10 if np.argmax(means) == 0 else trials[np.argmax(means) - 1]
        right = 2 * trials[len(means)-1] - trials[len(means)-2]  if np.argmax(means) == len(means)-1 else trials[np.argmax(means) + 1]
        
        left_arr = np.linspace(left, best_t, num=15, endpoint=False)
        right_arr = np.linspace(best_t, right, num=15, endpoint=False)
        trials = np.concatenate((left_arr,right_arr))[1:]

    return optimize_process(URM_train, weights,index,trials,recs, inits, fits,validations, l+1, levels)



def rinnova(URM_all, inits,split):
    URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_all, train_percentage = split)
    evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10])
    first = True
    for i in range(len(inits)):
        inits[i]["URM_train"] = URM_train
        if "S_matrix_target" in inits[i]:
           
            if not first:
                itemknn = {}
                itemknn["topK"] = 121
                itemknn["shrink"] = 2
                iknn = ItemKNNCFRecommender(URM_train)
                iknn.fit(**itemknn)
                inits[i]["S_matrix_target"] = iknn.W_sparse
            else:
                first = False
                iicm = {}
                iicm["topK"] = 893
                iicm["shrink"] = 2
                iicm["normalize"] = True
                iknn = ItemIcmKNNCFRecommender(URM_train, inits[i]["ICM"])
                iknn.fit(**iicm)
                inits[i]["S_matrix_target"] = iknn.W_sparse
                
            

    return URM_train,evaluator_validation,inits
