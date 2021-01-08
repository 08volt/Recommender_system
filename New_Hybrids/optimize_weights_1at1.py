import math
import sys

sys.path.insert(0, '../Lab/')
import numpy as np
from WeightedHybridV10 import WeightedHybridScoreRecommender
from Base.Evaluation.Evaluator import EvaluatorHoldout
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
import random
from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from KNN.UserKNNCFRecommender import UserKNNCFRecommender
from UserIcmKNNCFRecommender import UserIcmKNNCFRecommender
from ItemIcmKNNCFRecommender import ItemIcmKNNCFRecommender

def optimize_weights_1at1(URM_train,recs, inits, fits,levels = 6,weights = None, normalize_scores = False, rinnova_ds=False, split = 0.8):
    
    if weights == None:
        weights = [1 for i in range(len(recs))]
    optimization_order = list(range(len(recs)))
    recommender = None
    if not rinnova_ds:
        URM_train1,evaluator_validation,inits1 = rinnova(URM_train, inits, split)
        recommender = WeightedHybridScoreRecommender(URM_train1, recs, inits1, normalize_scores)
        recommender.fit(fits,weights)
    random.shuffle(optimization_order)
    for opt in optimization_order:
        trials = np.linspace(0.0001, 1, num=11, endpoint=True)
        #trials = [0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,50,80]
        print(f"optimize {opt}")
        if rinnova_ds:
            URM_train1,evaluator_validation,inits1 = rinnova(URM_train, inits, split)
            recommender = WeightedHybridScoreRecommender(URM_train1, recs, inits1, normalize_scores)
            recommender.fit(fits,weights)
   
        weights = optimize_process(recommender,evaluator_validation,weights.copy(),opt, trials, 0, levels)
    return weights

def optimize_process(recommender,evaluator_validation, weights,index, trials, l=0, levels=6):

    MAPS = []

    

    print(f"trials {trials}")

    for i,t in enumerate(trials):

        weights[index] = t
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
        if best_t <= 0.0001:
            best_t = 0
        weights[index] = best_t
        return weights
    
    if best_t <= 0.0001:
        trials = np.linspace(0.0001,trials[1], num=5, endpoint=False)
    elif best_t >= 1:
        trials = np.linspace(trials[-2],1, num=5, endpoint=False)
    else:
        left =  trials[0]/10 if np.argmax(MAPS) == 0 else trials[np.argmax(MAPS) - 1]
        right = trials[len(MAPS)-1]*1.5 if np.argmax(MAPS) == len(MAPS)-1 else trials[np.argmax(MAPS) + 1]
        left_arr = np.linspace(left, best_t, num=5, endpoint=False)
        right_arr = np.linspace(best_t, right, num=5, endpoint=False)
        trials = np.concatenate((left_arr,right_arr))[1:]

    return optimize_process(recommender,evaluator_validation, weights,index, trials, l+1, levels)



def rinnova(URM_all, inits,split):
    URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_all, train_percentage = split)
    evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10])
    first = True
    for i in range(len(inits)):
        inits[i]["URM_train"] = URM_train
        if "S_matrix_target" in inits[i]:
           
            if not first:
                itemknn = {}
                itemknn["topK"] = 141
                itemknn["shrink"] = 47
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
