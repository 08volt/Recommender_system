import math
import sys

sys.path.insert(0, '../Lab/')
import numpy as np
from WeightedHybridV2 import WeightedHybridScoreRecommender
from Base.Evaluation.Evaluator import EvaluatorHoldout
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample

def optimize_weights(URM_train,validations,recs, inits, fits,levels = 3,weights = None, index = 0):
    if index == len(recs):
        return weights
    if index == 0:
        weights = [1 for i in range(len(recs))]
        return optimize_weights(URM_train,validations,recs, inits, fits,levels, weights, 1)

    trials = [0.0001,0.001,0.01,0.1,1,10,100]
    print(f"optimize {index}")
    weights[:index+1] = optimize_process(URM_train,validations,recs[:index+1],inits[:index+1], fits[:index+1],weights[:index+1], trials, 0, levels)
    return optimize_weights(URM_train,validations,recs, inits, fits,levels,weights, index+1)

def optimize_process(URM_all,validations, recs, inits, fits, weights, trials, l=0, levels=3):

    MAPS = [[] for t in trials]

    for v in range(validations):
        URM_train,evaluator_validation,inits = rinnova(URM_all, inits)

        recommender = WeightedHybridScoreRecommender(URM_train, recs, inits)

        print(f"VALIDATION {v+1} with trials {trials}")

        for i,t in enumerate(trials):

            weights[len(weights) -1] = t
            print(f"weights : {weights}")
            recommender.fit(fits,weights)

            result_dict, _ = evaluator_validation.evaluateRecommender(recommender)
            print(f"MAP {result_dict[10]['MAP']} WITH WEIGHTS {weights}")
            MAPS[i].append(result_dict[10]["MAP"])
    means = [np.mean(m) for m in MAPS]
    best_t = trials[np.argmax(means)]
    best_MAP = np.max(means)
    print(f"BEST MAP {best_MAP} WITH T {best_t}")
    if (l + 1 == levels):
        print("-"*100 + "\n" + "-"*100+ "\n" + "-"*100)
        print(f"BEST T:{best_t} -> BEST MAP:{best_MAP}")
        weights[len(weights) -1] = best_t
        return weights

    diff_min = math.pow(10,-l-1)* best_t
    diff_magg = math.pow(10,-l)*best_t
    trials = [best_t - diff_min*8, 
              best_t - diff_min*5, 
              best_t - diff_min*2,
              best_t, 
              best_t + diff_magg*2, 
              best_t + diff_magg*5, 
              best_t + diff_magg*8]
    print(trials)
    return optimize_process(URM_all,validations, recs, inits, fits, weights, trials, l+1, levels)



def rinnova(URM_all, inits):
    URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_all, train_percentage = 0.85)
    evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10])

    for i in range(len(inits)):
        inits[i]["URM_train"] = URM_train

    return URM_train,evaluator_validation,inits
