
#import stuff
from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from KNN.UserKNNCFRecommender import UserKNNCFRecommender
from Base.NonPersonalizedRecommender import TopPop
from Base.BaseRecommender import BaseRecommender
from operator import add
import numpy as np
from Base.DataIO import DataIO

#for manually change parameters -- NOT IMPLEMENTED YET
HybridRecommender_params = {
    'a': 1.0,  # userCF
    'b': 1.0,  # SLIM
    'c': 1.0,  # itemCF
}   

def sumScores(s1,s2,s3):
    s1= list(map(add, s1, s2) )
    s1= list(map(add, s1, s3) )
    return s1

def sumScoresWeights(s1,s2,s3,ws1,ws2,ws3):
    s1= list(map(add, ws1*s1, ws2*s2) )
    s1= list(map(add, s1, ws3*s3) )
    return s1

import sys, os

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

class UserItemBPRHybridScoreRecommender(BaseRecommender):
    
    RECOMMENDER_NAME = "UserItemBPRHybridScoreRecommender"
    #initialization 
    def __init__(self, URM_train):

        super(UserItemBPRHybridScoreRecommender, self).__init__(URM_train)
       
        #ci asta ggiungere IALS faceva benino
        
        self.recommenderBPR = SLIM_BPR_Cython(self.URM_train, recompile_cython=False)
        
        #se uso questo devo anche farmi passare icm
        #self.itemCBF = ItemKNNCBFRecommender(data_wrapper.ICM_sub_class, self.URM_train)
        self.itemKNNCF = ItemKNNCFRecommender(self.URM_train)
        self.userRecommender = UserKNNCFRecommender(self.URM_train)
        self.topPop = TopPop(self.URM_train)
        

    #qui a,b,c sono i pesi del mix degli scores
    #fit function
    def fit(self, URM_train=None,ws1=1,ws2=1,ws3=1):
        print("--------FITTING IN PROGRESS...-------")
        #blockPrint()

        if URM_train is not None:
            self.URM_train = URM_train

        self.ws1 = ws1
        self.ws2 = ws2
        self.ws3 = ws3

        self.recommenderBPR.fit(epochs=300, batch_size=1, sgd_mode='sgd', learning_rate=0.001, positive_threshold_BPR=0.8)
        self.itemKNNCF.fit(shrink=110, topK=110)
        self.userRecommender.fit(shrink=0.5, topK=150)
        self.topPop.fit()
        #enablePrint()
        print("------FITTING END, SIAMO GROSSISSIMI ------")
        
    #qui calcolo score per ogni metodo e sommo e tutte quelle belle cose
    #questa funzione è chiamat dentro reccommend e ritorna lo score degli items
    def _compute_item_score(self, user_id_array, items_to_compute=None):
        item_weights = np.zeros((len(user_id_array), self.URM_train.shape[1]), dtype=np.float32)

        i = 0

        for user_id in user_id_array:
            user_profile_length = self.URM_train[user_id].getnnz(1)

            # recommend cold users
            if user_profile_length == 0:#topPop.compute>Itemscore
                
                item_weights[i]=self.topPop._compute_item_score_single_user()
                
            elif user_profile_length == 1:#silmBPR compute score
                
                item_weights[i]=self.recommenderBPR._compute_item_score(int(user_id), items_to_compute)
            else:#warm users
                scoresUser=self.userRecommender._compute_item_score(int(user_id), items_to_compute)
                scoresItem=self.itemKNNCF._compute_item_score(int(user_id), items_to_compute)
                scoresBPR=self.recommenderBPR._compute_item_score(int(user_id), items_to_compute)
                

                item_weights[i]=np.array(sumScoresWeights(scoresUser,scoresItem,scoresBPR,self.ws1,self.ws2,self.ws3))    
            
            #print(item_weights[i])     
            i += 1
        return item_weights

    #serve a salvare il modello alla fine della ricderva di parametri bayesiana
    def save_model(self, folder_path, file_name=None):
        
        if file_name is None:
            file_name = self.RECOMMENDER_NAME
        self._print("Saving model in file '{}'".format(folder_path + file_name))
        data_dict_to_save = HybridRecommender_params
        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save = data_dict_to_save)
        self._print("Saving complete")
        
        #self.itemScoresHybrid.save_model(SAVED_MODELS_PATH, None)
        #self.userScoresHybrid.save_model(SAVED_MODELS_PATH, None)

    def load_model(self, folder_path, file_name=None):
        
        if file_name is None:
            file_name = self.RECOMMENDER_NAME
        self._print("Loading model from file '{}'".format(folder_path + file_name))
        dataIO = DataIO(folder_path=folder_path)
        data_dict = dataIO.load_data(file_name=file_name)
        HybridRecommender_params = data_dict
        self._print("Loading complete")
        
        #self.itemScoresHybrid.load_model(SAVED_MODELS_PATH)
        #self.userScoresHybrid.load_model(SAVED_MODELS_PATH)
        