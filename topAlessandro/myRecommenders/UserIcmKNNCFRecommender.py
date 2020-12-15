import sys

sys.path.insert(0, '../Lab/')

from KNN.UserKNNCFRecommender import UserKNNCFRecommender
class UserIcmKNNCFRecommender(UserKNNCFRecommender):

    def __init__(self,URM_train, ICM):
        super(UserIcmKNNCFRecommender, self).__init__(URM_train=URM_train)
        self.new_reccomender = UserKNNCFRecommender(URM_train * ICM)
    def fit(self,shrink, topK, normalize=True):
        super().fit()
        self.new_reccomender.fit(shrink=shrink, topK=topK, normalize=normalize)
        self.W_sparse = self.new_reccomender.W_sparse
