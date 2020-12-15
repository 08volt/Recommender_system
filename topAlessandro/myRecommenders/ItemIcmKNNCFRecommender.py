import sys

sys.path.insert(0, '../Lab/')

from KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
class ItemIcmKNNCFRecommender(ItemKNNCFRecommender):

    def __init__(self,URM_train, ICM):
        super(ItemIcmKNNCFRecommender, self).__init__(URM_train=URM_train)
        self.new_reccomender = ItemKNNCFRecommender(ICM.transpose())
    def fit(self,shrink, topK, normalize=True):
        super().fit()
        self.new_reccomender.fit(shrink=shrink, topK=topK, normalize=normalize)
        self.W_sparse = self.new_reccomender.W_sparse
