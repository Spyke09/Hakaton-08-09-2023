import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

import instance


class Clusterizator:
    def get_centroids(self, emb):
        c = list()
        for i in range(emb['label'].nunique()):
            s = emb[emb['label'] == i]
            l = len(s)
            c.append((s.sum() / l).to_numpy())
        return np.array(c)

    def metric(self, x, y):
        return 1 - x.dot(y) / (np.linalg.norm(y) * np.linalg.norm(x))

    def get_cluster(self, cluster_id, emb):
        return emb[emb['label'] == cluster_id]

    def get_nearest(self, cluster_id, centrs, emb):
        centr = centrs[cluster_id]
        cluster = self.get_cluster(cluster_id, emb)
        argmin = None
        min_ = 1
        for idx, c in zip(cluster.index, cluster.to_numpy()):
            cur_min = self.metric(centr, c)
            if cur_min < min_:
                min_ = cur_min
                argmin = idx
        return argmin

    def select_best_cluster(self, questions_clusters, q_embedding):
        max_ = 1
        arg_max = None
        for i, j in enumerate(questions_clusters):
            cur_max = self.metric(j, q_embedding)
            if cur_max < max_:
                max_ = cur_max
                arg_max = i
        return arg_max

    def __init__(self):
        self._instance = None
        self._res = None

    def fit(self, instance: instance.Instance):
        self._instance: instance.Instance = instance

    def train(self):

        q_model = SentenceTransformer('jgammack/distilbert-base-mean-pooling')

        e_answers = q_model.encode(self._instance.answers)
        pca_15 = PCA(n_components=15, random_state=42)
        q_emb_15d = pca_15.fit_transform(e_answers)
        answers_clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0.6, affinity='cosine',
                                                     linkage='average').fit(q_emb_15d)
        e_answers = pd.DataFrame(e_answers)
        e_answers['label'] = answers_clustering.labels_
        centroids = self.get_centroids(e_answers)
        res_s = []
        for i in range(len(self._instance.answers)):
            iii = self.get_nearest(e_answers["label"].iloc[i], centroids, e_answers)
            res_s.append(self._instance.answers[iii])

        self._res = res_s

    def get_clusters(self):
        return self._res
