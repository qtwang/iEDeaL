# coding = utf-8

import logging  
from timeit import default_timer as timer

import numpy as np
from sklearn.cluster import AgglomerativeClustering as AC

from util.conf import Conf


class Sampler:
    def __init__(self, conf: Conf):
        # self.__conf = conf
        self.__logger = logging.getLogger(self.__class__.__name__) # configured in EEGTrainable.__init__

        self.n_samples = conf.getHP('num_training_negative_samples')
        self.n_clusters = self.n_samples # without grouping


    def sample(self, candidates: np.ndarray):
        assert len(candidates.shape) == 2 and self.n_clusters <= candidates.shape[0]

        start = timer()

        # affinitystr or callable, default=’euclidean’. Can be “euclidean”, “l1”, “l2”, “manhattan”, “cosine”, or “precomputed”.
        # linkage{‘ward’, ‘complete’, ‘average’, ‘single’}, default=’ward’
        clustering = AC(n_clusters=self.n_clusters, affinity='euclidean', compute_full_tree=False, linkage='complete', distance_threshold=None, compute_distances=False).fit(candidates)
        
        self.__logger.info('cluster {:d}/{:d} in {:.3f}s'.format(self.n_samples, candidates.shape[0], timer() - start))

        reverse_index = {}
        for sample_id, cluster_id in enumerate(clustering.labels_):
            if cluster_id not in reverse_index:
                reverse_index[cluster_id] = [sample_id]
            else:
                reverse_index[cluster_id].append(sample_id)

        n_sample_each = 1
        return np.squeeze(np.asarray([np.random.choice(reverse_index[cluster_id], size=n_sample_each, replace=False, p=None) for cluster_id in range(self.n_clusters)]))
