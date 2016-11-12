from sklearn.cluster import KMeans
import pandas as pd
import numpy as np


def prog_km(data, n_cluster):
    """ Default kmeans algorithm from sklearn.cluster.KMeans"""
    kmeans = KMeans(n_cluster).fit(data)
    return kmeans.labels_


def _progeny(data, cluster_alg, ext_cluster_alg=None,
             score_invert=False,
             n_cluster=range(2, 8), size=10, iteration=100):
    """ Performs progeny algorithm, and calculates the score for each number of
    clusters
    """

    if ext_cluster_alg is not None:
        assert callable(ext_cluster_alg)
        cluster_alg = ext_cluster_alg

    len_n_cluster = len(n_cluster)
    cluster_assignments = np.zeros((data.shape[0], len_n_cluster), np.int32)
    stability_scores = np.zeros(len_n_cluster)

    for k in range(len_n_cluster):
        clusters = n_cluster[k]

        labels = cluster_alg(data, clusters)
        cluster_assignments[:, k] = labels

        prob_size = size * clusters
        prob_matrix = np.zeros((prob_size, prob_size))

        for iter in range(iteration):
            progeny = np.zeros((prob_size, data.shape[1]))
            for index, lab in enumerate(range(1, n_cluster[k] + 1)):
                for col in range(data.shape[1]):
                    bottom = (lab - 1) * size
                    top = lab * size
                    bools = cluster_assignments[:, k] == index
                    progeny[bottom:top, col] = np.random.choice(data[[col]] \
                                                                .iloc[bools].values \
                                                                .flatten(), size, replace=True)

            pcluster = cluster_alg(progeny, clusters)
            for i in range(size * clusters):
                for j in range(size * clusters):
                    if pcluster[i] == pcluster[j]:
                        prob_matrix[i, j] = prob_matrix[i, j] + 1
        prob_matrix /= iteration

        true_prob = 0
        false_prob = 0
        for lab in range(1, n_cluster[k] + 1):
            bottom = (lab - 1) * size
            top = lab * size
            true_prob += np.sum(prob_matrix[bottom:top, bottom:top])
        false_prob = np.sum(prob_matrix) - true_prob

        score_num = (false_prob / (size * (clusters - 1) * size * clusters))
        score_dem = (true_prob - size * clusters) / ((size - 1) * size * clusters)
        if score_invert:
            stability_scores[k] = score_num / score_dem
        else:
            stability_scores[k] = score_dem / score_num
    return stability_scores, cluster_assignments


def _gap_method(scores, n_cluster):
    """Calculates the 'gap' between stability scores. Also makes sure that
    our sequence of clusters is continuous"""
    def consec_list(n_cluster):
        a = [i for i in sorted(set(n_cluster))]
        return (len(a) == (a[-1] - a[0] + 1))

    if consec_list(n_cluster):
        pass
    else:
        raise ValueError("n_cluster is not a continuous sequence of integers. Use 'score' method instead")

    if len(n_cluster) < 2:
        raise ValueError("Can't use gap method when considering less than 2 clusters")

    if min(n_cluster) < 2:
        raise ValueError("Algorithm requires a minimum of 2 clusters")

    def agg_score(scores, i):
        return (scores[i + 1] * 2) - scores[i] - scores[i + 2]

    return [agg_score(scores, i) for i in range(scores.shape[0] - 2)]


def _score_method(scores, n_cluster):
    raise NotImplementedError


def _optimal_gap(mean_gap, score_invert, n_cluster):
    assert mean_gap.ndim == 1, 'mean_gap should be one dimensional'

    if score_invert:
        n_gap =  mean_gap.argmin()
    else:
        n_gap = mean_gap.argmax()

    return n_cluster[n_gap + 1]




class Progeny:
    """Progeny Clustering

    Refer to <article> to learn more about progeny clustering.

    Parameters
    ----------

    n_clusters: int, required
        The range of clusters you want to test. Can either be a range or list.
        For the gap method, it requires a continuous sequence of integers.

    method: str, required
        Can be either 'gap', 'score', or 'both'.

    cluster_algorithm: function, default: kmeans
        The clustering algorithm the progeny function will use.

    score_invert: bool, default: False
        Determines whether score is inverted or not.

    """

    def __init__(self, n_cluster, method=None, cluster_algorithm=prog_km,
                 repeats=1, iteration=100, score_invert=False, scores=None,
                 mean_gap=None, sd_gap=None):
        self.n_cluster = n_cluster
        self.cluster_algorithm = cluster_algorithm
        self.method = method
        self.score_invert = score_invert
        self.repeats = repeats
        self.iteration = iteration
        self.scores = scores
        self.mean_gap = mean_gap
        self.sd_gap = sd_gap

    def __repr__(self):
        class_name = self.__class__.__name__
        return '%s: %s' % (class_name, repr({'n_cluster': self.n_cluster,
                                             'cluster_algorithm': self.cluster_algorithm,
                                             'method': self.method,
                                             'score_invert': self.score_invert}))

    def fit(self, data):
        scorem = pd.DataFrame(columns=self.n_cluster)
        for rep in range(self.repeats):
            _p = _progeny(data, cluster_alg=self.cluster_algorithm, score_invert=self.score_invert,
                          n_cluster=self.n_cluster, iteration = self.iteration)
            scorem.loc[rep] = _p[0]
        self.scores = scorem
        return self

    def score(self):
        assert self.scores is not None, "Run 'fit' first on the data " \
                                        "before computing %s score" % method

        gaps = np.zeros((p.repeats, self.scores.shape[1] - 2))
        if self.method == 'gap':
            for r in range(self.repeats):
                scores = self.scores.iloc[r].values
                gaps[r, :] = _gap_method(scores, self.n_cluster)
            self.mean_gap = np.mean(gaps, axis=0)
            self.sd_gap = np.std(gaps, axis=0)

    def get_optimal(self):
        if self.method == 'gap':
            assert self.mean_gap is not None, "Run 'score' first to compute mean_gap score"
            return _optimal_gap(self.mean_gap, self.score_invert,
                                self.n_cluster)
        elif self.method == 'score':
            raise NotImplementedError
        elif self.method == 'both':
            raise NotImplementedError



if __name__ == '__main__':
    data = pd.read_csv("/Users/baasman/Documents/misc.Datasets/ptest.csv")
    data.drop('Unnamed: 0', axis=1, inplace=True)
    p = Progeny(n_cluster=range(2,8), cluster_algorithm=prog_km, method='gap',
                score_invert=False, repeats=1)
    p.fit(data).score()
    p.get_optimal()

