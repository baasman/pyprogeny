from sklearn.cluster import KMeans
import pandas as pd
import numpy as np


def prog_km(data, n_cluster):
    """ Default kmeans algorithm from sklearn.cluster.KMeans"""
    kmeans = KMeans(n_cluster).fit(data)
    return kmeans.labels_


def _progeny(data, cluster_alg, ext_cluster_alg=None,
             score_invert=False, n_cluster=range(2, 8), size=10, iteration=100):
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

        for i in range(iteration):
            progeny = np.zeros((prob_size, data.shape[1]))
            for index, lab in enumerate(range(1, n_cluster[k] + 1)):
                for col in range(data.shape[1]):
                    bottom = (lab - 1) * size
                    top = lab * size
                    bools = cluster_assignments[:, k] == index
                    if isinstance(data, pd.core.frame.DataFrame):
                        progeny[bottom:top, col] = np.random.choice(data[[col]] \
                                                                    .iloc[bools].values \
                                                                    .flatten(), size, replace=True)
                    elif isinstance(data, np.ndarray):
                        progeny[bottom:top, col] = np.random.choice(data[:, col], size, replace=True)

            pcluster = cluster_alg(progeny, clusters)
            for i in range(size * clusters):
                for j in range(size * clusters):
                    if pcluster[i] == pcluster[j]:
                        prob_matrix[i, j] += 1
        prob_matrix /= iteration

        true_prob = 0
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


def _optimal_gap(mean_gap, score_invert, n_cluster):
    if score_invert:
        n_gap = mean_gap.argmin()
    else:
        n_gap = mean_gap.argmax()

    return n_cluster[n_gap]


def _optimal_score(mean_score, score_invert, n_cluster):
    if score_invert:
        opt_index = mean_score.argmin()
    else:
        opt_index = mean_score.argmax()

    return n_cluster[opt_index]


class Progeny:
    """Progeny Clustering

    Refer to <article> to learn more about progeny clustering.

    Parameters
    ----------

    n_cluster: int, required
        The range of clusters you want to test. Can either be a range or list.
        For the gap method, it requires a continuous sequence of integers.

    method: str, required
        Can be either 'gap', 'score', or 'both'.

    cluster_algorithm: function, default: kmeans
        The clustering algorithm the progeny function will use.

    score_invert: bool, default: False
        Determines whether score is inverted or not.

    """

    def __init__(self, n_cluster, method='gap', cluster_algorithm=prog_km,
                 repeats=1, iteration=100, size=10, score_invert=False, nrandom=10):
        self.n_cluster = n_cluster
        self.cluster_algorithm = cluster_algorithm
        self.method = method
        self.score_invert = score_invert
        self.repeats = repeats
        self.iteration = iteration
        self.size = size
        self.nrandom = nrandom
        self.scores = None
        self.mean_score = None
        self.sd_score = None
        self.mean_gap = None
        self.sad_gap = None

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
        return scorem

    def score(self, data=None):
        assert self.scores is not None, "Run 'fit' first on the data " \
                                        "before computing %s score" % self.method

        if self.method == 'gap':
            gaps = np.zeros((p.repeats, self.scores.shape[1] - 2))
            for r in range(self.repeats):
                scores = self.scores.iloc[r].values
                gaps[r, :] = self._gap_method(scores, self.n_cluster)
            self.mean_gap = np.mean(gaps, axis=0)
            self.sd_gap = np.std(gaps, axis=0)
        elif self.method == 'score':
            assert data is not None, "To use the score method, you must the data"
            assert isinstance(data, np.ndarray)
            assert self.nrandom is not None and self.nrandom > 0
            r_score = np.zeros((p.nrandom, len(self.n_cluster)))
            for r in range(self.nrandom):
                r_data = np.zeros(data.shape)
                for col in range(r_data.shape[1]):
                    r_data[:, col] = np.random.uniform(np.min(data[:, col]),
                                                       np.max(data[:, col]),
                                                       r_data.shape[0])
                r_score[r, :] = _progeny(data, self.cluster_algorithm, self.cluster_algorithm,
                                         self.score_invert, self.n_cluster, self.size,
                                         self.iteration)[0]
            self.mean_score = np.mean(r_score, axis=0)
            self.sd_score = np.mean(r_score, axis=0)

    @staticmethod
    def _gap_method(scores, n_cluster):
        """Calculates the 'gap' between stability scores. Also makes sure that
        our sequence of clusters is continuous"""

        def consec_list(n_cluster):
            a = [i for i in sorted(set(n_cluster))]
            return len(a) == (a[-1] - a[0] + 1)

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

    def get_optimal(self):
        assert self.mean_gap is not None or \
               self.mean_score is not None, "Run 'score' first to compute a score"

        if self.method == 'gap':
            return _optimal_gap(self.mean_gap, self.score_invert,
                                self.n_cluster)
        elif self.method == 'score':
            return _optimal_score(self.mean_score, self.score_invert,
                                  self.n_cluster)
        elif self.method == 'both':
            best_gap = _optimal_gap(self.mean_gap, self.score_invert,
                                    self.n_cluster)
            best_score = _optimal_score(self.mean_score, self.score_invert, self.n_cluster)
            return best_gap, best_score


if __name__ == '__main__':
    # example code
    from sklearn.datasets import load_iris

    data = load_iris().data
    p = Progeny(n_cluster=range(2, 6), cluster_algorithm=prog_km, method='score',
                score_invert=False, repeats=1, nrandom=5)
    p.fit(data)
    p.score(data)
    p.get_optimal()

