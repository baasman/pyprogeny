import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def prog_km(X, n_cluster):
    """ Default kmeans algorithm from sklearn.cluster.KMeans"""
    kmeans = KMeans(n_cluster).fit(X)
    return kmeans.labels_


def _progeny(X, cluster_alg, ext_cluster_alg=None,
             score_invert=False, n_cluster=range(2, 8), size=10, iteration=100):
    """ Performs progeny algorithm, and calculates the score for each number of
    clusters
    """

    if ext_cluster_alg is not None:
        assert callable(ext_cluster_alg)
        cluster_alg = ext_cluster_alg

    len_n_cluster = len(n_cluster)
    cluster_assignments = np.zeros((X.shape[0], len_n_cluster))
    stability_scores = np.zeros(len_n_cluster)

    for k in range(len_n_cluster):

        clusters = n_cluster[k]

        labels = cluster_alg(X, clusters)
        cluster_assignments[:, k] = labels

        prob_size = size * clusters
        prob_matrix = np.zeros((prob_size, prob_size))

        for _ in range(iteration):
            progeny = np.zeros((prob_size, X.shape[1]))
            for index, lab in enumerate(range(1, n_cluster[k] + 1)):
                for col in range(X.shape[1]):
                    bottom = (lab - 1) * size
                    top = lab * size
                    bools = cluster_assignments[:, k] == index
                    if isinstance(X, pd.core.frame.DataFrame):
                        progeny[bottom:top, col] = np.random.choice(X[[col]]
                                                                    .iloc[bools].values
                                                                    .flatten(), size, replace=True)
                    elif isinstance(X, np.ndarray):
                        progeny[bottom:top, col] = np.random.choice(X[bools, col], size, replace=True)

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
            try:
                stability_scores[k] = score_dem / score_num
            except ZeroDivisionError:
                stability_scores[k] = score_dem / (score_num + .001)
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
        The range of clusters to iterate over. Can either be a range or list.
        For the gap method, it requires a continuous sequence of integers.

    method: str, default: 'gap'
        Can be either 'gap', 'score', or 'both'.
        If 'gap', the optimal cluster is chosen based on what score has the biggest/smallest gap
        from neighboring cluster scores.
        If 'score', the cluster number is based on the biggest/smallest average score over nrandom
        iterations of progeny scores based on random datasets. As expected, this is considerably slower
        than the gap method.
        if 'both', both 'gap' and 'score' methods will be used.

    cluster_algorithm: function, default: kmeans
        The clustering algorithm the progeny algorithm will use.

    ext_cluster_algorithm: function, default: kmeans
        The clustering algorithm the progeny algorithm will use.

    score_invert: bool, default: False
        Determines whether score is inverted or not. If False, each score is
        the result of the true classification over false classification.

    repeats: int, default: 1
        How often to repeat progeny function when computing scores. If greater than 1,
        will return a multidimensional scores table.

    nrandom: int, default: None
        Only used when using method 'score'. This integer specifies how many random
        datasets are generated to use in computing mean and std score per cluster.

    Examples
    --------
    >>> from pyprogeny import Progeny, prog_km
    >>> import numpy as np
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [4, 2], [4, 4], [4, 0]])
    >>> progeny = Progeny(n_cluster=range(2,5), method='score', cluster_algorithm=prog_km,
    ...                   repeats=1)
    >>> progeny.fit(X)
    >>> progeny.score(X)
    >>> progeny.get_optimal()
    """

    def __init__(self, n_cluster, method='gap', cluster_algorithm=prog_km,
                 ext_cluster_algorithm=None, repeats=1, iteration=100, size=10, score_invert=False,
                 nrandom=10):
        self.n_cluster = n_cluster
        self.cluster_algorithm = cluster_algorithm
        self.ext_cluster_algorithm = ext_cluster_algorithm
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
        self.sd_gap = None

    def __repr__(self):
        class_name = self.__class__.__name__
        return '%s: %s' % (class_name, repr({'n_cluster': self.n_cluster,
                                             'cluster_algorithm': self.cluster_algorithm,
                                             'method': self.method,
                                             'score_invert': self.score_invert}))

    # noinspection PyPep8Naming
    def fit(self, X):
        """Compute the stability scores for each cluster in the given range.
        Parameters
        ----------
        X : array-like matrix, or pandas dataframe.

        Returns
        -------
        array: stability score for each cluster in specified range
        """
        scorem = pd.DataFrame(columns=self.n_cluster)
        for rep in range(self.repeats):
            _p = _progeny(X, cluster_alg=self.cluster_algorithm, score_invert=self.score_invert,
                          n_cluster=self.n_cluster, iteration=self.iteration)
            scorem.loc[rep] = _p[0]
        self.scores = scorem
        return scorem

    def score(self, X=None):
        """Calculate scores based on specified score method.

        Parameters
        -------
        X: pandas DataFrame or numpy ndarray, default: None
            Only necessary if calculating stability scores based on 'score'
            method

        Returns
        -------
        tuple : mean and std score based on specified method
        """
        assert self.scores is not None, "Run 'fit' first on the data " \
                                        "before computing %s score" % self.method

        if self.method == 'gap' or self.method == 'both':
            gaps = np.zeros((p.repeats, self.scores.shape[1] - 2))
            for r in range(self.repeats):
                scores = self.scores.iloc[r].values
                gaps[r, :] = self._gap_method(scores, self.n_cluster)
            self.mean_gap = np.mean(gaps, axis=0)
            self.sd_gap = np.std(gaps, axis=0)
            if self.method != 'both':
                return self.mean_gap, self.sd_score
        if self.method == 'score' or self.method == 'both':
            assert X is not None, "To use the score method, you must the data"
            assert isinstance(X, np.ndarray)
            assert self.nrandom is not None
            assert self.nrandom > 0, "Must use a positive integer for nrandom"
            r_score = np.zeros((p.nrandom, len(self.n_cluster)))
            for r in range(self.nrandom):
                r_X = np.zeros(X.shape)
                for col in range(r_X.shape[1]):
                    r_X[:, col] = np.random.uniform(np.min(X[:, col]),
                                                    np.max(X[:, col]),
                                                    r_X.shape[0])
                r_score[r, :] = _progeny(r_X, self.cluster_algorithm, self.cluster_algorithm,
                                         self.score_invert, self.n_cluster, self.size,
                                         self.iteration)[0]
            self.mean_score = np.mean(r_score, axis=0)
            self.sd_score = np.mean(r_score, axis=0)
            if self.method != 'both':
                return self.mean_score, self.sd_score

        return self.mean_gap, self.sd_gap, self.mean_score, self.sd_score

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
        """Return optimal number of clusters as integer based on method specified

        Returns
        -------
        If method == 'both', this function will return a tuple specifying the
        optimal cluster based on gap, then score.

        """

        assert self.mean_gap is not None or \
               self.mean_score is not None, "Run 'score' first to compute a score"

        if self.method == 'gap':
            return _optimal_gap(self.mean_gap, self.score_invert,
                                self.n_cluster)
        if self.method == 'score':
            return _optimal_score(self.mean_score, self.score_invert,
                                  self.n_cluster)
        if self.method == 'both':
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
    print(p.mean_score)
    print("Optimal clusters in data: %d" % p.get_optimal())
