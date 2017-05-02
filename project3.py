import random
import time

import numpy as np
from utils import plot_kmeans_clusters


def k_means(data, k, eps=1e-4, mu=None):
    """ Run the k-means algorithm
    data - an nxd ndarray
    k - number of clusters to fit
    eps - stopping criterion tolerance
    mu - an optional KxD ndarray containing initial centroids

    returns: a tuple containing
        mu - a KxD ndarray containing the learned means
        cluster_assignments - an N-vector of each point's cluster index
    """
    n, d = data.shape
    if mu is None:
        # randomly choose k points as initial centroids
        mu = data[random.sample(range(data.shape[0]), k)]
    else:
        # if initial means provided, copy to new array so method does not mutate inputs
        mu = np.array(mu)

    cost = None
    cost_next = None

    # Run k-means iterations until the cost plateaus
    while cost is None or cost_next is None or np.abs(cost - cost_next) > eps:
        cost = cost_next
        clusters = [[] for cluster in range(k)]

        # Given fixed representatives, assign points to clusters
        for i in range(n):
            representative_displacements = data[i] - mu
            representative_distances = (np.inner(representative_displacements,representative_displacements)).diagonal()
            nearest_representative = np.argmin(representative_distances)
            clusters[nearest_representative].append(i)

        # Given fixed clusters, assign cluster centroids to representatives, then compute cost
        cost_next = 0

        for cluster, cluster_member_indices in enumerate(clusters):
            if not len(cluster_member_indices):
                # This representative is not the closest to any points - salvage it by assigning it to a point
                cluster_member_indices = [min(n-1, cluster)]

            # Update representatives
            cluster_members = data[cluster_member_indices]
            mu[cluster] = np.average(cluster_members, axis=0)

            # Compute cost associated with this representative and cluster members
            data_displacements = cluster_members - mu[cluster]
            data_distances = (np.inner(data_displacements, data_displacements)).diagonal()
            cost_next += np.sum(data_distances)

    # Assign points to clusters based on final representatives
    cluster_assignments = np.zeros(n)
    for i in range(n):
        representative_displacements = data[i] - mu
        representative_distances = (np.inner(representative_displacements, representative_displacements)).diagonal()
        nearest_representative = np.argmin(representative_distances)
        cluster_assignments[i] = nearest_representative

    return (cost_next, mu, cluster_assignments)





class MixtureModel(object):
    def __init__(self, k):
        self.k = k
        self.params = {
            'pi': np.random.dirichlet([1]*k),
        }

    def __getattr__(self, attr):
        if attr not in self.params:
            raise AttributeError()
        return self.params[attr]

    def __setstate__(self, state):
        for k, v in state.items():
            setattr(self, k, v)

    def e_step(self, data):
        """ Performs the E-step of the EM algorithm
        data - an NxD pandas DataFrame

        returns a tuple containing
            (float) the expected log-likelihood
            (NxK ndarray) the posterior probability of the latent variables
        """
        raise NotImplementedError()

    def m_step(self, data, p_z):
        """ Performs the M-step of the EM algorithm
        data - an NxD pandas DataFrame
        p_z - an NxK numpy ndarray containing posterior probabilities

        returns a dictionary containing the new parameter values
        """
        raise NotImplementedError()

    def fit(self, data, eps=1e-4, verbose=True, max_iters=100):
        """ Fits the model to data
        data - an NxD pandas DataFrame
        eps - the tolerance for the stopping criterion
        verbose - whether to print ll every iter
        max_iters - maximum number of iterations before giving up

        returns a boolean indicating whether fitting succeeded

        if fit was successful, sets the following properties on the Model object:
          n_train - the number of data points provided
          max_ll - the maximized log-likelihood
        """
        last_ll = np.finfo(float).min
        start_t = last_t = time.time()
        i = 0
        while True:
            i += 1
            if i > max_iters:
                return False
            ll, p_z = self.e_step(data)
            new_params = self.m_step(data, p_z)
            self.params.update(new_params)
            if verbose:
                dt = time.time() - last_t
                last_t += dt
                print('iter %s: ll = %.5f  (%.2f s)' % (i, ll, dt))
                last_ts = time.time()
            if abs((ll - last_ll) / ll) < eps:
                break
            last_ll = ll

        setattr(self, 'n_train', len(data))
        setattr(self, 'max_ll', ll)
        self.params.update({'p_z': p_z})

        print('max ll = %.5f  (%.2f min, %d iters)' %
              (ll, (time.time() - start_t) / 60, i))

        return True


class GMM(MixtureModel):
    def __init__(self, k, d):
        super(GMM, self).__init__(k)
        self.params['mu'] = np.random.randn(k, d)

    def e_step(self, data):
        """ Performs the E-step of the EM algorithm
        data - an NxD pandas DataFrame

        returns a tuple containing
            (float) the expected log-likelihood
            (NxK ndarray) the posterior probability of the latent variables
        """

        n, d = data.shape

        # Compute N(x; mu, sig^2*I) for all data pt, class pairs
        PDFs = np.zeros((n, self.k))
        for j in range(self.k):
            sigsq = self.params['sigsq'][j]
            mu = self.params['mu'][j]

            coeff = 1 / np.sqrt(2 * np.pi * sigsq)

            # Compute |x^(i) - mu^(j)| for all i
            mean_displacements = data - mu # n by d
            mean_dists_sq = np.inner(mean_displacements, mean_displacements).diagonal() # 1 by n

            PDFs[:, j] = coeff * np.exp(-mean_dists_sq / (2 * sigsq))

        # Compute pi*N(x; mu, sig^2*I)
        weighted_PDFs = self.params['pi'] * PDFs # n by k

        # normalize class probabilites for each data point (so sum_j p(i, j) = 1 for all i)
        p = weighted_PDFs / np.vstack([weighted_PDFs.sum(axis=1)]*self.k).T
        return p

    def m_step(self, data, pz_x):
        raise NotImplementedError()

        return {
            'pi': new_pi,
            'mu': new_mu,
            'sigsq': new_sigsq,
        }

    def fit(self, data, *args, **kwargs):
        self.params['sigsq'] = np.asarray([np.mean(data.var(0))] * self.k)
        return super(GMM, self).fit(data, *args, **kwargs)


class CMM(MixtureModel):
    def __init__(self, k, ds):
        """d is a list containing the number of categories for each feature"""
        super(CMM, self).__init__(k)
        self.params['alpha'] = [np.random.dirichlet([1]*d, size=k) for d in ds]

    def e_step(self, data):
        raise NotImplementedError()

    def m_step(self, data, p_z):
        raise NotImplementedError()

        return {
            'pi': new_pi,
            'alpha': new_alpha,
        }

    @property
    def bic(self):
        raise NotImplementedError()
