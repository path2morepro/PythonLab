from scipy.stats import rv_continuous
from sklearn.mixture import GaussianMixture
import numpy as np
from scipy.stats import norm
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture

class GMM(rv_continuous):
    def _argcheck(self, wgt, mu, sigma): 
        ks, ds, _ = sigma.shape
        # shape of sigma: (k, d, d)
        print(mu.shape)
        km, dm = mu.shape
        # shape of mu: (k, d)
        k = len(wgt)
        # shape of wgt: (k, 1)
        if not np.isclose(np.sum(wgt), 1):
            raise ValueError(f"The sum of wgt is {np.sum(wgt)}, should be 1.")
        elif np.any(wgt <= 0) :
            raise ValueError("All elements in wgt should greater than 0.")
        
        if not (k == km and k == ks):
            raise ValueError("The k of mu and wgt do not match.")
        elif ds != dm:
            raise ValueError("The d of mu and wgt do not match.")
        
        return True
    
    def _pdf(self, x, wgt, mu, sigma):
        # pdf = weights * Guassian(multiple RV)
        # shape of x: (1, d)
        pdf = 0
        k = len(wgt)
        for i in range(k):
            pdf += wgt[i] * multivariate_normal.pdf(x, mean=mu[i], cov=sigma[i])
        return pdf
    
    def _cdf(self, x, wgt, mu, sigma):
        pass

    def _rvs(self, wgt, mu, sigma, size=None, random_state=None):
        # random sample based on pi
        # generate K sample of multiple gaussian distribution
        if random_state is None:
            random_state = np.random.default_rng()
        if size is None:
            size = 1

        k = len(wgt)  
        d = mu.shape[1]  
        component_choices = random_state.choice(k, size=size, p=wgt)
        samples = np.zeros((size, d))
        for i in range(k):
            indices = np.where(component_choices == i)[0]
            samples[indices] = random_state.multivariate_normal(mu[i], sigma[i], size=len(indices))
        
        return samples
    
    def fit(self, data, K):
        gmm = GaussianMixture(n_components=K, covariance_type='full', random_state=123)
        gmm.fit(data)
        self.weights_ = gmm.weights_
        self.means_ = gmm.means_
        self.covariances_ = gmm.covariances_
        return self
        
if __name__ == "__main__":
    wgt = np.array([0.6, 0.4]) 
    mu = np.array([[0, 0], [3, 3]]) 
    sigma = [
        np.array([[1, 0], [0, 1]]),  
        np.array([[1, 0.5], [0.5, 1]]) 
    ]
    gmm = GMM()
    gmm.rvs(wgt = wgt, mu = mu, sigma = sigma)
