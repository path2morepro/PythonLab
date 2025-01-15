from scipy.stats import rv_continuous
from sklearn.mixture import GaussianMixture
import numpy as np
from scipy.stats import norm
from scipy.stats import multivariate_normal



class GMM():
    def pdf(self, x, wgt, mu, sigma):
        # pdf = weights * Guassian(multiple RV)
        # shape of x: (1, d)
        pdf = 0
        # pdf_build_in = 0
        k = len(wgt)
        d = mu.shape[1]
        for i in range(k):
            # pdf_build_in += wgt[i] * multivariate_normal.pdf(x, mean=mu[i], cov=sigma[i])
            # I think I should do it manually
            det_sigma = np.linalg.det(sigma[i])  
            inv_sigma = np.linalg.inv(sigma[i])  
            norm_factor = 1 / (np.sqrt((2 * np.pi)**d * det_sigma))
            diff = x - mu[i]
            exponent = -0.5 * diff.T @ inv_sigma @ diff
            pdf += wgt[i] * norm_factor * np.exp(exponent)
        return pdf
    
    def cdf(self, x, wgt, mu, sigma):
        k = len(wgt)
        cdf = 0
        for i in range(k):
            mvn = multivariate_normal(mean=mu[i], cov=sigma[i])
            cdf += wgt[i] * mvn.cdf(x) 
        return cdf

    def rvs(self, wgt, mu, sigma, size=None, random_state=None):
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
        return [gmm.weights_ , gmm.means_, gmm.covariances_]
        
if __name__ == "__main__":
    wgt = np.array([0.6,0.4]) 
    mu = np.array([[0, 0], [3, 3]]) 
    sigma = np.array([[[1, 0], [0, 1]],
                    [[1, 0.5], [0.5, 1]]])
    gmm = GMM()
    sample = gmm.rvs(wgt = wgt, mu = mu, sigma = sigma, size = 1000)
    # print(sample)
    x = sample[0]
    pdf = gmm.pdf(x = x, wgt=wgt, mu=mu, sigma=sigma)
    parameters = gmm.fit(data = sample, K = 2)

