import numpy as np
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture

class GMM:
    def pdf(self, x, wgt, mu, sigma):
        # Convert x to at least 2D for consistent indexing (N, d)
        x = np.asarray(x)
        if x.ndim == 1:
            x = x[np.newaxis, :]  # Make it (1, d)

        N, d = x.shape
        K = len(wgt)

        pdf_vals = np.zeros(N)  # will hold the PDF for each row in x

        for n in range(N):
            # For each point, accumulate mixture component densities
            pdf_n = 0.0
            for i in range(K):
                # Manual (multivariate) calculation:
                diff = x[n] - mu[i]  # shape (d,)
                inv_sigma = np.linalg.inv(sigma[i])  # shape (d,d)
                det_sigma = np.linalg.det(sigma[i])

                # Normalization constant
                norm_factor = 1.0 / np.sqrt((2 * np.pi) ** d * det_sigma)

                # Exponent term
                exponent = -0.5 * diff @ inv_sigma @ diff  # shape ()

                pdf_n += wgt[i] * norm_factor * np.exp(exponent)

            pdf_vals[n] = pdf_n

        # If the input was a single point, return a single float
        if pdf_vals.size == 1:
            return pdf_vals.item()
        return pdf_vals

    def cdf(self, x, wgt, mu, sigma):
        x = np.asarray(x)
        if x.ndim == 1:
            x = x[np.newaxis, :]

        N, d = x.shape
        K = len(wgt)

        cdf_vals = np.zeros(N)

        for n in range(N):
            cdf_n = 0.0
            for i in range(K):
                mvn = multivariate_normal(mean=mu[i], cov=sigma[i])
                cdf_n += wgt[i] * mvn.cdf(x[n])
            cdf_vals[n] = cdf_n

        if cdf_vals.size == 1:
            return cdf_vals.item()
        return cdf_vals

    def rvs(self, wgt, mu, sigma, size=None, random_state=None):
        if random_state is None:
            random_state = np.random.default_rng()

        if size is None:
            size = 1

        K = len(wgt)
        d = mu.shape[1]

        # First choose which component each sample comes from
        choices = random_state.choice(K, size=size, p=wgt)

        # Allocate array for the samples
        samples = np.zeros((size, d))

        # For each component i, draw from the corresponding Gaussian
        for i in range(K):
            indices = np.where(choices == i)[0]  # all indices assigned to comp i
            if len(indices) > 0:
                samples[indices] = random_state.multivariate_normal(
                    mean=mu[i],
                    cov=sigma[i],
                    size=len(indices)
                )

        if size == 1:
            return samples[0]

        return samples


    def fit(self, data, K):
        gmm = GaussianMixture(n_components=K, covariance_type="full")
        gmm.fit(data)

        return [gmm.weights_, gmm.means_, gmm.covariances_]


if __name__ == "__main__":
    np.random.seed(0)  # For reproducibility in tests

    # True mixture parameters
    wgt = np.array([0.6, 0.4])
    mu = np.array([[0, 0],
                   [3, 3]])
    sigma = np.array([
        [[1, 0],
         [0, 1]],
        [[1, 0.5],
         [0.5, 1]]
    ])

    gmm = GMM()

    # Generate 1000 samples
    samples = gmm.rvs(wgt=wgt, mu=mu, sigma=sigma, size=1000)
    print("Generated samples shape:", samples.shape)

    # Evaluate pdf at a single point (take the first sample)
    x_single = samples[0]
    pdf_single = gmm.pdf(x_single, wgt, mu, sigma)
    print("PDF at single point:", pdf_single)

    # Evaluate pdf at multiple points
    pdf_values = gmm.pdf(samples[:5], wgt, mu, sigma)
    print("PDF at first 5 points:", pdf_values)

    # Fit a mixture model with 2 components
    fitted_params = gmm.fit(samples, K=2)
    print("Fitted parameters:")
    print(" - Weights:", fitted_params[0])
    print(" - Means:\n", fitted_params[1])
    print(" - Covariances:\n", fitted_params[2])

