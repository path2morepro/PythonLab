# transfer the string into number 
import numpy as np
import sys 
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal


def string_to_number(str, d):
    # I am confused about it
    # if I don't add a parameter d
    # I cannot fit the gmms
    values = str.split(",")
    try:
        number = [float(value.strip()) for value in values]
        number = np.array(number).reshape(-1, d)
    except ValueError as e:
        raise(f"cannot transfer {str} into number")
    return number

def calc_llh(data, wgt, mu, sigma):
    llh_sum = 0
    for i in range(data.shape[0]):
        llh = 0
        for k in range(wgt.shape[0]):
            mul_norm = multivariate_normal.pdf(x = data[i], mean=mu[k], cov=sigma[k])
            llh += wgt[k] * mul_norm
        llh_sum += np.log(llh)

    return llh_sum

def calc_AIC(data, wgt, mu, sigma):
    k = wgt.shape[0]
    d = mu.shape[1]
    param_amount = k * d + k * (d * (d + 1)) / 2 + (k - 1)
    llh = calc_llh(data, wgt, mu, sigma)
    aic = 2 * param_amount - 2 * llh
    return aic


if __name__ == '__main__':
    # python analyze.py ./data/data_file1.csv
    file_path = sys.argv[1]
    try:
        with open(file_path, 'r') as file:
            str = file.read()
    except FileNotFoundError:
        print(f"{file_path} not found")
    except Exception as e:
        print(f"file read error: {e}")
    d=5
    data = string_to_number(str=str, d=d)
    # print(data.shape)

    gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=123)
    model = gmm.fit(data)
    k_list = list(range(2,11))
    aic_list = []
    results = []
    for k in k_list:
        gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=123)
        # set random_state is fair for any k
        model = gmm.fit(data) 
        aic = calc_AIC(data = data, wgt = model.weights_, mu = model.means_, sigma = model.covariances_)
        aic_list.append(aic)

        results.append({
            'k': k,
            'aic': aic,
            'means': model.means_,
            'weights': model.weights_,
            'covariances': model.covariances_,
        })
        
    optimal_result = min(results, key=lambda x: x['aic'])
    print(f"The optimal k is: {optimal_result['k']}")
    print(f"Means: {optimal_result['means']}")
    print(f"Weights: {optimal_result['weights']}")
    print(f"Covariances: {optimal_result['covariances']}")