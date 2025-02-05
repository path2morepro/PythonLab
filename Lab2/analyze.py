import numpy as np
import sys

from gmm import GMM
from scipy.stats import multivariate_normal

def string_to_number(text):
    lines = text.strip().splitlines()  # Split on newlines
    all_data = []

    for line_idx, line in enumerate(lines):
        if not line.strip():
            continue
        values = line.split(",")
        row = [float(v.strip()) for v in values]
        all_data.append(row)

    data = np.array(all_data, dtype=float)
    if data.shape[0] == 1:
        data = data.reshape(-1, 1)
    return data


def calc_llh(data, wgt, mu, sigma):
    llh_sum = 0.0
    N = data.shape[0]
    K = wgt.shape[0]

    for i in range(N):
        mixture_pdf = 0.0
        for k in range(K):
            comp_pdf = multivariate_normal.pdf(
                x=data[i],
                mean=mu[k],
                cov=sigma[k]
            )
            mixture_pdf += wgt[k] * comp_pdf
        llh_sum += np.log(mixture_pdf)
    return llh_sum


def calc_AIC(data, wgt, mu, sigma):
    K = wgt.shape[0]
    d = mu.shape[1]
    param_amount = K * d + K * (d * (d + 1) / 2.0) + (K - 1)
    llh = calc_llh(data, wgt, mu, sigma)
    aic = 2.0 * param_amount - 2.0 * llh
    return aic

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze.py <data_file.csv>")
        sys.exit(1)

    file_path = sys.argv[1]

    try:
        with open(file_path, "r") as f:
            text = f.read()
    except FileNotFoundError:
        print(f"{file_path} not found.")
        sys.exit(1)
    except Exception as e:
        print(f"File read error: {e}")
        sys.exit(1)

    data = string_to_number(text)

    k_list = range(2, 11)
    results = []

    best_aic = None
    best_k = None
    best_params = None

    for k in k_list:
        gmm_model = GMM()
        wgt, mu, sigma = gmm_model.fit(data, K=k)
        llh = calc_llh(data, wgt, mu, sigma)
        aic = calc_AIC(data, wgt, mu, sigma)
        results.append({
            "k": k,
            "llh": llh,
            "aic": aic,
            "weights": wgt,
            "means": mu,
            "covariances": sigma
        })

        if (best_aic is None) or (aic < best_aic):
            best_aic = aic
            best_k = k
            best_params = (wgt, mu, sigma)

    print("Model comparison results:")
    for res in results:
        print(f"  K = {res['k']}: LLH = {res['llh']:.2f}, AIC = {res['aic']:.2f}")

    print("\nBest AIC model:")
    print(f"  Number of components (K) = {best_k}")
    print(f"  AIC = {best_aic:.2f}")
    print("  Weights:", best_params[0])
    print("  Means:\n", best_params[1])
    print("  Covariances:\n", best_params[2])
