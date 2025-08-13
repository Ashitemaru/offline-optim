import numpy as np
from scipy import stats


def constant_hazard(lam, r):
    """
    Hazard function for bayesian online learning
    Arguments:
        lam - inital prob
        r - R matrix
    """
    return 1 / lam * np.ones(r.shape)


def gaussian_hazard(mu, sigma, r):
    # Hazard function = pdf/survival = pdf/(1-cdf)
    # Greater run length will have greater hazard. So we use stats.norm.pdf(r) directly which will return an array of hazard values for each run length
    pdf = stats.norm.pdf(
        r, mu, sigma
    )  # Computes probability for each runlength starting from 0 to t
    survival = 1 - stats.norm.cdf(
        r, mu, sigma
    )  # Computes survival for each runlength starting from 0 to t
    hazard_arr = (
        pdf / survival
    )  # Computes hazard value for each runlength starting from 0 to t
    # hazard_arr = hazard*np.ones(r.shape) # Apply the same hazard for all vaules of column t in R[] ---> Not reqired since the previous line itself is an array of r.shape values
    return hazard_arr
