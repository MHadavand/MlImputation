import numpy as np
from scipy.stats import norm
try:
    from .lambda_distribution import GeneralizedLambdaDist as GLD
except (ImportError, ModuleNotFoundError):
    from lambda_distribution import GeneralizedLambdaDist as GLD


def update_bayesian_ind_cdf(gld, spatial_mean, spatial_variance, n_sample=150,
                            normalize=True):
    '''
    A method to implement bayesian updates based on combining conditional distributions represented by a
    generalized lambda distribution, Gaussian distribution and the prior normal (0,1) distribution based on
    full independence assumption.

    This method uses discretized CDF values instead of probability intervals
    '''

    if not isinstance(gld, GLD):
        raise ValueError(
            'The provided gld is not a generalized lambda distribution object')

    # Get the CDF values between 0 and 1 (This belongs to the GLD)
    F_gld = np.linspace(0.005, 0.995, n_sample)

    # Get the corresponding X values for the provided CDF values (for GLD)
    x_vals = list(map(gld.quantile_function, F_gld))

    # Extend the x_vals range
    n_extra = 20

    insert = np.linspace(x_vals[-1], 6, n_extra)
    x_vals = np.append(x_vals[:-1], insert)
    F_gld = np.append(F_gld, np.ones(n_extra - 1))

    insert = np.linspace(-6, x_vals[0], n_extra)
    x_vals = np.insert(x_vals[1:], 0, insert, axis=0)
    F_gld = np.insert(F_gld, 0, np.zeros(n_extra - 1))

    # Get the normal CDFs for the spatial mean and variance
    F_spatial = norm.cdf(x_vals, spatial_mean, np.sqrt(spatial_variance))

    # Get the normal CDFs for the prior distribution (standard normal after NScore transformation)
    F_global = norm.cdf(x_vals, 0.0, 1.0)

    # Combine based on independence between the spatial and multivariate distributions
    # Note that diff had values for all Xvalues except the first one
    F_updated = (F_gld * F_spatial) / F_global

    if normalize:
        F_updated = F_updated / F_updated[-1]

    return x_vals, F_gld, F_spatial, F_global, F_updated


def update_bayesian_pr_cdf(gld, spatial_mean, spatial_variance, n_sample=150,
                           normalize=True):
    '''
    A method to implement bayesian updates based on combining conditional distributions represented by a
    generalized lambda distribution, Gaussian distribution and the prior normal (0,1) distribution based on
    conditional independence assumption i.e permanence of ratios.

    This method uses discretized CDF values instead of probability intervals
    '''

    # Get the CDF values between 0 and 1 (This belongs to the GLD)
    F_gld = np.linspace(0.002, 0.998, n_sample)

    # Get the corresponding X values for the provided CDF values (for GLD)
    x_vals = list(map(gld.quantile_function, F_gld))

    # Extend the x_vals range
    n_extra = 20

    insert = np.linspace(x_vals[-1], 6, n_extra)
    x_vals = np.append(x_vals[:-1], insert)
    F_gld = np.append(F_gld, np.ones(n_extra - 1))

    insert = np.linspace(-6, x_vals[0], n_extra)
    x_vals = np.insert(x_vals[1:], 0, insert, axis=0)
    F_gld = np.insert(F_gld, 0, np.zeros(n_extra - 1))

    # Get the normal CDFs for the spatial mean and variance
    F_spatial = norm.cdf(x_vals, spatial_mean, np.sqrt(spatial_variance))

    # Get the normal CDFs for the prior distribution (standard normal after NScore transformation)
    F_global = norm.cdf(x_vals, 0.0, 1.0)

    # Combine based on independence between the spatial and multivariate distributions
    # Note that diff had values for all Xvalues except the first one
    part_a = (1 - F_global) / F_global
    part_b = (1 - F_spatial) / F_spatial
    part_c = (1 - F_gld) / F_gld

    F_updated = part_a / (part_a + part_b * part_c)

    if normalize:
        F_updated = F_updated / np.max(F_updated)

    return x_vals, F_gld, F_spatial, F_global, F_updated


def update_bayesian_ind(gld, spatial_mean, spatial_variance, n_sample=150,
                        normalize=True):
    '''
    A method to implement bayesian updates based on combining conditional distributions represented by a
    generalized lambda distribution, Gaussian distribution and the prior normal (0,1) distribution based on
    full independence assumption.

    This method uses probability values/intervals calculated based on discretized CDF
    '''

    # Get the CDF values between 0 and 1 (This belongs to the GLD)
    F_gld = np.linspace(0.005, 0.995, n_sample)

    # Get the corresponding X values for the provided CDF values (for GLD)
    x_vals = list(map(gld.quantile_function, F_gld))

    # Extend the x_vals range
    n_extra = 20

    insert = np.linspace(x_vals[-1], 6, n_extra)
    x_vals = np.append(x_vals[:-1], insert)
    F_gld = np.append(F_gld, np.ones(n_extra - 1))

    insert = np.linspace(-6, x_vals[0], n_extra)
    x_vals = np.insert(x_vals[1:], 0, insert, axis=0)
    F_gld = np.insert(F_gld, 0, np.zeros(n_extra - 1))

    # Get the normal CDFs for the spatial mean and variance
    F_spatial = norm.cdf(x_vals, spatial_mean, np.sqrt(spatial_variance))

    # Get the normal CDFs for the prior distribution (standard normal after NScore transformation)
    F_global = norm.cdf(x_vals, 0.0, 1.0)

    # Combine based on independence between the spatial and multivariate distributions
    # Note that diff had values for all Xvalues except the first one
    diff_F_updated = (np.diff(F_gld) * np.diff(F_spatial)) / np.diff(F_global)

    # For the first value, since the GLD CDF is zero, then the updated value would be zero too. So, we prepend zero to
    # the diff array
    diff_F_updated = np.insert(diff_F_updated, 0, 0, axis=0)

    F_updated = np.cumsum(diff_F_updated)

    if normalize:
        F_updated = F_updated / F_updated[-1]

    return x_vals, F_gld, F_spatial, F_global, F_updated


def update_bayesian_pr(gld, spatial_mean, spatial_variance, n_sample=150,
                       normalize=True):
    '''
    A method to implement bayesian updates based on combining conditional distributions represented by a
    generalized lambda distribution, Gaussian distribution and the prior normal (0,1) distribution based on
    conditional independence assumption i.e permanence of ratios.

    This method uses probability values/intervals calculated based on discretized CDF
    '''

    # Get the CDF values between 0 and 1 (This belongs to the GLD)
    F_gld = np.linspace(0.002, 0.998, n_sample)

    # Get the corresponding X values for the provided CDF values (for GLD)
    x_vals = list(map(gld.quantile_function, F_gld))

    # Extend the x_vals range
    n_extra = 20

    insert = np.linspace(x_vals[-1], 6, n_extra)
    x_vals = np.append(x_vals[:-1], insert)
    F_gld = np.append(F_gld, np.ones(n_extra - 1))

    insert = np.linspace(-6, x_vals[0], n_extra)
    x_vals = np.insert(x_vals[1:], 0, insert, axis=0)
    F_gld = np.insert(F_gld, 0, np.zeros(n_extra - 1))

    # Get the normal CDFs for the spatial mean and variance
    F_spatial = norm.cdf(x_vals, spatial_mean, np.sqrt(spatial_variance))

    # Get the normal CDFs for the prior distribution (standard normal after NScore transformation)
    F_global = norm.cdf(x_vals, 0.0, 1.0)

    # Combine based on indepdnence between the spatial and multivariate distributions
    # Note that diff had values for all Xvalues except the first one
    part_a = (1 - np.diff(F_global)) / np.diff(F_global)
    part_b = (1 - np.diff(F_spatial)) / np.diff(F_spatial)
    part_c = (1 - np.diff(F_gld)) / np.diff(F_gld)

    diff_F_updated = part_a / (part_a + part_b * part_c)

    # For the first value, since the GLD CDF is zero, then the updated value would be zero too. So, we prepend zero to
    # the diff array
    diff_F_updated = np.insert(diff_F_updated, 0, 0, axis=0)

    F_updated = np.cumsum(diff_F_updated)

    if normalize:
        F_updated = F_updated / F_updated[-1]

    return x_vals, F_gld, F_spatial, F_global, F_updated
