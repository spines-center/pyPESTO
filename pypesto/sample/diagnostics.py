import numpy as np
import logging

from ..result import Result
from .geweke_test import burn_in_by_sequential_geweke
from .auto_correlation import autocorrelation_sokal

logger = logging.getLogger(__name__)


def geweke_test(result: Result, zscore: float = 2.) -> np.ndarray:
    """
    Calculates the burn-in of MCMC chains.

    Parameters
    ----------
    result:
        The pyPESTO result object with filled sample result.
    zscore:
        The Geweke test threshold. Default 2.

    Returns
    -------
    burn_in:
        Iteration where the first and the last fraction of the chain
        do not differ significantly regarding Geweke test -> Burn-In

    """
    # Get parameter samples as numpy arrays
    result.sample_result.burn_in = np.empty((result.sample_result.trace_x.shape[0]))

    for chain_id in range(result.sample_result.trace_x.shape[0]):
        chain = result.sample_result.trace_x[chain_id]

        # Calculate burn in index
        burn_in = burn_in_by_sequential_geweke(chain=chain,
                                               zscore=zscore)

        # Log
        logger.info(f'Geweke burn-in index: {burn_in}')

        # Fill in burn-in value into result
        result.sample_result.burn_in[chain_id] = burn_in

    return result.sample_result.burn_in


def auto_correlation(result: Result) -> np.ndarray:
    """
    Calculates the autocorrelation of the MCMC chains.

    Parameters
    ----------
    result:
        The pyPESTO result object with filled sample result.

    Returns
    -------
    auto_correlation:
        Estimate of the integrated autocorrelation time of
        the MCMC chains.
    """
    # Check if burn in index is available
    if result.sample_result.burn_in is None:
        geweke_test(result)

    # Get burn in index
    burn_in = result.sample_result.burn_in

    # Get chain length
    chain_length = result.sample_result.trace_x.shape[1]
    result.sample_result.auto_correlation = np.empty((result.sample_result.trace_x.shape[0]))

    for chain_id in range(result.sample_result.trace_x.shape[0]):
        if burn_in[chain_id] == chain_length:
            logger.warning("The autocorrelation can not "
                           "be estimated. The chain seems to "
                           "not have converged yet.\n"
                           "You may want to use a larger number "
                           "of samples.")
            result.sample_result.auto_correlation[chain_id] = chain_length
            continue

        # Get converged parameter samples as numpy arrays
        chain = np.asarray(result.sample_result.trace_x[chain_id, int(burn_in[chain_id]):, :])

        # Calculate autocorrelation
        auto_correlation_vector = autocorrelation_sokal(chain=chain)

        # Take the maximum over all components
        _auto_correlation = np.max(auto_correlation_vector)

        # Log
        logger.info(f'Estimated chain {chain_id} autocorrelation: {_auto_correlation}')

        # Fill in autocorrelation value into result
        result.sample_result.auto_correlation[chain_id] = _auto_correlation

    return result.sample_result.auto_correlation


def effective_sample_size(result: Result) -> float:
    """
    Calculate the effective sample size of the MCMC chains.

    Parameters
    ----------
    result:
        The pyPESTO result object with filled sample result.

    Returns
    -------
    ess:
        Estimate of the effective sample size of
        the MCMC chains.
    """

    # Check if autocorrelation is available
    if result.sample_result.auto_correlation is None:
        # Calculate autocorrelation
        auto_correlation(result)

    # Get burn in index
    burn_in = result.sample_result.burn_in

    # Get estimated chain autocorrelation
    _auto_correlation = result.sample_result.auto_correlation

    if _auto_correlation.shape == ():
        return -1.0

    result.sample_result.effective_sample_size = 0.0

    for chain_id in range(result.sample_result.trace_x.shape[0]):
        # Get converged parameter samples as numpy arrays
        chain = np.asarray(result.sample_result.trace_x[chain_id, int(burn_in[chain_id]):, :])

        # Get length of the converged chain
        N = chain.shape[0]

        # Calculate effective sample size
        ess = N / (1 + _auto_correlation[chain_id])

        # Log
        logger.info(f'Estimated chain {chain_id} effective sample size: {ess}')

        # Fill in effective sample size value into result
        result.sample_result.effective_sample_size += ess

    return result.sample_result.effective_sample_size
