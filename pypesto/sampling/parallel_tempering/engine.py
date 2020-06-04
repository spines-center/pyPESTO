import abc
from typing import List
import multiprocessing as mp
import concurrent.futures
import os
import cloudpickle as pickle

from ..sampler import InternalSampler


class PTEngine(abc.ABC):
    """Abstract base class for parallel tempering execution/parallelization
    engines."""

    @abc.abstractmethod
    def sample(
            self,
            n_samples: int,
            samplers: List[InternalSampler],
            betas: List[float]
    ) -> List[InternalSampler]:
        """Perform the actual sampling.

        Parameters
        ----------
        n_samples:
            The number of samples to generate.
        samplers:
            The samplers maintaining the single chains.
        betas:
            The inverse temperatures for the samplers.

        Returns
        -------
        samplers:
            A list of samplers in the same order as the input samplers.
        """


class SingleCorePTEngine(PTEngine):
    """Perform the sampling sequentially on a single process."""

    def sample(
            self,
            n_samples: int,
            samplers: List[InternalSampler],
            betas: List[float]
    ) -> List[InternalSampler]:
        """
        Parameters
        ----------
        TODO

        Returns
        -------
        TODO
        """
        for sampler, beta in zip(samplers, betas):
            sampler.sample(n_samples=n_samples, beta=beta)
        return samplers


def work_pickled(args):
    return pickle.dumps(work(pickle.loads(args)))


def work(args):
    sampler, beta, n_samples = args
    sampler.sample(n_samples=1, beta=beta)
    return sampler


class MultiProcessPTEngine(PTEngine):
    """Parallelize the sampling using multiprocessing.

    Attributes
    ----------
    n_procs:
        The maximum number of processes to use in parallel.
    """

    def __init__(self, n_procs: int = None):
        if n_procs is None:
            n_procs = os.cpu_count()
        self.n_procs = n_procs

    def sample(
            self,
            n_samples: int,
            samplers: List[InternalSampler],
            betas: List[float]
    ) -> List[InternalSampler]:
        args = [pickle.dumps((sampler, beta, n_samples))
                for sampler, beta in zip(samplers, betas)]

        n_procs = min(self.n_procs, len(args))
        with mp.Pool(processes=n_procs) as pool:
            rets = pool.map(work_pickled, args)
        samplers = [pickle.loads(ret) for ret in rets]
        return samplers


class MultiThreadPTEngine(PTEngine):
    """
    Parallelize the sampling using multithreading.

    Parameters
    ----------
    n_threads:
        The maximum number of threads to use in parallel.
    """

    def __init__(self, n_threads: int = None):
        if n_threads is None:
            n_threads = os.cpu_count()
        self.n_threads: int = n_threads

    def sample(
            self,
            n_samples: int,
            samplers: List[InternalSampler],
            betas: List[float]
    ) -> List[InternalSampler]:
        """Deepcopy tasks and distribute work over parallel threads."""
        args = [(sampler, beta, n_samples)
                for sampler, beta in zip(samplers, betas)]

        n_threads = min(self.n_threads, len(args))
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=n_threads) as pool:
            samplers = pool.map(work, args)
        return list(samplers)
