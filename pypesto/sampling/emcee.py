from typing import Dict, List, Union
import numpy as np

from ..problem import Problem
from .sampler import Sampler
from .result import McmcPtResult
from ..startpoint import uniform

try:
    import emcee
except ImportError:
    emcee = None


class EmceeSampler(Sampler):

    def __init__(
            self,
            nwalkers: int = 1,
            sampler_args: Dict = None,
            run_args: Dict = None):
        super().__init__()
        self.nwalkers = nwalkers

        if sampler_args is None:
            sampler_args = {}
        self.sampler_args = sampler_args
        if run_args is None:
            run_args = {}
        self.run_args = run_args

        # set in initialize
        self.problem = None
        self.sampler = None
        self.state = None

    def initialize(self,
                   problem: Problem,
                   x0: Union[np.ndarray, List[np.ndarray]]):
        self.problem = problem

        objective = self.problem.objective
        ndim = len(self.problem.x_free_indices)

        def log_prob(x):
            return - 1. * objective(x)

        self.sampler = emcee.EnsembleSampler(
            nwalkers=self.nwalkers, ndim=ndim, log_prob_fn=log_prob,
            **self.sampler_args)

        self.state = uniform(
            n_starts=self.nwalkers, lb=problem.lb, ub=problem.ub)

    def sample(
            self, n_samples: int, beta: float = 1.
    ):
        self.state = self.sampler.run_mcmc(
            self.state, n_samples, **self.run_args)

    def get_samples(self) -> McmcPtResult:
        trace_x = self.sampler.get_chain(flat=True)
        trace_fval = - self.sampler.get_log_prob(flat=True)

        result = McmcPtResult(
            trace_x=np.array([trace_x]),
            trace_fval=np.array([trace_fval]),
            betas=np.array([1.])
        )
        return result
