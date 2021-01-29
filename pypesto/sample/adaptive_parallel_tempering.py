from typing import Dict, Sequence, Union
import numpy as np

from .parallel_tempering import ParallelTemperingSampler
from .sampler import InternalSample


class AdaptiveParallelTemperingSampler(ParallelTemperingSampler):
    """Parallel tempering sampler with adaptive temperature adaptation."""

    @classmethod
    def default_options(cls) -> Dict:
        options = super().default_options()
        # scaling factor for temperature adaptation
        options['eta'] = 100
        # controls the adaptation degeneration velocity of the temperature
        # adaption.
        options['nu'] = 1e3

        return options

    def adjust_betas(self, i_sample: int, in_swapped: Sequence[Union[None, bool, InternalSample]]):
        """Update temperatures as in Vousden2016."""
        if len(self.betas) == 1:
            return

        # parameters
        nu = self.options['nu']
        eta = self.options['eta']
        betas = self.betas

        # booleans to integer array
        # swapped = np.array([int(swap) for swap in in_swapped])
        swapped = np.zeros_like(in_swapped)
        for idx, swap in enumerate(in_swapped):
            if swap is bool:
                swapped[idx] = int(swap)
            elif swap is InternalSample:
                swapped[idx] = 1

        # update betas
        kappa = nu / (i_sample + 1 + nu) / eta
        ds = kappa * (swapped[:-1] - swapped[1:])
        dtemp = np.diff(1. / betas[:-1])
        dtemp = dtemp * np.exp(ds)
        betas[:-1] = 1 / np.cumsum(np.insert(dtemp, obj=0, values=1.))

        # fill in
        self.betas = betas
