import logging
from typing import Dict, List, Sequence, Union
from tqdm import tqdm
import numpy as np
import copy
from multiprocess import Manager, Queue, Process, Event
import queue

from ..problem import Problem
from .sampler import Sampler, InternalSampler, InternalSample
from .result import McmcPtResult


logger = logging.getLogger(__name__)


class ParallelTemperingSamplerWorker(Process):
    def __init__(self, work_queue: Queue, return_queue: Queue, final_queue: Queue,
                 idx: int, sampler_obj: InternalSampler):
        super().__init__()
        self._exit = Event()
        self._q = work_queue
        self._r = return_queue
        self._f = final_queue
        self._idx = idx
        self._sampler = sampler_obj

    def run(self) -> None:
        while not self._exit.is_set():
            try:
                idx, new_last_sample, beta, stop = self._q.get(block=True, timeout=0.01)
                if idx != self._idx:
                    # logger.debug(f'sampler {self._idx} got {idx}')
                    self._q.task_done()
                    self._q.put((idx, new_last_sample, beta, stop))
                    continue
                if stop:
                    # logger.debug(f'STOPPING sampler {self._idx} trace_x: {self._sampler.trace_x}')
                    self._f.put((self._idx, copy.deepcopy(self._sampler)))
                    self._q.task_done()
                    return
                if new_last_sample is not None:
                    self._sampler.set_last_sample(new_last_sample)
                self._sampler.sample(n_samples=1, beta=beta)
                # logger.debug(f'sampler {idx} trace_x: {self._sampler.trace_x}')
                self._r.put((idx, self._sampler.get_last_sample(), beta))
                self._q.task_done()
            except (EOFError, queue.Empty):
                continue

    def terminate(self) -> None:
        self._exit.set()


class ParallelTemperingSampler(Sampler):
    """Simple parallel tempering sampler."""

    def __init__(
            self,
            internal_sampler: InternalSampler,
            betas: Sequence[float] = None,
            n_chains: int = None,
            options: Dict = None):
        super().__init__(options)

        # set betas
        if (betas is None) == (n_chains is None):
            raise ValueError("Set either betas or n_chains.")
        if betas is None:
            betas = near_exponential_decay_betas(
                n_chains=n_chains, exponent=self.options['exponent'],
                max_temp=self.options['max_temp'])
        if betas[0] != 1.:
            raise ValueError("The first chain must have beta=1.0")
        self.betas0 = np.array(betas)
        self.betas = None

        self.temper_lpost = self.options['temper_log_posterior']

        self.samplers = [copy.deepcopy(internal_sampler)
                         for _ in range(len(self.betas0))]
        # configure internal samplers
        for sampler in self.samplers:
            sampler.make_internal(temper_lpost=self.temper_lpost)

    @classmethod
    def default_options(cls) -> Dict:
        return {
            'max_temp': 5e4,
            'exponent': 4,
            'temper_log_posterior': False,
        }

    def initialize(self,
                   problem: Problem,
                   x0: Union[np.ndarray, List[np.ndarray]]):
        # initialize all samplers
        n_chains = len(self.samplers)
        if isinstance(x0, list):
            x0s = x0
        else:
            x0s = [x0 for _ in range(n_chains)]
        for sampler, x0 in zip(self.samplers, x0s):
            _problem = copy.deepcopy(problem)
            sampler.initialize(_problem, x0)
        self.betas = self.betas0

    def sample(self,
               n_samples: int,
               beta: float = 1.):
        # loop over iterations
        # TODO: add switch for pool type (multiprocessing, multiprocess, MPI, or None)
        with Manager() as mgr:
            workqueue = mgr.Queue(maxsize=len(self.samplers))
            donequeue = mgr.Queue(maxsize=len(self.samplers))
            finalqueue = mgr.Queue(maxsize=len(self.samplers))
            workers = [ParallelTemperingSamplerWorker(workqueue, donequeue, finalqueue, idx, copy.deepcopy(sampler))
                       for idx, sampler in enumerate(self.samplers)]
            # for worker in workers:
            #     worker.daemon = True
            [worker.start() for worker in workers]

            swapped = [None for _ in self.samplers]
            last_samples = [None for _ in self.samplers]
            for i_sample in tqdm(range(int(n_samples))):
                # sample
                for idx, beta in enumerate(self.betas):
                    workqueue.put((idx, swapped[idx], beta, False))
                    # sampler.sample(n_samples=1, beta=beta)
                workqueue.join()  # blocks until all samplers have processed an item

                for _ in range(len(self.samplers)):
                    idx, last_sample, beta = donequeue.get()
                    last_samples[idx] = last_sample

                # swap samples
                swapped = self.swap_samples(last_samples)

                # adjust temperatures
                self.adjust_betas(i_sample, swapped, last_samples)
            # logger.debug('stopping workers...')
            [workqueue.put((idx, None, 0.00, True)) for idx, _ in enumerate(self.samplers)]
            workqueue.join()
            # logger.debug('reached getting from finalqueue')
            for _ in self.samplers:
                idx, sampler_obj = finalqueue.get()
                # logger.debug(f'GATHERED sampler {idx} trace_x: {sampler_obj.trace_x}')
                self.samplers[idx] = sampler_obj
            [worker.join() for worker in workers]
            [worker.terminate() for worker in workers]
            # logger.debug('joined all workers')

    def get_samples(self) -> McmcPtResult:
        """Concatenate all chains."""
        results = [sampler.get_samples() for sampler in self.samplers]
        trace_x = np.array([result.trace_x[0] for result in results])
        trace_neglogpost = np.array([result.trace_neglogpost[0]
                                     for result in results])
        trace_neglogprior = np.array([result.trace_neglogprior[0]
                                      for result in results])
        return McmcPtResult(
            trace_x=trace_x,
            trace_neglogpost=trace_neglogpost,
            trace_neglogprior=trace_neglogprior,
            betas=self.betas
        )

    def swap_samples(self, last_samples: List[InternalSample]) -> List[Union[InternalSample, None]]:
        """Swap samples as in Vousden2016."""
        # for recording swaps
        swapped = copy.deepcopy(last_samples)

        if len(self.betas) == 1:
            # nothing to be done
            return swapped

        # beta differences
        dbetas = self.betas[:-1] - self.betas[1:]

        # loop over chains from highest temperature down
        # TODO: instead of sampler1, sampler2 -- use indices

        # for dbeta, sampler1, sampler2 in reversed(
        #         list(zip(dbetas, self.samplers[:-1], self.samplers[1:]))):
        for dbeta, sampler1_idx, sampler2_idx in reversed(list(zip(
                dbetas, list(range(len(self.samplers[:-1]))), list(range(len(self.samplers[1:])))))):
            # extract samples
            sample1 = last_samples[sampler1_idx]
            sample2 = last_samples[sampler2_idx]

            # extract log likelihood values
            sample1_llh = sample1.lpost - sample1.lprior
            sample2_llh = sample2.lpost - sample2.lprior

            # swapping probability
            p_acc_swap = dbeta * (sample2_llh - sample1_llh)

            # flip a coin
            u = np.random.uniform(0, 1)

            # check acceptance
            swap = np.log(u) < p_acc_swap
            if swap:
                # swap
                # sampler2.set_last_sample(sample1)
                # sampler1.set_last_sample(sample2)
                swapped[sampler2_idx] = sample1
                swapped[sampler1_idx] = sample2
            else:
                swapped[sampler2_idx] = sample2
                swapped[sampler1_idx] = sample1
            # record
            # swapped.insert(0, swap)
        return swapped

    def adjust_betas(self, i_sample: int,
                     swapped: Sequence[Union[None, InternalSample]],
                     last_samples: Sequence[Union[None, InternalSample]]):
        """Adjust temperature values. Default: Do nothing."""


def near_exponential_decay_betas(
        n_chains: int, exponent: float, max_temp: float) -> np.ndarray:
    """Initialize betas in a near-exponential decay scheme.

    Parameters
    ----------
    n_chains:
        Number of chains to use.
    exponent:
        Decay exponent. The higher, the more small temperatures are used.
    max_temp:
        Maximum chain temperature.
    """
    # special case of one chain
    if n_chains == 1:
        return np.array([1.])

    temperatures = np.linspace(1, max_temp ** (1 / exponent), n_chains) \
        ** exponent
    betas = 1 / temperatures

    return betas
