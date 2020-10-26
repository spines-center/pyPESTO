import os

import numpy as np

from ..objective import CsvHistory


class StartingPointsFromHistory:

    def __init__(self, source):
        self.source = source
        self.starting_points = None

        self._get_starting_points()

    @property
    def n_starts(self) -> int:
        return self.starting_points.shape[0]

    def __call__(self, **kwargs) -> np.ndarray:
        return self.starting_points

    def _get_starting_points(self):
        if os.path.isdir(self.source):
            starting_points = []
            for file_name in os.listdir(self.source):
                history = CsvHistory(
                    file=os.path.join(self.source, file_name),
                    load_from_file=True
                )
                xs = history.get_x_trace(-1)
                # get_x_trace()
                # xs.reshape(1, len(xs))
                starting_points.append(xs)
            self.starting_points = np.array(starting_points)
        else:

            history = CsvHistory(
                file=self.source,
                load_from_file=True
            )
            starting_points = history.get_x_trace(-1)
            self.starting_points = starting_points.reshape(
                1, len(starting_points))
