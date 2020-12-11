import logging
import os
import pandas as pd
import random
from typing import Dict, TypedDict

import dask
import dask.dataframe as dd
from dask.distributed import Client

from .utils import score, outcome


class CorridorThresholds(TypedDict):
    threshold_bottom: float
    threshold_top: float


class TopBottomThreshold:
    dir_path = os.path.dirname(os.path.realpath(__file__))
    _repetitions = None
    costs = None

    def __init__(self, *args, **kwargs):
        client = Client(n_workers=4, threads_per_worker=8, processes=False, memory_limit='10GB', silence_logs='error')

    def calibrate_thresholds(self, df: pd.DataFrame, **kwargs):
        """
        Produce a model with the data provided in df

        Args:
            df (dataframe): training data
        """
        if self._repetitions is None:
            raise Exception(f"TopBottomThreshold._repetitions cannot be None")
        if self.costs is None:
            raise Exception("TopBottomThreshold.costs cannot be None")

        ddf = dask.dataframe.from_pandas(df[['corridor', 'risk_score', 'fraud', 'weight']].dropna(),
                                         npartitions=10)
        self.thresholds = self.fit_function(ddf)

    def score(self, df: pd.DataFrame):
        df = score(df, self.thresholds, 'risk_score', 'real_result')
        return df

    def fit_function(self, df: pd.DataFrame) -> Dict[str, CorridorThresholds]:
        """ This outer function is the setup for the inner spark-pandas_udf fitting
            function. Here we define the costs dictionary, response schema, and
            repetition parameter.

            Args: df (Spark DataFrame): Contains data entries with risk_score, corridor,
                                        weight and fraud

            Returns: DataFrame: Contains fitted thresholds for each corridor
        """
        costs = self.costs
        repetitions = self._repetitions

        def fit(g: dask.dataframe) -> Dict[str, float]:
            """ Finds the best scoring top and bottom threshold combinations for
                data in dataframe g.

                Args: g (Spark DataFrame of one grouped_map): Entries for fitting

                Returns: (pandas DataFrame): one line dataframe containing fit results
            """

            def get_thresholds() -> (float, float):
                """ Generates top and bottom threshold guesses. Top must be
                    larger than bottom.

                    Returns: int, int: threshold_bottom, threshold_top
                """
                threshold_bottom = random.randrange(250, 850, 1)
                threshold_top = random.randrange(threshold_bottom, 1000, 1)
                return threshold_bottom / 10, threshold_top / 10

            best_m = None
            best_threshold_bottom = None
            best_threshold_top = None
            corridor = g['corridor'].unique()[0]
            logging.info(f"Starting Corridor {corridor}")

            # Thresholds are randomly tried for number 'repetitions' times
            for _ in range(1, repetitions):
                threshold_bottom, threshold_top = get_thresholds()

                g = score(g,
                          {corridor: {"threshold_top": threshold_top,
                                      "threshold_bottom": threshold_bottom}},
                          'risk_score',
                          'result')

                m_effect, _ = outcome(g, costs, 'result')

                # The most effective combination is saved
                if best_m is None or m_effect > best_m:
                    best_threshold_bottom = threshold_bottom
                    best_threshold_top = threshold_top
                    best_m = m_effect

            return {"threshold_top": best_threshold_top,
                    "threshold_bottom": best_threshold_bottom}

        results = df.groupby('corridor').apply(fit, meta=object).compute().sort_index()
        results = {key: value for key, value in zip(results.index, results)}
        return results
