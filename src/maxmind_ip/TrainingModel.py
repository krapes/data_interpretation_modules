import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import datetime
import numpy as np
import random
import logging
import sys
import os


logging.basicConfig(level=logging.INFO)
logging.StreamHandler(sys.stdout)
logger = logging.getLogger(__name__)

import dask
import dask.dataframe as dd
from dask.distributed import Client
client = Client(n_workers=4, threads_per_worker=8, processes=False, memory_limit='5GB')


from typing import Dict, TypedDict, Any

from .utils import score, outcome, reconcile, today_result, build_weights, cal_impact

class CorridorThresholds(TypedDict):
    threshold_bottom: float
    thresold_top: float



class TrainingModel:
    dir_path = os.path.dirname(os.path.realpath(__file__))

    def __init__(self,
                 data: pd.DataFrame,
                 costs: Dict[str, int],
                 lookback: int = None,
                 step: int = None,
                 cutoff: int = 25,
                 repetitions: int = None) -> None:
        self.costs = costs
        self._cutoff = cutoff
        print(f"TrainingModel Data Size: {len(data)}")
        self._data = today_result(data, cutoff)
        self._repetitions = repetitions if repetitions is not None else 700
        self._lookback = (self.calibrate_lookback(self._data, step=step)
                          if lookback is None else lookback)
        self._step = (self.calibrate_step(self.data, self.lookback)
                      if step is None else step)
        self._data = build_weights(self._data, lookback=self._lookback)

    @property
    def lookback(self):
        """ The number of days (or 'time_delta' units) that should be considered when
            fitting the data
        """
        return self._lookback

    @lookback.setter
    def lookback(self, value: int) -> None:
        self._lookback = value
        self._data = self.build_weights(self._data, lookback=self._lookback)

    @property
    def cutoff(self):
        """ The threshold value used in the baseline ("today") approach
        """
        return self._cutoff

    @cutoff.setter
    def cutoff(self, value: int) -> None:
        self._cutoff = value
        self._data = self.today_result(self._data, cutoff)

    @property
    def repetitions(self):
        """ The number of threshold guesses that should be tried when fitting
        """
        return self._repetitions

    @repetitions.setter
    def repetitions(self, value: int) -> None:
        self._repetitions = value

    @property
    def data(self):
        """ Pandas DataFrame containing necessary data"""
        return self._data

    @property
    def step(self):
        """ The number of days between each refitting """
        return self._step

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
                threshold_bottom = random.randrange(1, 850, 1)
                threshold_top = random.randrange(threshold_bottom, 1000, 1)
                return threshold_bottom / 10, threshold_top / 10

            best_m = None
            corridor = g['corridor'].unique()[0]
            print(f"Starting Corridor {corridor}")

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


    def calibrate_thresholds(self, df: pd.DataFrame) -> Dict[str, CorridorThresholds]:
        """ Calculates the best thresholds for each corridor via the fit_function

            Args: df (DataFrame): data containing information necessary for calibration
                  verbose (bool): if True the function will log progress

            Returns dict: contains the best top and bottom threshold for each corridor
        """
        ddf = dask.dataframe.from_pandas(df[['corridor', 'risk_score', 'fraud', 'weight']].dropna(),
                                         npartitions=10)
        thresholds = self.fit_function(ddf)
        return thresholds

    def calibration(self, data: pd.DataFrame, parm: Dict[str, int]) -> (list, list):
        """ Simulates time by walking through the data in intervals of
            'step' and 1) fitting thresholds then 2) testing those thresholds
            on the next interval of data

            Args: data (DataFrame): data containing information necessary for calibration
                  parm (dict): A dictionary containing the lookback and step values to be used

            Returns: impacts (list): a list containing the difference in cost between the current
                                    approach and the new approach for each step interval
                    dates (list): a list containing the start date for each interval assiciated
                                    with the impacts list
        """

        date_min = data.when_created.min()
        date_max = data.when_created.max()
        date_max = datetime.datetime(date_max.year, date_max.month, date_max.day)
        date = datetime.datetime(date_min.year, date_min.month, date_min.day)

        impacts = []
        dates = []
        grp = -1
        while date < date_max:
            date += datetime.timedelta(days=parm['step'])
            grp += 1
            print(f"Date: {date}       Lookback: {parm['lookback']}   Step: {parm['step']}")
            df = build_weights(data[data['when_created'] < date].copy(), lookback=parm['lookback'])
            df = df[df['weight'] != 0]
            thresholds = self.calibrate_thresholds(df)

            today = data[(data['when_created'] >= date)
                         & (data['when_created'] < date + datetime.timedelta(days=parm['step']))].copy()
            today = score(today, thresholds, 'risk_score', 'real_result')
            today['weight'] = 1

            r, today = cal_impact(today, 'today_result', 'real_result', self.costs)
            data.loc[today.index, f"real_result_{parm['lookback']}_{parm['step']}"] = today['real_result']
            data.loc[today.index, f"real_result_cost_{parm['lookback']}_{parm['step']}"] = today['real_result_cost']
            data.loc[today.index, f"today_result_cost_{parm['lookback']}_{parm['step']}"] = today['today_result_cost']
            data.loc[today.index, 'grp'] = grp
            for corridor, g in today.groupby('corridor'):
                if corridor in thresholds.keys():
                    data.loc[g.index, 'thr_top'] = thresholds[corridor]['threshold_top']
                    data.loc[g.index, 'thr_bottom'] = thresholds[corridor]['threshold_bottom']
                else:
                    data.loc[g.index, 'thr_top'] = np.nan
                    data.loc[g.index, 'thr_bottom'] = np.nan
            impacts.append(r)
            dates.append(date)

        return impacts, dates

    @staticmethod
    def plot(matrix: Dict[Any, Any], title: str, save_loc: str, ax: plt.Axes = None) -> plt.Axes:
        """ Plots the x and y list for every key in matrix dictonary

            Args: matrix (dict): a dictionary containing:
                                {category_name: {x: [list of x values],
                                                 y: [list of y values]}}
                  title (str): Name of plot

            Returns lineplot
        """
        plt.figure(figsize=(15, 8))
        for i, l in enumerate(matrix.keys()):
            if ax is None:
                ax = sns.lineplot(x=matrix[l]['x'],
                                  y=matrix[l]['y'],
                                  label=l)
            else:
                ax = sns.lineplot(x=matrix[l]['x'],
                                  y=matrix[l]['y'],
                                  label=l,
                                  ax=ax)
        ax.set_title(title)
        plt.savefig(save_loc)
        return ax

    def evaluate(self, data: pd.DataFrame) -> None:
        """ Runs one calibration loop with the instance lookback and step values
        """
        self.calibration(data, {'lookback': self._lookback, 'step': self._step})
        data['real_result'] = data[f"real_result_{self._lookback}_{self._step}"]
        data['real_result_cost'] = data[f"real_result_cost_{self._lookback}_{self._step}"]
        data['today_result_cost'] = data[f"today_result_cost_{self._lookback}_{self._step}"]

    def calibrate_lookback(self, data: pd.DataFrame, step: int) -> int:
        """ Finds the best performing lookback window (window in time for data is considered
            when fitting the thresholds) from a discrete list of options.

            Args: data (DataFrame): data containing information necessary for calibration
                  step(int): value that should be used for the step during
                                  calibration.

            Return int: The best performing lookback value
        """

        if step is None:
            step = 14

        matrix = {}
        for lookback in [30, 90, 270, 365]:
            impacts, dates = self.calibration(data, {'lookback': lookback, 'step': step})
            matrix[lookback] = {'y': impacts, 'x': dates}
            title = 'Lookback_Results'
            self.plot(matrix, title, f"{self.dir_path}/plots/{title}.png")
        scores = [(l, sum(matrix[l]['y'])) for l in matrix.keys()]
        scores.sort(key=lambda tup: tup[1], reverse=True)
        return scores[0][0]

    def calibrate_step(self, data: pd.DataFrame, lookback: int) -> int:
        """ Finds the best performing step (how often (in time) the thresholds are refitted)
            from a discrete list of options

            Args: data (DataFrame): data containing information necessary for calibration
                  lookback(int): value that should be used for the lookback period during
                                  calibration.

            Return int: The best performing step value
        """

        matrix = {}
        for step in [30, 14, 7, 3]:
            impacts, dates = self.calibration(data, {'lookback': lookback, 'step': step})
            matrix[step] = {'y': impacts, 'x': dates}
            self.plot(matrix, 'Step_Results')

        scores = [(l, sum(matrix[l]['y'])) for l in matrix.keys()]
        scores.sort(key=lambda tup: tup[1], reverse=True)
        return scores[0][0]