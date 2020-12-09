import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import datetime

import logging
import sys
import os
import h2o


pd.options.mode.chained_assignment = None

logging.basicConfig(level=logging.INFO)
logging.StreamHandler(sys.stdout)
logger = logging.getLogger(__name__)


from typing import Dict

from .utils import today_result, build_weights, cal_impact
from .TopBottomThreshold import TopBottomThreshold
from .H2OModel import H20Model


class TrainingModel:
    dir_path = os.path.dirname(os.path.realpath(__file__))
    _best_case_model = None

    def __init__(self,
                 data: pd.DataFrame,
                 costs: Dict[str, int],
                 lookback: int = None,
                 step: int = None,
                 cutoff: int = 25,
                 model_type: str = None,
                 cost_matrix_loss_metric: bool = False,
                 search_time: int = None,
                 ip: str = None,
                 username: str = None,
                 password: str = None,
                 port: int = None,
                 repetitions: int = 700) -> None:

        self.costs = costs

        self._cutoff = cutoff
        self._data = today_result(data, cutoff)
        self.model_type = model_type
        if model_type == 'TopBottomThreshold':
            self.model_shell = TopBottomThreshold()
            self.model_shell._repetitions = repetitions
        elif model_type == 'GradientBoosting' or model_type == 'AutoML':
            self.model_shell = H20Model(ip=ip,
                                        username=username,
                                        password=password,
                                        port=port)
            self.model_shell.cost_matrix_loss_metric = cost_matrix_loss_metric
            self.model_shell.search_time = search_time if search_time is not None else int(60 * 1.5)
            self.model_shell.inverse_costs = {key: value * -1 for (key, value) in costs.items()}
            self.model_shell.model_type = model_type

        self._lookback = (self.calibrate_lookback(self._data, step=step)
                          if lookback is None else lookback)
        self._step = (self.calibrate_step(self.data, self.lookback)
                      if step is None else step)
        self._data = build_weights(self._data, lookback=self._lookback)
        self.model_shell._lookback = self._lookback
        self.model_shell._step = self._step
        self.model_shell.costs = self.costs
        self.model_shell._data = self._data

    @property
    def lookback(self) -> int:
        """ The number of days (or 'time_delta' units) that should be considered when
            fitting the data
        """
        return self._lookback

    @lookback.setter
    def lookback(self, value: int) -> None:
        self._lookback = value
        self._data = build_weights(self._data, lookback=self._lookback)

    @property
    def cutoff(self) -> int:
        """ The threshold value used in the baseline ("today") approach
        """
        return self._cutoff

    @cutoff.setter
    def cutoff(self, value: int) -> None:
        self._cutoff = value
        self._data = self.today_result(self._data, self._cutoff)

    @property
    def repetitions(self) -> int:
        """ The number of threshold guesses that should be tried when fitting
        """
        return self._repetitions

    @repetitions.setter
    def repetitions(self, value: int) -> None:
        self._repetitions = value

    @property
    def data(self) -> pd.DataFrame:
        """ Pandas DataFrame containing necessary data"""
        return self._data

    @property
    def step(self) -> int:
        """ The number of days between each refitting """
        return self._step

    @property
    def best_case_model(self, path: str = None) -> str:
        """ The best model that can be produced for future predictions
        Args:
            path (str): Optional location of saved model
        Return:
        """
        if self._best_case_model is None:
            self.model_shell.calibrate_thresholds(self._data)
            if path is None:
                path = os.path.join(self.dir_path, 'models/')
            self._best_case_model = h2o.save_model(self.model_shell.model, path=path, force=True)
            print(type(self._best_case_model))
        return self._best_case_model


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

            # Train model
            df = build_weights(data[data['when_created'] < date].copy(), lookback=parm['lookback'])
            df = df[df['weight'] != 0]
            self.model_shell.calibrate_thresholds(df)

            # Take the data one step in the future of the date and evaluate how the model would have done
            today = data[(data['when_created'] >= date)
                         & (data['when_created'] < date + datetime.timedelta(days=parm['step']))].copy()
            today = self.model_shell.score(today)
            today['weight'] = 1

            # Calculate the monetary impact of the model verse the the baseline
            r, today = cal_impact(today, 'today_result', 'real_result', self.costs)
            data.loc[today.index, f"real_result_{parm['lookback']}_{parm['step']}"] = today['real_result']
            data.loc[today.index, f"real_result_cost_{parm['lookback']}_{parm['step']}"] = today['real_result_cost']
            data.loc[today.index, f"today_result_cost_{parm['lookback']}_{parm['step']}"] = today['today_result_cost']
            data.loc[today.index, 'grp'] = grp

            impacts.append(r)
            dates.append(date)

        return impacts, dates

    @staticmethod
    def plot(matrix: dict, title: str, save_loc: str, ax: plt.Axes = None) -> plt.Axes:
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
        self._data = data
        return data

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
            _ = self.plot(matrix, title, f"{self.dir_path}/plots/{title}.png")
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
