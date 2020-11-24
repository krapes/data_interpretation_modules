import pandas as pd
import datetime
import json
import os
import logging
import time
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt

from .TrainingModel import TrainingModel
from .utils import predict, avg_value

from typing import Tuple

logger = logging.getLogger(__name__)


class MaxmindIp:
    dir_path = os.path.dirname(os.path.realpath(__file__))
    config_location = os.path.join(dir_path, 'config')
    _data = None

    def __init__(self,
                 ip: str = None,
                 username: str = None,
                 password: str = None,
                 port: int = None):
        self._config = self._load_config()
        self.ip = ip
        self.username = username
        self.password = password
        self.port = port

    @property
    def config(self):
        """Dictionary containing configuration information like costs, lookback, step,
            cutoff for the instance"""
        return self._config

    @config.setter
    def config(self, value):
        self._config = value

    @staticmethod
    def _load_config(folder: str = config_location) -> dict:
        """ Loads the configuration file for this defense

            Args:  folder(str): path to containing the configuration file

            Returns: dict: configuration dictionary
        """
        with open(f"{folder}/config.json") as json_file:
            config = json.load(json_file)
        return config

    def _save_config(self, folder: str = config_location) -> None:
        """ Saves the configuration dict to file

            Args: folder(str): path to containing the configuration file

            Returns: None
        """
        with open(f"{folder}/config.json", "w") as f:
            json.dump(self.config, f)

    def load_data(self, sample_size: int = None, date_range: tuple = None) -> pd.DataFrame:
        """ Loads data necessary for training/analyzing this defense

            Args: sample_size (int):  Size of the random sample that should be returned
                  date_range (tuple of datetimes): The start date (0) and end date (1) range of data
                                                    that should be returned

            Returns: Pandas Dataframe
        """

        def get_file(filename):
            return os.path.join(self.dir_path, filename)

        filenames = ['data/maxmind_scores.csv',
                     'data/maxmind_scores2.csv',
                     'data/maxmind_scores3.csv',
                     'data/maxmind_scores4.csv']

        data = pd.read_csv(get_file(filenames[0]))
        for filename in filenames[1:]:
            data = data.append(pd.read_csv(get_file(filename)), ignore_index=True)
        data['when_created'] = pd.to_datetime(data.when_created)
        print(f"data available: {len(data)}")
        if sample_size:
            logging.info(f"Using sample_size {sample_size}")
            data = data.sample(n=sample_size, random_state=1)
        if date_range:
            data = data[(data['when_created'] > date_range[0]) & (data['when_created'] < date_range[1])]
        print(f"data length: {len(data)}")
        return data

    def train(self,
              reset_lookback: bool = False,
              reset_step: bool = False,
              sample_size: bool = None,
              evaluate: bool = False,
              model_type: str = None,
              cost_matrix_loss_metric: bool = False,
              search_time: int = None
              ) -> pd.DataFrame:
        """ This function retrains/refits the data and overwrites the config with the results

            Args: reset_lookback(bool): Lookback is the window in time for data is considered
                                        when fitting the thresholds. If True, several windows will
                                        be tried to find the best performing
                  reset_step(bool): Step is how often (in time) the thresholds are refitted.
                                    Example: every day, every 7 days, etc.
                                    If True, several step intervals will be tried to find the
                                    best performing.
                  sample_size(int): The size of a random subset of the data that should be used
                  repetitions(int): When fitting the thresholds the algorthim tries several random
                                    examples. This variable defines how many, generally speaking the
                                    more repetitions the better the fit.

            Returns: Pandas Dataframe: training output

        """
        cutoff = self._config.get('cutoff', 25)
        if reset_lookback and reset_step:
            data = self.load_data(sample_size=sample_size)
            todays_update = TrainingModel(data,
                                          self._config['costs'],
                                          cutoff=cutoff,
                                          model_type=model_type,
                                          cost_matrix_loss_metric=cost_matrix_loss_metric,
                                          search_time=search_time,
                                          ip=self.ip,
                                          username=self.username,
                                          password=self.password,
                                          port=self.port)

            self.config['step'] = todays_update.step
            logger.info(f"Lookback length set to {self._config['lookback']}")
            logger.info(f"Step length set to {self._config['step']}")

        elif reset_lookback:
            data = self.load_data(sample_size=sample_size)
            step = self._config.get('step', 14)
            todays_update = TrainingModel(data,
                                          self._config['costs'],
                                          step=step,
                                          cutoff=cutoff,
                                          model_type=model_type,
                                          cost_matrix_loss_metric=cost_matrix_loss_metric,
                                          search_time=search_time,
                                          ip=self.ip,
                                          username=self.username,
                                          password=self.password,
                                          port=self.port
                                          )
            self._config['lookback'] = todays_update.lookback
            logger.info(f"Lookback length set to {self._config['lookback']}")

        elif reset_step:
            data = self.load_data(sample_size=sample_size)
            lookback = self.config.get('lookback', 90)
            todays_update = TrainingModel(data,
                                          self._config['costs'],
                                          lookback=lookback,
                                          cutoff=cutoff,
                                          model_type=model_type,
                                          cost_matrix_loss_metric=cost_matrix_loss_metric,
                                          search_time=search_time,
                                          ip=self.ip,
                                          username=self.username,
                                          password=self.password,
                                          port=self.port
                                          )
            self.config['step'] = todays_update.step
            logger.info(f"Step length set to {self._config['step']}")

        else:
            lookback = self._config.get('lookback', None)
            step = self._config.get('step', None)
            start = datetime.datetime.now() - datetime.timedelta(days=lookback)
            end = datetime.datetime.now()
            data = self.load_data(date_range=(start, end), sample_size=sample_size)
            todays_update = TrainingModel(data,
                                          self.config['costs'],
                                          lookback=lookback,
                                          step=step,
                                          cutoff=cutoff,
                                          model_type=model_type,
                                          cost_matrix_loss_metric=cost_matrix_loss_metric,
                                          search_time=search_time,
                                          ip=self.ip,
                                          username=self.username,
                                          password=self.password,
                                          port=self.port
                                          )

        if evaluate:
            self._data = todays_update.evaluate(data)

        self.config['model'] = todays_update.best_case_model
        print(todays_update.best_case_model, self.config['model'])
        self._save_config()

        return todays_update.data

    def inference(self, risk_score: int, send_country: str, receive_country: str) -> float:
        """Evaluates if a given event is risky (positive or 1) or not risky (negative  or 0)

        Args: risk_score (int): The risk_score given by the third party Maxmind
              send_country (str): User send country
              receive_country (str): User receive country

        Returns int: 0 for not risky behavior 1 for risky behavior
        """
        # TODO modify inference script
        corridor = f"{send_country}-{receive_country}"
        thresholds = self.config['thresholds']
        known_corridors = thresholds.keys()
        if corridor not in known_corridors:
            logger.warning(f"corridor {corridor} not found in known_corridors. Using average thresholds")
            threshold_bottom = avg_value('threshold_bottom', known_corridors, thresholds)
            threshold_top = avg_value('threshold_top', known_corridors, thresholds)
        else:
            corridor_thresholds = self.config['thresholds'][corridor]
            threshold_bottom = corridor_thresholds['threshold_bottom']
            threshold_top = corridor_thresholds['threshold_top']

        result = predict(pd.DataFrame({'risk_score': risk_score}, index=[0]),
                         threshold_bottom,
                         threshold_top,
                         'risk_score')[0]

        # Value returned as float to maintain consistency with receiving script
        return float(result)

    def plot_vs_baseline(self) -> plt.Axes:
        data = self._data
        plt.figure(figsize=(15, 8))
        ax = sns.lineplot(data=data, x='grp', y='real_result_cost', label='real_result_cost')
        ax = sns.lineplot(data=data, x='grp', y='today_result_cost', label='today_result_cost', ax=ax)
        cost_baseline = data.today_result_cost.sum()
        cost_new_approach = data.real_result_cost.sum()

        print(f"Cost Baseline: {cost_baseline}    Cost New Approach: {cost_new_approach}")
        print(f"Impact: {cost_new_approach - cost_baseline}")
        return ax

    def plot_step_lookback_analysis(self) -> Tuple[plt.Figure, plt.Axes]:

        if self._data is None:
            raise Exception("Instance does not contain self._data, run method train with evaluate=True")

        data = self._data
        fig, axes = plt.subplots(nrows=2, figsize=(15, 24))
        col_dic = {col: 'sum' for col in data.columns
                   if 'real_result_cost' in col or 'today_result_cost' in col}
        date_agg = data.groupby(pd.to_datetime(data.when_created).dt.month).agg(col_dic)

        for col in col_dic.keys():
            if col not in ['real_result_cost', 'today_result_cost']:
                date_agg[f"impact_{col}"] = date_agg[col] - date_agg['today_result_cost']
                sns.lineplot(data=date_agg, x=date_agg.index, y=col, ax=axes[0], label=col)
                if 'today' not in col:
                    sns.lineplot(data=date_agg, x=date_agg.index, y=f"impact_{col}", ax=axes[1], label=f"impact_{col}")
        return fig, axes

    def _volume_difference(self) -> int:
        if self._data is None:
            raise Exception("Instance does not contain self._data, run method train with evaluate=True")

        data = self._data
        data = data.loc[data.real_result.dropna().index, :]
        t = data[data['today_result'].isin(['tp', 'fp'])].when_created.count()
        r = data[data['real_result'].isin(['tp', 'fp'])].when_created.count()

        return t - r

    def cost_impact(self) -> int:
        data = self._data
        data = data.loc[data.real_result.dropna().index, :]
        r = data.real_result_cost.sum()
        t = data.today_result_cost.sum()
        return r - t

    def configure_volume_equals_baseline(self,
                                         search_time: int = -1,
                                         sample_size: int = None,
                                         model_type: str = 'AutoML',
                                         cost_matrix_loss_metric: bool = False) -> dict:
        def better_vol(v):
            return True if v > 0 else False

        def continue_searching(start_time, search_time, volume_diff):
            print(f"Elasped time: {round((time.time() - start_time) * 60, 2)} min")
            t = 500
            if search_time > 0:
                return (time.time() - start_time) < search_time and abs(volume_diff) > t
            else:
                if abs(volume_diff) < t:
                    print(f"Volume difference of {abs(volume_diff)} is less that {t} ... ending search.")
                    return False
                return True


        if search_time > 0  and search_time < 300:
            raise Exception("Search time must be greater than 300 seconds")

        last_difference = self._volume_difference()
        volume_diff = self._volume_difference()
        start_time = time.time()
        step = 2

        while continue_searching(start_time, search_time, volume_diff):

            step = step if np.sign(volume_diff) == np.sign(last_difference) else step + 1
            if better_vol(volume_diff) is False:
                self._config['costs']['cost_fn'] = self._config['costs']['cost_fn'] \
                                                   - (self._config['costs']['cost_fn'] / step)
            else:
                self._config['costs']['cost_fn'] = self._config['costs']['cost_fn'] * (1 + 1 / step)
            print(f"Using costs {self._config['costs']} \n"
                  f"Using step: {step}")

            data = self.train(reset_lookback=False,
                              reset_step=False,
                              sample_size=sample_size,
                              model_type=model_type,
                              cost_matrix_loss_metric=cost_matrix_loss_metric,
                              search_time=60*5,
                              evaluate=True)
            last_difference = volume_diff
            volume_diff = self._volume_difference()
            print(f"evaluated cost function: {self._config['costs']}\n" +
                  f"vol_diff: The new approach {'saved' if volume_diff > 0 else 'created'} {abs(volume_diff)} tickets \n" +
                  f"cost_diff: The new approach had and impact of {self.cost_impact()} \n" +
                  f"cost_saving: {self._data.today_result_cost.sum() - self._data.real_result_cost.sum()}\n")
        return self._config

    def stats_volume_vs_baseline(self) -> None:

        if self._data is None:
            raise Exception("Instance does not contain self._data, run method train with evaluate=True")

        data = self._data
        data = data.loc[data.real_result.dropna().index, :]

        t = data[data['today_result'].isin(['tp', 'fp'])].when_created.count()
        r = data[data['real_result'].isin(['tp', 'fp'])].when_created.count()
        print(f"Total Volume sent to UPS (TP + FP): \n\t Baseline: {t}  New Approach: {r}")
        print(f"The new approach has {'an increase' if r > t else 'a decrease'} of {abs(t - r)} tickets")
        print(f"This represents {'an increase' if r > t else 'a decrease'} of {(t - r) / t * 100} percent")

    def stats_tp_vs_baseline(self) -> None:
        if self._data is None:
            raise Exception("Instance does not contain self._data, run method train with evaluate=True")

        data = self._data
        data = data.loc[data.real_result.dropna().index, :]

        t = data[data['today_result'].isin(['tp'])].when_created.count()
        r = data[data['real_result'].isin(['tp'])].when_created.count()
        print(f"Total Fraud Cases Detected (TP): \n\t Baseline: {t}  New Approach: {r}")
        print(f"The new approach has {'an increase' if r > t else 'a decrease'} of {abs(t - r)} True Positives")
        print(f"This represents change of {(t - r) / t * 100} percent")

    def stats_fp_vs_baseline(self) -> None:
        if self._data is None:
            raise Exception("Instance does not contain self._data, run method train with evaluate=True")

        data = self._data
        data = data.loc[data.real_result.dropna().index, :]

        t = data[data['today_result'].isin(['fp'])].when_created.count()
        r = data[data['real_result'].isin(['fp'])].when_created.count()
        print(f"Total Fraud Cases Detected (FP): \n\t Baseline: {t}  New Approach: {r}")
        print(f"The new approach has {'an increase' if r > t else 'a decrease'} of {abs(t - r)} False Positives")
        print(f"This represents change of {(t - r) / t * 100} percent")
