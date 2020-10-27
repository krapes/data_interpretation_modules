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
import h2o
from h2o.automl import H2OAutoML
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.utils.distributions import CustomDistributionGaussian
from h2o.grid.grid_search import H2OGridSearch


logging.basicConfig(level=logging.INFO)
logging.StreamHandler(sys.stdout)
logger = logging.getLogger(__name__)
'''
import dask
import dask.dataframe as dd
from dask.distributed import Client
client = Client(n_workers=4, threads_per_worker=8, processes=False, memory_limit='5GB')
'''

from typing import Dict, TypedDict, Any

from .utils import score, outcome, reconcile, today_result, build_weights, cal_impact, predict
from .utils_model_metrics import WeightedFalseNegativeLossMetric, CostMatrixLossMetric


class CorridorThresholds(TypedDict):
    threshold_bottom: float
    thresold_top: float




class TrainingModel:
    h2o.init()

    #weighted_false_negative_loss_func = h2o.upload_custom_metric(WeightedFalseNegativeLossMetric,
    #                                                             func_name="WeightedFalseNegativeLoss",
    #                                                             func_file="weighted_false_negative_loss.py")
    cost_matrix_loss_metric_func = h2o.upload_custom_metric(CostMatrixLossMetric,
                                                                 func_name="CostMatrixLossMetric",
                                                                 func_file="cost_matrix_loss_metric.py")

    dir_path = os.path.dirname(os.path.realpath(__file__))
    _best_case_model = None

    def __init__(self,
                 data: pd.DataFrame,
                 costs: Dict[str, int],
                 lookback: int = None,
                 step: int = None,
                 cutoff: int = 25,
                 repetitions: int = None) -> None:
        print(f"lookback: {lookback}  step: {step}")
        self.costs = costs
        self._cutoff = cutoff
        print(f"TrainingModel Data Size: {len(data)}")
        self._data = today_result(data, cutoff)
        # self._repetitions = repetitions if repetitions is not None else 700
        self._lookback = (self.calibrate_lookback(self._data, step=step)
                          if lookback is None else lookback)
        self._step = (self.calibrate_step(self.data, self.lookback)
                      if step is None else step)
        self._data = build_weights(self._data, lookback=self._lookback)




    @property
    def lookback(self) -> int:
        """ The number of days (or 'time_delta' units) that should be considered when
            fitting the data
        """
        return self._lookback

    @lookback.setter
    def lookback(self, value: int) -> None:
        self._lookback = value
        self._data = self.build_weights(self._data, lookback=self._lookback)

    @property
    def cutoff(self) -> int:
        """ The threshold value used in the baseline ("today") approach
        """
        return self._cutoff

    @cutoff.setter
    def cutoff(self, value: int) -> None:
        self._cutoff = value
        self._data = self.today_result(self._data, cutoff)

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
        if self._best_case_model is None:
            model = self.calibrate_thresholds(self._data)
            if path is None:
                path = os.path.join(self.dir_path, 'models/')
            self._best_case_model = h2o.save_model(model, path=path, force=True)
            print(type(self._best_case_model))
        return self._best_case_model



    def train(self, train, x, y, weight):
        '''
        gboost = H2OGradientBoostingEstimator(custom_metric_func=self.weighted_false_negative_loss_func)
        gboost.train(x=x, y=y,
                  training_frame=train,
                  weights_column=weight
                  )
        '''
        def sort_models(gbm_grid: object) -> list:
            functioning_list_of_models = []
            for model_name in gbm_grid.model_ids:
                try:
                    result = [h2o.get_model(model_name).model_performance(xval=True).custom_metric_value(), model_name]
                    functioning_list_of_models.append(result)
                except AttributeError:
                    print(f"Error with {x}")
                    pass

            outputs = sorted(functioning_list_of_models)
            for output in outputs:
                print(output)
            return outputs

        gbm_hyper_parameters = {'learn_rate': [0.01, 0.1]}#,
                                #'max_depth': [3, 5, 9],
                                #'sample_rate': [0.8, 1.0],
                                #'col_sample_rate': [0.2, 0.5, 1.0]}
        print(gbm_hyper_parameters)
        gbm_grid = H2OGridSearch(H2OGradientBoostingEstimator(custom_metric_func=self.cost_matrix_loss_metric_func,
                                                              nfolds=3),
                                 gbm_hyper_parameters)
        gbm_grid.train(x=x, y=y, training_frame=train, weights_column="weight", grid_id="gbm_grid")

        best_model = h2o.get_model(sort_models(gbm_grid)[0][1])
        #best_model.show()
        train_pd = train.as_data_frame()
        train_pd['model_score'] = best_model.predict(test_data=train).as_data_frame()['p1']
        matrix = {'basic': {'x': [], 'y': []}}
        for t in range(1, 100):
            t = t/100
            train_pd['prediction'] = predict(train_pd, t, 1, 'model_score')
            train_pd = reconcile(train_pd, 'prediction', 'fraud', f"CM_{t}")
            t_cost, train_pd = outcome(train_pd, {'cost_tp': 0, 'cost_fp': 60, 'cost_fn': 2400000, 'cost_tn': -0.1}, f"CM_{t}", f"costs_{t}")
            matrix['basic']['x'].append(t)
            matrix['basic']['y'].append(t_cost)
        title = 'threshold_calculation'
        self.plot(matrix, title, f"{self.dir_path}/plots/{title}.png")
        #cost_metric = CostMatrixLossMetric()
        #train_pd['cost'] = train_pd.apply(lambda row: cost_metric.map([None, None, row.prediction], [row.fraud], row.weight, row.index, row.index)[0], axis=1)
        print(train_pd[[f"costs_{t}", 'prediction', 'fraud', 'model_score']])
        print(train_pd.model_score.max(), train_pd.model_score.min())

        optimum_threshold = matrix['basic']['x'][matrix['basic']['y'].index(min(matrix['basic']['y']))]
        print(f"optimum_threshold: {optimum_threshold}")
        print(best_model.confusion_matrix(thresholds=optimum_threshold))

        return best_model

    def df_to_hf(self, df: pd.DataFrame, cols: list, cat_cols: list):
        cleaned_df = df[cols].dropna()
        hf = h2o.H2OFrame(cleaned_df)
        for col in cat_cols:
            hf[col] = hf[col].asfactor()
        return hf, cleaned_df

    def calibrate_thresholds(self, df: pd.DataFrame) -> Dict[str, CorridorThresholds]:
        """ Calculates the best thresholds for each corridor via the fit_function

            Args: df (DataFrame): data containing information necessary for calibration
                  verbose (bool): if True the function will log progress

            Returns dict: contains the best top and bottom threshold for each corridor
        """
        #ddf = dask.dataframe.from_pandas(df[['corridor', 'risk_score', 'fraud', 'weight']].dropna(),
        #                                 npartitions=10)
        #thresholds = self.fit_function(ddf)
        train, _ = self.df_to_hf(df, ['corridor', 'risk_score', 'fraud', 'weight'], ['corridor', 'fraud'])
        model = self.train(train,
                            ['corridor', 'risk_score'],
                           'fraud',
                           'weight')
        return model





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
            model = self.calibrate_thresholds(df)

            today = data[(data['when_created'] >= date)
                         & (data['when_created'] < date + datetime.timedelta(days=parm['step']))].copy()
            #today = score(today, thresholds, 'risk_score', 'real_result')
            hf_today, df_w_drops = self.df_to_hf(today, ['corridor', 'risk_score', 'fraud'], ['corridor'])
            predictions = model.predict(test_data=hf_today).as_data_frame()
            today.loc[df_w_drops.index, 'prediction'] = predictions.predict.to_list()
            #today['prediction'] = today.prediction.apply(lambda x: 0 if x < 0 else 1)
            today = reconcile(today, 'prediction', 'fraud', 'real_result')

            today['weight'] = 1

            r, today = cal_impact(today, 'today_result', 'real_result', self.costs)
            data.loc[today.index, f"real_result_{parm['lookback']}_{parm['step']}"] = today['real_result']
            data.loc[today.index, f"real_result_cost_{parm['lookback']}_{parm['step']}"] = today['real_result_cost']
            data.loc[today.index, f"today_result_cost_{parm['lookback']}_{parm['step']}"] = today['today_result_cost']
            data.loc[today.index, 'grp'] = grp
            '''
            for corridor, g in today.groupby('corridor'):
                if corridor in thresholds.keys():
                    data.loc[g.index, 'thr_top'] = thresholds[corridor]['threshold_top']
                    data.loc[g.index, 'thr_bottom'] = thresholds[corridor]['threshold_bottom']
                else:
                    data.loc[g.index, 'thr_top'] = np.nan
                    data.loc[g.index, 'thr_bottom'] = np.nan
            '''
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