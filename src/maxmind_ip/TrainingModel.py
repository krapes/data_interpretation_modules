import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import datetime
import time
import logging
import sys
import os
import h2o
from h2o.automl import H2OAutoML
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.grid.grid_search import H2OGridSearch
from h2o.estimators import H2OGenericEstimator
import re

root = logging.getLogger()
root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
logging.StreamHandler(sys.stdout)
logger = logging.getLogger(__name__)

from typing import Dict, Tuple, List

from .utils import outcome, reconcile, today_result, build_weights, cal_impact, predict


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
                 port: int = None) -> None:
        if ip is not None or username is not None or password is not None or port is not None:
            if ip is None or username is None or password is None or port is None:
                raise Exception("If using a remote H2O cluster ALL of following fields must be present"
                                "  [ip, username, password, port]")
            else:
                h2o.init(ip=ip, username=username, password=password, port=port)
        else:
            h2o.init()
        self.costs = costs
        self.inverse_costs = {key: value * -1 for (key, value) in costs.items()}
        self._cutoff = cutoff
        self.model_type = model_type
        self.cost_matrix_loss_metric = cost_matrix_loss_metric
        self.search_time = search_time if search_time is not None else 60*10
        self._data = today_result(data, cutoff)
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
            model, threshold = self.calibrate_thresholds(self._data)
            if path is None:
                path = os.path.join(self.dir_path, 'models/')
            self._best_case_model = h2o.save_model(model, path=path, force=True)
            print(type(self._best_case_model))
        return self._best_case_model

    def optimum_threshold(self, hf: h2o.H2OFrame, model: H2OGenericEstimator) -> float:
        """ Selects the best threshold for this model given the cost values of this instance

        Args:
            df (DataFrame): Data used for evaluation. Must contain ground truth column named fraud
            model (H2OModel): A model object to be evaluated
        Returns: optimum_threshold (float): Indicates that if a model p1 value is less than this number
                                            the prediction is 0 (not fraud). If the model p1 value is greater than
                                            this number the prediction is 1 (fraud)
        """
        # Extract the probability of the positive class from the predictions
        df = hf.as_data_frame()
        df['model_score'] = model.predict(test_data=hf).as_data_frame()['p1']

        matrix = {str(model.model_id): {'x': [], 'y': []}}
        # Calculate cost function for ever 1/100 ranging from 0 to 1
        for t in range(1, 100):
            t = t / 100
            df['prediction'] = predict(df, t, 1, 'model_score')
            df = reconcile(df, 'prediction', 'fraud', f"CM_{t}")
            t_cost, df = outcome(df, self.inverse_costs, f"CM_{t}", f"costs_{t}")
            matrix[str(model.model_id)]['x'].append(t)
            matrix[str(model.model_id)]['y'].append(t_cost)
        title = f'threshold_calculation_{model.model_id}'
        self.plot(matrix, title, f"{self.dir_path}/plots/{title}.png")

        # Return threshold that produced the minimum cost
        idx_min_cost = matrix[str(model.model_id)]['y'].index(min(matrix[str(model.model_id)]['y']))
        optimum_threshold = matrix[str(model.model_id)]['x'][idx_min_cost]
        print(f"optimum_threshold: {optimum_threshold}")
        return optimum_threshold

    def train_gradientboosting(self, train: h2o.H2OFrame,
                               x: List[str],
                               y: str,
                               weight: str,
                               cost_matrix_loss_metric: bool) -> H2OGenericEstimator:
        """ Use a  H2O gradient boosting base model and a gridsearch to build model

        Args:
            train (h2o dataframe): training data containing columns x, y, and weight
            x (list of str): column names of model features
            y (list of str): column name of ground truth
            weight (str): column name of row weights

        Return
            H2OGenericEstimator: best model out of the training grid

        """

        def sort_models(grid: H2OGridSearch) -> List[list]:
            """ Sorts models in the grid by their custom_metric_value or the score reported by the custom
                metric set at model declaration.
            Args:
                grid (H2OGridSearch): a grid search object containing models with the custom metric
            Returns:
                Sorted list of decreasing custom_metric_value
            """
            functioning_list_of_models = []
            for model_name in grid.model_ids:
                try:
                    result = [h2o.get_model(model_name).model_performance(xval=True).custom_metric_value(), model_name]
                    functioning_list_of_models.append(result)
                except AttributeError:
                    # Some models fail because they don't have a custom_metric_value, it's unclear why at this time
                    print(f"Error with {x}")
                    pass

            return sorted(functioning_list_of_models)

        def grid_train(base_model: H2OGradientBoostingEstimator, search_time: int) -> H2OGridSearch:
            """ Given base model train a search grid to find the optimum hyper parameters
            Args:
                base_model (H2OGradientBoostingEstimator): model that should be used in hyper parameter search
            Return:
                H2OGridSearch : trained grid
            """
            gbm_hyper_parameters = {'learn_rate': [0.01, 0.1],
                                    'max_depth': [3, 5, 9],
                                    'sample_rate': [0.8, 1.0],
                                    'col_sample_rate': [0.2, 0.5, 1.0]}
            logger.info(f"Searching Hyper Parameter Space:\n {gbm_hyper_parameters}")
            grid = H2OGridSearch(base_model,
                                 gbm_hyper_parameters,
                                 search_criteria={'strategy': "RandomDiscrete", 'max_runtime_secs': search_time})
            grid.train(x=x, y=y, training_frame=train, weights_column=weight, grid_id="gbm_grid")
            return grid

        def get_cost_matrix_loss_metric_class() -> object:
            """ This function modifies the text in the file utils_model_metrics to include the cost dictionary in
                this instance before importing the file. The strategy is messy and I don't believe it is the correct
                way to do this, but it is the only way I could find to complete the tasks inside the allotted time
                today.
            Returns the class CostMatrixLossMetric with cost dictionary overwritten
            """
            file_path = os.path.join(self.dir_path, 'utils_model_metrics.py')
            with open(file_path, 'r') as file:
                file_data = file.read()
            target = r"\{'cost_tp': -?\d*\.?\d, 'cost_fp': -?\d*\.?\d, 'cost_tn': -?\d*\.?\d*, 'cost_fn': -?\d*\.?\d*\}"
            file_data = re.sub(target, str(self.inverse_costs), file_data)
            with open(file_path, 'w') as file:
                file.write(file_data)
                print("file written")

            from .utils_model_metrics import CostMatrixLossMetric
            return CostMatrixLossMetric

        if cost_matrix_loss_metric:
            # If cost_matrix_loss_metric upload it to cluster and include it in base model

            cost_matrix_loss_metric_func = h2o.upload_custom_metric(get_cost_matrix_loss_metric_class(),
                                                                    func_name="CostMatrixLossMetric",
                                                                    func_file="cost_matrix_loss_metric.py")
            base_model = H2OGradientBoostingEstimator(custom_metric_func=cost_matrix_loss_metric_func,
                                                      nfolds=3)
            gbm_grid = grid_train(base_model, self.search_time)
            # Custom metrics are not available in .get_grid so we must use our own function to select the
            # best model
            best_model = h2o.get_model(sort_models(gbm_grid)[0][1])
        else:
            base_model = H2OGradientBoostingEstimator(nfolds=3)
            gbm_grid = grid_train(base_model, self.search_time)
            best_model = gbm_grid.get_grid(sort_by='auc', decreasing=True).models[0]

        return best_model


    def train_automl(self, train: h2o.H2OFrame, x: List[str], y: str, weight: str) -> H2OGenericEstimator:
        """ Use AutoML to build model

        Args:
            train (h2o dataframe): training data containing columns x, y, and weight
            x (list of str): column names of model features
            y (list of str): column name of ground truth
            weight (str): column name of row weights

        Return
            H2OGenericEstimator: best model out of the training grid

        """
        aml = H2OAutoML(max_runtime_secs=self.search_time, seed=1)
        aml.train(x=x, y=y, training_frame=train, weights_column=weight)
        best_model = aml.leader

        return best_model

    def df_to_hf(self, df: pd.DataFrame, cols: List[str], cat_cols: List[str]) -> Tuple[h2o.H2OFrame, pd.DataFrame]:
        """ Converts a pandas dataframe into a h2o dataframe. Part of the conversion includes dropping null
            values because they create errors in h2o training and declaring any categorical columns (asfactor)
        Args:
            df (pandas dataframe): DataFrame to be converted
            cols (list of str): column names to be extracted from df
            cat_cols (list of str): column names that are categorical
        Returns:
            hf: H2O dataframe
            cleaned_df: A pandas dataframe containing the same rows, columns, and indexes as hf
        """

        cleaned_df = df[cols].dropna()
        hf = h2o.H2OFrame(cleaned_df)
        for col in cat_cols:
            hf[col] = hf[col].asfactor()
        return hf, cleaned_df

    def calibrate_thresholds(self, df: pd.DataFrame,
                             model_type: str = 'H2OAutoML',
                             cost_matrix_loss_metric: bool = False) -> Tuple[H2OGenericEstimator, float]:
        """ Calculates the best thresholds for each corridor via the fit_function

            Args: df (DataFrame): data containing information necessary for calibration
                  model_type (str): the type of model or strategy that should be used
                  cost_matrix_loss_metric (bool): if true, training and model selection will be performed with
                                                    the CostMatrixLossMetric class

            Returns H2OModel, float: trained model and it corresponding threshold
        """

        train, _ = self.df_to_hf(df, ['corridor', 'risk_score', 'fraud', 'weight'], ['corridor', 'fraud'])
        if model_type == 'GradientBoosting':
            logging.info(f"Training Gradient Boosting {'with' if cost_matrix_loss_metric else 'without'} " +
                         "cost_matrix_loss_metric")
            model = self.train_gradientboosting(train,
                                                ['corridor', 'risk_score'],
                                                'fraud',
                                                'weight',
                                                cost_matrix_loss_metric)
        else:
            logging.info("Training AutoML")
            model = self.train_automl(train,
                                      ['corridor', 'risk_score'],
                                      'fraud',
                                      'weight')

        threshold = self.optimum_threshold(train, model)
        print(model.confusion_matrix(thresholds=threshold))
        return model, threshold

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
            model, threshold = self.calibrate_thresholds(df, model_type=self.model_type,
                                                         cost_matrix_loss_metric=self.cost_matrix_loss_metric)

            # Take the data one step in the future of the date and evaluate how the model would have done
            today = data[(data['when_created'] >= date)
                         & (data['when_created'] < date + datetime.timedelta(days=parm['step']))].copy()
            hf_today, df_w_drops = self.df_to_hf(today, ['corridor', 'risk_score', 'fraud'], ['corridor'])
            today.loc[df_w_drops.index, 'prediction'] = model.predict(test_data=hf_today).as_data_frame()['p1']
            today['prediction'] = predict(today, threshold, 1, 'prediction')
            today = reconcile(today, 'prediction', 'fraud', 'real_result')

            # All data is of the same importance when testing
            today['weight'] = 1

            # Calculate the monetary impact of the model verse the the baseline
            r, today = cal_impact(today, 'today_result', 'real_result', self.costs)
            data.loc[today.index, f"real_result_{parm['lookback']}_{parm['step']}"] = today['real_result']
            data.loc[today.index, f"real_result_cost_{parm['lookback']}_{parm['step']}"] = today['real_result_cost']
            data.loc[today.index, f"today_result_cost_{parm['lookback']}_{parm['step']}"] = today['today_result_cost']
            data.loc[today.index, 'grp'] = grp

            # Wipe the cloud with a cluster restart
            # (the models, grids, and functions will no longer be available)
            # h2o.cluster().shutdown()
            # time.sleep(5)
            # h2o.init(ip='34.236.237.138', username='h2o', password='i-06b4d70af2a7c74e3', port=54321)
            # h2o.removeAll()
            # Create frame of objects
            h_objects = h2o.ls()
            for key in h_objects['key']:
                h2o.remove(key)
            # append information gained in this iteration
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
