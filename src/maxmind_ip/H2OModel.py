import logging
import os
import re
import warnings
from typing import List, Tuple

import h2o
import pandas as pd
from h2o.automl import H2OAutoML
from h2o.estimators import H2OGenericEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.grid.grid_search import H2OGridSearch

from .utils import outcome, predict, reconcile

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


class H20Model:
    threshold = None
    model = None
    cost_matrix_loss_metric = None
    search_time = None
    model_type = None
    inverse_costs = None
    dir_path = os.path.dirname(os.path.realpath(__file__))

    def __init__(self,
                 ip: str = None,
                 username: str = None,
                 password: str = None,
                 port: int = None):

        if ip is not None or username is not None or password is not None or port is not None:
            if ip is None or username is None or password is None or port is None:
                raise Exception("If using a remote H2O cluster ALL of following fields must be present"
                                "  [ip, username, password, port]")
            else:
                h2o.init(ip=ip, username=username, password=password, port=port)
        else:
            h2o.init()

    def score(self, today: pd.DataFrame):
        if self.threshold is None:
            raise Exception(f"self.threshold is {self.threshold}."
                            "Run the calibrate_thresholds method before running score")

        hf_today, df_w_drops = self.df_to_hf(today, ['corridor', 'risk_score', 'fraud'], ['corridor'])
        today.loc[df_w_drops.index, 'prediction'] = self.model.predict(test_data=hf_today).as_data_frame()['p1']
        today['prediction'] = predict(today, self.threshold, 1, 'prediction')
        today = reconcile(today, 'prediction', 'fraud', 'real_result')
        return today

    def calibrate_thresholds(self, df: pd.DataFrame, **kwargs):
        if self.model_type == 'GradientBoosting':
            self.wipe_h2o_cluster()
            train, _ = self.df_to_hf(df, ['corridor', 'risk_score', 'fraud', 'weight'], ['corridor', 'fraud'])
            logging.info(f"Training Gradient Boosting {'with' if self.cost_matrix_loss_metric else 'without'} " +
                         "cost_matrix_loss_metric")
            model = self.train_gradientboosting(train,
                                                ['corridor', 'risk_score'],
                                                'fraud',
                                                'weight',
                                                self.cost_matrix_loss_metric)
            threshold = self.optimum_threshold(train, model)
            print(model.confusion_matrix(thresholds=threshold))
            self.model = model
            self.threshold = threshold

        elif self.model_type == 'AutoML':
            self.wipe_h2o_cluster()
            train, _ = self.df_to_hf(df, ['corridor', 'risk_score', 'fraud', 'weight'], ['corridor', 'fraud'])
            logging.info("Training AutoML")
            model = self.train_automl(train,
                                      ['corridor', 'risk_score'],
                                      'fraud',
                                      'weight')
            threshold = self.optimum_threshold(train, model)
            print(model.confusion_matrix(thresholds=threshold))
            self.model = model
            self.threshold = threshold

        else:
            raise Exception(f"model_type {self.model_type} not known")

    def optimum_threshold(self, hf: h2o.H2OFrame, model: H2OGenericEstimator) -> float:
        """ Selects the best threshold for this model given the cost values of this instance

        Args:
            hf (DataFrame): Data used for evaluation. Must contain ground truth column named fraud
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
        # title = f'threshold_calculation_{model.model_id}'
        # self.plot(matrix, title, f"{self.dir_path}/plots/{title}.png")

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
            cost_matrix_loss_metric (bool): indicates if a custom loss function should be used in model selection

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
                    result = [h2o.get_model(model_name).model_performance(xval=True).custom_metric_value(),
                              model_name]
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
                search_time (int): max time in seconds that h2o should spend searching for a model in the grid
            Return:
                H2OGridSearch : trained grid
            """
            gbm_hyper_parameters = {'learn_rate': [0.01, 0.1],
                                    'max_depth': [3, 5, 9],
                                    'sample_rate': [0.8, 1.0],
                                    'col_sample_rate': [0.2, 0.5, 1.0]}
            logging.info(f"Searching Hyper Parameter Space:\n {gbm_hyper_parameters}")
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

    def wipe_h2o_cluster(self):
        # Wipe the cloud with a cluster restart
        # (the models, grids, and functions will no longer be available)
        h_objects = h2o.ls()
        logging.info(h_objects)
        for key in h_objects['key']:
            try:
                h2o.remove(key)
            except:
                logging.info(f"Error while attempting to remove {key}")
        # append information gained in this iteration
