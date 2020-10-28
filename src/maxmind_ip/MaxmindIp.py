import pandas as pd
import datetime
import json
import os
import logging

from .TrainingModel import TrainingModel
from .utils import predict, avg_value

logger = logging.getLogger(__name__)


class MaxmindIp:
    dir_path = os.path.dirname(os.path.realpath(__file__))
    config_location = os.path.join(dir_path, 'config')

    def __init__(self):
        self._config = self._load_config()

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
              client: object = None,
              reset_lookback: bool = False,
              reset_step: bool = False,
              sample_size: bool = None,
              evaluate: bool = False) -> pd.DataFrame:
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
                                          cutoff=cutoff)
            self.config['lookback'] = todays_update.lookback
            self.config['step'] = todays_update.step
            logger.info(f"Lookback length set to {self._config['lookback']}")
            logger.info(f"Step length set to {self._config['step']}")

        elif reset_lookback:
            data = self.load_data(sample_size=sample_size)
            step = self._config.get('step', 14)
            todays_update = TrainingModel(data,
                                          self._config['costs'],
                                          step=step,
                                          cutoff=cutoff)
            self._config['lookback'] = todays_update.lookback
            logger.info(f"Lookback length set to {self._config['lookback']}")

        elif reset_step:
            data = self.load_data(sample_size=sample_size)
            lookback = self.config.get('lookback', 90)
            todays_update = TrainingModel(data,
                                          self._config['costs'],
                                          lookback=lookback,
                                          cutoff=cutoff)
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
                                          cutoff=cutoff)

        self.config['model'] = todays_update.best_case_model
        print(todays_update.best_case_model, self.config['model'])
        self._save_config()

        if evaluate:
            todays_update.evaluate(data)

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
