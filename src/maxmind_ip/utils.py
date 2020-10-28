import numpy as np
import pandas as pd
import types


def predict(g: pd.DataFrame,
            threshold_bottom: float,
            threshold_top: float,
            target_score: str) -> np.array:
    """ This function accepts a pandas dataframe and 1 where the target_score
        column is between the top and bottom thresholds, otherwise returns 0.

        Args: g (DataFrame): pandas dataframe containing the target_score column
              threshold_bottom (float): minimum value of target_column necessary to
                                      be considered 1
              threshold_top (float): maximum value of target_column necessary to be
                                   be considered 1
              target_score (str):  column name that should be compared to the top
                                      and bottom thresholds
        Returns numpy array: Array containing 1 or 0 for each element in g
    """
    return np.where((threshold_bottom < g[target_score]) & (g[target_score] < threshold_top), 1, 0)


def avg_value(micro_key: str, macro_key_list: list, dic: str) -> float:
    """ Returns the average value of all the micro_keys
    """
    a = [dic[c][micro_key] for c in macro_key_list if c in dic.keys()]
    return sum(a) / len(a)


def score(df: pd.DataFrame, thresholds: dict, target_score: str, col: str) -> pd.DataFrame:
    """ For each corridor it collects the correct thresholds and predicts
        1 or 0. If thresholds do not exist for that corridor it takes the
        average of all the corridors. Finally, it uses the reconcile function
        to return the entries confusion matrix quadrant.

        Args: df (DataFrame):  dataframe containing target_score column and
                                corridors
            thresholds (dict): dict containing top and bottom thresholds for
                                each corridor
            target_score (str): column name which should be analised
            col (str): Column name where results should be written
    """



    p_col = f"{col}_prediction"
    df[p_col] = np.nan
    for corridor, g in df.groupby('corridor'):
        if corridor in thresholds.keys():
            df.loc[g.index, p_col] = predict(g,
                                             thresholds[corridor]['threshold_bottom'],
                                             thresholds[corridor]['threshold_top'],
                                             target_score)
        else:
            avg_bottom = avg_value('threshold_bottom', df.corridor.unique(), thresholds)
            avg_top = avg_value('threshold_top', df.corridor.unique(), thresholds)
            df.loc[g.index, p_col] = predict(g, avg_bottom, avg_top, target_score)
    return reconcile(df, p_col, 'fraud', col)


def outcome(df: pd.DataFrame, costs: dict, col: str, name: str = 'cost') -> (float, pd.DataFrame):
    """ Mulitples each confusion matrix outcome by it's cost and weight
        to return both a total "cost" and line-by-line in the dataframe

        Args: df (DataFrame): Dataframe containing column "col" with confusion
                                matrix values
              costs (dict): Dict containing (int) weights for each
                              of the 4 quadrant of the confusion matrix
              col (str): column name containing confusion matrix values
              name (str): Name of column where output should be written

        Return int: total cost
                DataFrame: The df DataFrame now containing a "cost" column
    """
    df[name] = np.nan
    df[name] = np.where(df[col] == 'fp', costs['cost_fp'] * df['weight'], df[name])
    df[name] = np.where(df[col] == 'tp', costs['cost_tp'] * df['weight'], df[name])
    df[name] = np.where(df[col] == 'tn', costs['cost_tn'] * df['weight'], df[name])
    df[name] = np.where(df[col] == 'fn', costs['cost_fn'] * df['weight'], df[name])

    return float(df[name].sum()), df


def reconcile(df: pd.DataFrame, prediction: str, truth: str, name: str) -> pd.DataFrame:
    """ Compare the prediction to ground truth values

        Args: df (DataFrame): Dataframe containing prediction and truth columns
              prediction (str): Column name of predictions
              truth (str): Column name of ground truth values
              name (str): Name of column where results should be written

        Return DataFrame: df dataframe with result written in column "name"
    """
    df[name] = np.nan
    df[name] = np.where((df[prediction] == 1) & (df[truth] == 1), 'tp', df[name])
    df[name] = np.where((df[prediction] == 1) & (df[truth] == 0), 'fp', df[name])
    df[name] = np.where((df[prediction] == 0) & (df[truth] == 0), 'tn', df[name])
    df[name] = np.where((df[prediction] == 0) & (df[truth] == 1), 'fn', df[name])
    return df


def today_result(df: pd.DataFrame, t: float) -> pd.DataFrame:
    """  Calculates what the prediction would have been using a single threshold approach.
         Then calls reconcile to determine it's confusion matrix quadrent.

         Args: df (DataFrame): Dataframe containing column 'risk_score'
               t (int): Threshold for determining if the prediction is 1 fraud
                           or 0 not fraud

        Return: DataFrame: df dataframe with result written in column 'today_result'
    """
    df['today'] = np.where((t < df['risk_score']), 1, 0)
    df = reconcile(df, 'today', 'fraud', 'today_result')
    return df


def build_weights(df: pd.DataFrame, lookback: int = 30) -> pd.DataFrame:
    """ Creates the weight each datapoint should have when being considered in the cost
        function. It uses a linear function with the most recent data having a weight of 1
        while the data at or beyond the lookback window have a weight of 0.

        Args: df (DataFrame): dataframe containing column 'time_delta'
              lookback (int): Number of time_delta that should be considered in the weighting

        Returns: DataFrame: df containing results written to 'weight' column
    """
    max_date = df.when_created.max()
    df['time_delta'] = df.when_created.apply(lambda x: abs((x - max_date).days))
    df['weight'] = np.where(df['time_delta'] < lookback, 1 - df['time_delta'] / lookback, 0)
    return df


def cal_impact(df: pd.DataFrame, r1: str, r2: str, costs: dict) -> (float, pd.DataFrame):
    """ Calculates the cost (by calling the outcome function) for both the baseline approach
        and the new approach. Then returns the difference between those two approaches (impact)

        Args: df (DataFrame): dataframe containing columns r1 and r2
              r1 (str): name of column containing baseline (today) approach
              r2 (str): name of column containing new approach
              costs (dict): dictionary containing cost values for each of the 4 quadrants
                              in the confusion matrix
        Returns:  r (int): the value difference between the two approaches (impact)
                  df (DataFrame): df with new cost columns added
    """
    today, df = outcome(df, costs, r1, f"{r1}_cost")
    new_approach, df = outcome(df, costs, r2, f"{r2}_cost")
    r = new_approach - today
    print(f"The modified system has an impact of {round(r, 2)}")
    print(f"new_approach: {new_approach}  -  today: {today}")
    return r, df


def enhance_method(klass, method_name, replacement):
    """replace a method with an enhanced version"""
    method = getattr(klass, method_name)
    def enhanced(*args, **kwds): return replacement(method, *args, **kwds)
    setattr(klass, method_name, types.MethodType(enhanced, klass))