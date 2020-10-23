import sys
sys.path.append("..")

from src import MaxmindIp
from helpers import function_time
#from dask.distributed import Client

import h2o
from src.maxmind_ip.plotting import evaluate, print_evaluation

@function_time
def main():
    maxmind = MaxmindIp()
    data = maxmind.train(reset_lookback=True, reset_step=False, sample_size=200000, repetitions=100)


def plot_loss_eval():
    maxmind = MaxmindIp()
    data = maxmind.load_data(sample_size=200000)
    train, _ = maxmind.df_to_hf(data, ['corridor', 'risk_score', 'fraud'], ['corridor'])
    gbm_gaussian = h2o.load_model(maxmind.config['model'])
    # Predict
    predictions = gbm_gaussian.predict(test_data=train).as_data_frame()
    predictions['predict'] = predictions.predict.apply(lambda x: 0 if x < 0 else 1)
    # Evalute and print summary
    items, less, more_or_perfect = evaluate(train.as_data_frame(), predictions)

    print_evaluation(predictions, less, more_or_perfect)



if __name__ == "__main__":
    #client = Client(memory_limit='8GB')
    main()
    #plot_loss_eval()