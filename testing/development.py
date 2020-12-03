import sys
sys.path.append("..")

from src import MaxmindIp
from helpers import function_time

import h2o


@function_time
def main(client):
    maxmind = MaxmindIp()
    data = maxmind.train(
                         reset_lookback=False,
                         reset_step=False,
                         evaluate=True,
                         model_type='GradientBoosting',
                         sample_size=200000,
                         search_time=60*2,
                         repetitions=100)

    '''
    model = h2o.load_model(maxmind.config['model'])
    print("\n\nSHOWING RETURNED MODEL\n\n")
    print(model)
    h2o.cluster().shutdown()
    '''


if __name__ == "__main__":
    main(client)