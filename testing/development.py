import sys
sys.path.append("..")

from src import MaxmindIp
from helpers import function_time

import h2o


@function_time
def main():
    maxmind = MaxmindIp()
    data = maxmind.train(reset_lookback=True, reset_step=False, sample_size=None)
    model = h2o.load_model(maxmind.config['model'])
    print("\n\nSHOWING RETURNED MODEL\n\n")
    print(model)
    h2o.cluster().shutdown()

if __name__ == "__main__":
    main()