import sys
sys.path.append("..")

from src import MaxmindIp
from helpers import function_time

@function_time
def main(client):
    maxmind = MaxmindIp()
    data = maxmind.train(
                         reset_lookback=False,
                         reset_step=False,
                         evaluate=True,
                         model_type='TopBottomThreshold',
                         sample_size=200,
                         repetitions=100)


if __name__ == "__main__":
    main(client=None)