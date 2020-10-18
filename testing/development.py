import sys
sys.path.append("..")

from src import MaxmindIp
from helpers import function_time

@function_time
def main(client):
    maxmind = MaxmindIp()
    data = maxmind.train(client, reset_lookback=True, reset_step=False, sample_size=200, repetitions=100)


if __name__ == "__main__":
    main(client)