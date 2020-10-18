import sys
import time
sys.path.append("..")

from src import MaxmindIp
from helpers import function_time
from dask.distributed import Client

@function_time
def main(client):
    start = time.time()
    maxmind = MaxmindIp()
    data = maxmind.train(client, reset_lookback=True, reset_step=False, sample_size=200, repetitions=5)


if __name__ == "__main__":
    client = Client()
    main(client)