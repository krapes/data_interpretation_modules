import sys
sys.path.append("..")

from src import MaxmindIp
from helpers import function_time
#from dask.distributed import Client

@function_time
def main():
    maxmind = MaxmindIp()
    data = maxmind.train(reset_lookback=True, reset_step=False, sample_size=200000, repetitions=100)


if __name__ == "__main__":
    #client = Client(memory_limit='8GB')
    main()