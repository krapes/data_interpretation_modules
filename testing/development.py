import sys
sys.path.append("..")

from src import MaxmindIp

from dask.distributed import Client


def main(client):
    maxmind = MaxmindIp()
    data = maxmind.train(client, reset_lookback=True, reset_step=False, sample_size=200, repetitions=700)


if __name__ == "__main__":
    client = Client()
    main(client)