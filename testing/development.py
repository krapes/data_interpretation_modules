import sys
sys.path.append("..")

from src import MaxmindIp
maxmind = MaxmindIp()
data = maxmind.train(reset_lookback=True, reset_step=False, sample_size=2000, repetitions=700)