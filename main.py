import pandas as pd 
import numpy as np

from simulator import Sim
from load_data import load_md_from_file
from strategy import BestPosStrategy
from get_info import get_pnl


if __name__ == "__main__":
    PATH_TO_FILE = '/Users/ilyalobanov/CMF HFT/HW3/simulator/Data2/'
    NROWS = 200000
    md = load_md_from_file(path=PATH_TO_FILE, nrows=NROWS)
    latency = pd.Timedelta(10, 'ms').delta
    md_latency = pd.Timedelta(10, 'ms').delta
    sim = Sim(md, latency, md_latency)
    delay = pd.Timedelta(0.1, 's').delta
    hold_time = pd.Timedelta(10, 's').delta
    strategy = BestPosStrategy(delay, 12, 1, 0.21, hold_time)
    trades_list, md_list, updates_list, all_orders = strategy.run(sim)
    df = get_pnl(updates_list)
    df.to_csv('pnl.csv')