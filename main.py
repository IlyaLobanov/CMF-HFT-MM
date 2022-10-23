from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import datetime
import random

import sys
import logging


@dataclass
class Order:  # Наши ордера
    order_id: int
    size: float
    price: float
    timestamp: datetime 
    side: str

@dataclass
class AnonTrade:  # трейды из маркет даты
    timestamp: datetime
    size: float
    price: str
    side: str 
    

@dataclass
class OwnTrade:  # успешные исполнения наших трейдов
    timestamp: datetime
    trade_id: int 
    order_id: int
    size: float
    price: float
    side: str


@dataclass
class OrderbookSnapshotUpdate:  #  ордербук
    timestamp: datetime
    asks: list[tuple[float, float]]  
    bids: list[tuple[float, float]]


@dataclass
class MdUpdate:  # Data of a tick
    orderbook: Optional[OrderbookSnapshotUpdate] = None
    trades: Optional[list[AnonTrade]] = None



class Strategy:
    logger = logging.getLogger(__name__ + '.Strategy')
    H = logging.StreamHandler(sys.stdout)
    H.setLevel(logging.INFO)
    H.setFormatter(
        logging.Formatter(
            fmt="SStrategy: [%(asctime)s] %(levelname)s: %(message)s",
            datefmt="%d/%m/%Y ( %H:%M:%S )"
        ))
    logger.addHandler(H)

    def __init__(self, max_position: float, t_0: int, maker_fee: int, pnl_data) -> None:
        self.max_position = max_position 
        self.t_0 = t_0
        self.total_size = 0.0
        self.pnl = 0
        self.size = 0.001
        self.maker_fee = maker_fee
        self.pnl_data = pnl_data
        #self.orders_dict_latency = {}

    def run(self, sim: "Sim"):
        while True:
            try:
                orderbook, self.pnl, orders_dict_latency = sim.tick(self.maker_fee, self.pnl)

                side_ = random.randint(0,1) # получаем рандомное направление сделки
                #контроль макспозы 
                if (self.total_size < self.max_position) and (side_ == 0): # проверяем ограничение на покупку битка
                    side = 'BID'
                    price = orderbook.bids[0]
                    sim.place_order(side, self.size, price, orderbook.timestamp)
                    self.logger.info(f'Placing BID order, price: {price}, size: {self.size}')
                elif (self.total_size > self.max_position * (-1)) and (side_ == 1): # проверяем ограничение на продажу битка
                    side = 'ASK'
                    price = orderbook.asks[0]
                    sim.place_order(side, self.size, price, orderbook.timestamp)
                    self.logger.info(f'Placing ASK order, price: {price}, size: {self.size}')

                #отменим те ордера, которые выставлены и не исполнились за t_0
                for order_id, order in orders_dict_latency.copy().items():
                    if (orderbook.timestamp - order.timestamp) >= (self.t_0)*1000000:
                        sim.cancel_order(order_id)
                        self.logger.info(f'Canceling order')
                    else:
                        break #так как ордера в словаре упорядочены по возрастанию timestamp
                self.pnl_data.append([self.pnl, orderbook.timestamp])
            except StopIteration:
                return self.pnl_data
                break
        


def load_md_from_file(path: str) -> list[MdUpdate]:
    #загружаем
    btc_lobs =  pd.read_csv(f'{path}/lobs_bit.csv')
    btc_trades = pd.read_csv(f'{path}/trades_bit.csv')

    #оставляем данные только о best bid / best ask
    btc_lobs = btc_lobs.set_index('receive_ts', drop=False)
    btc_lobs = btc_lobs.iloc[:, :6] 
    btc_lobs.columns = [i.replace('btcusdt:Binance:LinearPerpetual_', '') for i in btc_lobs.columns] 

    #мержим
    md = btc_trades.groupby(by='exchange_ts').agg({'receive_ts': 'last', 'price': 'max', 'aggro_side': ['last', 'count']})
    md.columns = ['_'.join(i) for i in md]
    df = pd.merge_asof(md, btc_lobs.iloc[:, 2:6], left_on='receive_ts_last', right_index=True)

    
    #снэпшоты ордербука и список трейдов
    orderbooks = []
    trades = []

    asks = df[['ask_price_0', 'ask_vol_0']].values
    bids = df[['bid_price_0', 'bid_vol_0']].values
    receive_ts_ = df['receive_ts_last'].values

    #снэпшоты и трейды
    for i in range(df.shape[0]):
        orderbook = OrderbookSnapshotUpdate(receive_ts_[i], asks[i], bids[i]) 
        orderbooks.append(orderbook)
        trade = AnonTrade(receive_ts_[i], 0.001, df['price_max'].values[i], df['aggro_side_last'].values[i])
        trades.append(trade)
          

    return orderbooks, trades


class Sim:
    def __init__(self, execution_latency: int, md_latency: int) -> None:
        self.orderbooks, self.trades = load_md_from_file("/Users/ilyalobanov/CMF HFT/HW 2 /Data")
        self.orderbook = iter(self.orderbooks)
        self.trade = iter(self.trades) 
        self.execution_latency = execution_latency
        self.md_latency = md_latency
        self.orders_dict = {} #ордера в очереди на выставление
        self.orders_dict_latency = {} #ордера, подождавшие в orders_list время execution_latency
        self.own_trades = {} #успешно исполненные ордера
        self.total_size = 0.0
        self.trade_id = 1
        self.order_id = 1


    def tick(self, maker_fee, pnl) -> MdUpdate:
        trade = next(self.trade)
        orderbook = next(self.orderbook)
        #добавляем в orders_list_latency только те, что подождали execution_latency
        for order_id, order in self.orders_dict.copy().items():
            if (orderbook.timestamp - order.timestamp) >= (self.execution_latency)*1000000:
                self.prepare_orders(order_id, order)
                self.orders_dict.pop(order_id)
                #self.total_size += 0.001

        pnl = self.execute_orders(trade, orderbook, maker_fee, pnl)

        return orderbook, pnl, self.orders_dict_latency

    def prepare_orders(self, order_id, order):
        self.orders_dict_latency[order_id] = order

        return self.orders_dict_latency
    def execute_orders(self, trade, orderbook, maker_fee, pnl):
        for order_id, order in self.orders_dict_latency.copy().items():

            #проверка на то, исполнился ли наш ордер (с учетом комиссии биржи)
            if (order.side == 'BID' and order.price >=orderbook.asks[0] + maker_fee) or \
                 (order.side == 'ASK' and order.price <= orderbook.bids[0] + maker_fee):
                
                if order.side == 'BID':
                    pnl -= (orderbook.asks[0] + maker_fee)*order.size
                    self.total_size += order.size # меняем total_size 
                elif order.side == 'ASK':
                    pnl += (orderbook.bids[0] + maker_fee)*order.size
                    self.total_size -= order.size #  меняем total_size

                #timestamp + md_latency, так как наш ордер исполнился не в момент timestamp, а с задержкой md_latency
                own_trade_order = OwnTrade(trade.timestamp + (self.md_latency)*1000000, self.trade_id, order.order_id, order.size, order.price, order.side)
                self.own_trades[self.trade_id] = own_trade_order
                self.orders_dict_latency.pop(order_id)
                self.trade_id += 1
            
            elif (order.side == 'BID' and trade.side == 'ASK' and order.price >= trade.price + maker_fee) or \
               (order.side == 'ASK' and trade.side == 'BID' and order.price <= trade.price + maker_fee):
                
                if order.side == 'BID':
                    pnl -= (trade.price + maker_fee)*order.size
                    self.total_size += order.size # меняем total_size 
                elif order.side == 'ASK':
                    pnl += (trade.price + maker_fee)*order.size
                    self.total_size -= order.size #  меняем total_size

                #timestamp + md_latency, так как наш ордер исполнился не в момент timestamp, а с задержкой md_latency
                own_trade_order = OwnTrade(trade.timestamp + (self.md_latency)*1000000, self.trade_id, order.order_id, order.size, order.price, order.side)
                self.own_trades[self.trade_id] = own_trade_order
                self.orders_dict_latency.pop(order_id)
                self.trade_id += 1

            

        return pnl

    def place_order(self, side, size, price, timestamp):
        order = Order(self.order_id, size, price, timestamp, side)
        self.order_id += 1

        self.orders_dict[self.order_id] = order

        return self.orders_dict

    def cancel_order(self, order_id):
        self.orders_dict_latency.pop(order_id)


if __name__ == "__main__":
    strategy = Strategy(10, 400, 0, [])
    sim = Sim(10, 10)
    pnl_data = strategy.run(sim)
    pnl_data = pd.DataFrame(pnl_data)
    pnl_data.to_csv('pnl_data.csv')

