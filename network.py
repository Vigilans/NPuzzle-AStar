from AStar import Board
from mxnet import nd, gluon, init, gpu
from mxnet.gluon import nn, Trainer
from dataclasses import dataclass
from typing import Optional
import os

@dataclass
class Config:
    # Configuration of training
    loss       : gluon.loss.Loss
    optimizer  : str
    epochs     : int
    batch_size : int
    params_file   : str
    # Configuration of trainer
    learning_rate : float
    momentum      : Optional[float] = None
    weight_decay  : Optional[float] = None

class Network:
    def __init__(self, net: nn.Block, cfg: Config, transform, predict):
        self.net = net
        self.cfg = cfg
        self.transform = transform
        self.net_predict = predict
        if os.path.exists(cfg.params_file):
            self.net.load_parameters(cfg.params_file, ctx=gpu(0))
        else:
            print("Params file not found. Initializing...")
            self.net.initialize(init=init.Xavier(), ctx=gpu(0))
        self.loss = cfg.loss
        self.trainer = None
        self.load_trainer()

    def load_trainer(self):
        opt = self.cfg.optimizer
        opt_dict = { "learning_rate": self.cfg.learning_rate }
        if self.cfg.momentum is not None:
            opt_dict["momentum"] = self.cfg.momentum
        if self.cfg.weight_decay is not None:
            opt_dict["wd"] = self.cfg.weight_decay
        if self.trainer is None:
            self.trainer = Trainer(self.net.collect_params(), opt, opt_dict)
        else:
            self.trainer._init_optimizer(opt, opt_dict)

    def predict(self, cur: Board, goal: Board) -> int:
        return self.net_predict(self.net, cur.state, goal.state)
