from typing import Any, Dict, Optional

import time
import numpy as np
import scipy as sp
import torch
from omegaconf import OmegaConf
from contextlib import contextmanager

import wandb
import logging
import pandas as pd


class TimeStats(object):
    def __init__(self) -> None:
        self.flag_timestem = {}
        self._time_dict = {'set_parameters': 0, 'fit': 0, 'compress': 0, 'get_parameters': 0, 'client_time': 0}
        self._pca_vars = []

    def __str__(self):
        return str(self._time_dict)
    
    def reset(self, *flat_names):
        if len(flat_names) == 0:
            self._time_dict = {'set_parameters': 0, 'fit': 0, 'compress': 0, 'get_parameters': 0, 'client_time': 0}

        else:
            for name in flat_names:
                self._time_dict[name] = 0
    
    def set_aggregation_epoch(self, aggregation_epoch):
        self._aggregation_epoch = aggregation_epoch

    @contextmanager
    def timer(self, flag_name, max_agg=False):
        self.flag_timestem[flag_name] = time.time()
        yield self.flag_timestem[flag_name]
        if max_agg:
            self._time_dict[flag_name] = max(self._time_dict.get(flag_name, 0), time.time() - self.flag_timestem[flag_name])
        else:
            self._time_dict[flag_name] = self._time_dict.get(flag_name, 0) + time.time() - self.flag_timestem[flag_name]
        del self.flag_timestem[flag_name]

    def mark_start(self, name):
        self.flag_timestem[name] = time.time()
    
    def mark_end(self, name):
        self._time_dict[name] += time.time() - self.flag_timestem[name]
        del self.flag_timestem[name]

    def stats_transfer_params(self, cid, stat_dict):
        self._pca_vars.append({"cid": cid, "aggregation_epoch": self._aggregation_epoch, **stat_dict})

class Logger():
    def __init__(self, cfg, model, wandb=True) -> None:
        self.wandb = wandb
        if wandb:
            self.run = self.init_wandb(cfg, model, project_name=cfg.EXP.project)
        self.hist = []

    def log(self, log_dict, term_out=True):
        if self.wandb:
            wandb.log(log_dict)
        self.hist.append(log_dict)
        if term_out:
            console_log_dict = {k: v for k, v in log_dict.items() if not (k.startswith('time/') or k.startswith('update_norms/')) }
            logging.info(console_log_dict)
        
    def finish(self, **kwargs):
        if self.wandb:
            self.run.finish(**kwargs)
        hist_df = pd.DataFrame(self.hist)
        
        return hist_df


    @classmethod
    def init_wandb(cls, cfg, model, project_name):
        hparams = log_hyperparameters({"cfg": cfg, "model": model, "trainer": None})
        # start a new wandb run to track this script
        run = wandb.init(
            # set the wandb project where this run will be logged
            project=project_name,
            
            # track hyperparameters and run metadata
            config=hparams,
            reinit=True
        )
        if cfg.FED.compression_kwargs.method != "none":
            run.name = f"{cfg.net.name}-{cfg.FED.compression_plot_name}-{cfg.FED.num_clients}-{cfg.FED.local_epochs}-{run.name.split('-')[-1]}"
        else:
            run.name = f"{cfg.net.name}-{cfg.FED.num_clients}-{cfg.FED.local_epochs}-{run.name.split('-')[-1]}"
        return run

# Adapted from https://github.com/ashleve/lightning-hydra-template
def log_hyperparameters(object_dict: Dict[str, Any]) -> None:
    """Controls which config parts are saved by Lightning loggers.

    Additionally saves:
        - Number of model parameters

    :param object_dict: A dictionary containing the following objects:
        - `"cfg"`: A DictConfig object containing the main config.
        - `"model"`: The Lightning model.
        - `"trainer"`: The Lightning trainer.
    """
    hparams = {}

    cfg = OmegaConf.to_container(object_dict["cfg"])
    model = object_dict["model"]

    hparams["model"] = cfg["net"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    hparams["data"] = cfg["DATA"]
    hparams["dataloader"] = cfg["DATALOADER"]
    hparams["fed"] = cfg["FED"]
    hparams["trainer"] = cfg["TRAIN"]

    # hparams["callbacks"] = cfg.get("callbacks")
    hparams["exp"] = cfg.get("EXP")
    hparams["eval"] = cfg.get("EVAL")

    hparams["task_name"] = cfg.get("task_name")
    # hparams["tags"] = cfg.get("tags")
    # hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")

    return hparams

# Adapted from https://github.com/ashleve/lightning-hydra-template
def get_metric_value(metric_dict: Dict[str, Any], metric_name: Optional[str]) -> Optional[float]:
    """Safely retrieves value of the metric logged in run_server.

    :param metric_dict: A dict containing metric values.
    :param metric_name: If provided, the name of the metric to retrieve.
    :return: If a metric name was provided, the value of the metric.
    """
    if not metric_name:
        logging.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name]
    logging.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value