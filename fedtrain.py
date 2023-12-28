import pickle
import hydra
from omegaconf import OmegaConf
import os
import pandas as pd
from pathlib import Path
import logging
from utils.stats import Logger, get_metric_value
from fedrec.core import FedDataModule, ClientSampler, SimpleServer, FedRecClient


os.environ['EXP_DIR'] = str(Path.cwd())

def run_server(
    cfg,
) -> pd.DataFrame:

    ############################## PREPARE DATASET ##########################
    feddm = FedDataModule(cfg)
    feddm.setup()
    num_items = feddm.num_items
    num_users = feddm.num_users

    def prepare_dataloader(participants, pid, n_workers, queue):
        '''
        Prepare function for workers in client sampler
        '''
        i = pid
        step_size = n_workers
        n_participants = len(participants)
        while True:
            client_permuted_index = i % n_participants
            client = participants[client_permuted_index]
            # print(f'Preparing client {client.cid}')
            train_loader = feddm.train_dataloader(cid=[client.cid])
            train_loader = pickle.dumps(train_loader)
            queue.put((client_permuted_index, client.cid, train_loader))
            del train_loader   # save memory
            i += step_size

    logging.info("Num users: %d" % num_users)
    logging.info("Num items: %d" % num_items)
    
    # define server side model
    logging.info("Init model")
    fedmodel = hydra.utils.instantiate(cfg.net.init, item_num=num_items)
    mylogger = Logger(cfg, fedmodel, wandb=cfg.TRAIN.wandb)
    
    fedmodel.to(cfg.TRAIN.device)

    logging.info("Init clients")
    client_sampler = ClientSampler(feddm.num_users, prepare_dataloader, n_workers=1)
    client_sampler.initialize_clients(FedRecClient, fedmodel)
    client_sampler.suffle_client_set(shuffle_seed=cfg.FED.shuffle_seed)
    client_sampler.prepare_dataloader(n_clients_per_round=cfg.FED.num_clients*10)
    try:
        logging.info("Init server")
        server = SimpleServer(cfg, fedmodel, client_sampler)

        for epoch in range(cfg.FED.agg_epochs):
            train_log = server.train_round(epoch_idx=epoch)

            log_dict = {"epoch": epoch}
            log_dict.update(train_log)
            nan_flag = False
            if (cfg.EVAL.interval > 0) and ((epoch % cfg.EVAL.interval == 0) or (epoch == cfg.FED.agg_epochs - 1)):                
                test_metrics = server.evaluate(feddm.val_dataloader(), 
                                               feddm.test_dataloader(), 
                                               train_loader=feddm.train_dataloader(for_eval=True))
                # if math.isnan(test_metrics['train/loss']):
                #     nan_flag = True
                log_dict.update(test_metrics)

            time_log = {f"time/{k}": v for k, v in server._timestats._time_dict.items()}
            log_dict.update(time_log)
            if (epoch % cfg.TRAIN.log_interval == 0) or (epoch == cfg.FED.agg_epochs - 1):
                mylogger.log(log_dict, term_out=True)
            server._timestats.reset('client_time', 'server_time')
            if nan_flag:
                break
    except KeyboardInterrupt:
        logging.info("Interrupted")
    except Exception as e:
        logging.exception(e)
    finally:
        client_sampler.close()
        hist_df = mylogger.finish(quiet=True)
    return hist_df, log_dict

@hydra.main(config_path=str(Path.cwd() / 'configs'), config_name='fedtrain.yaml', version_base="1.2")
def main(cfg):
    OmegaConf.resolve(cfg)
    logging.info(cfg)
    out_dir = Path(cfg.paths.output_dir)
    hist_df, log_dict = run_server(cfg)
    hist_df.to_csv(out_dir / "hist.csv", index=False)
    
    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=log_dict, metric_name=cfg.get("optimized_metric")
    )
    return metric_value

if __name__ == '__main__':
    main()