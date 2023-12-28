from multiprocessing import Process, Queue
import pickle
import random
from typing import Dict, List, Tuple
from utils.stats import TimeStats

from .models import FedRecModel

class BaseClient:
    def __init__(self, cid) -> None:
        self._cid = cid
        self._trainloader = None
    
    @property
    def cid(self):
        return self._cid
    
    def set_trainloader(self, trainloader):
        self._trainloader = trainloader
    
    @property
    def train_loader(self):
        return self._trainloader
    
    def init_private_params(self, fedmodel: FedRecModel):
        raise NotImplementedError

class FedRecClient(BaseClient):
    def __init__(
        self,
        cid,
    ) -> None:
        super().__init__(cid)
    
    def init_private_params(self, fedmodel: FedRecModel):
        fedmodel.reinit_private_params()
        self._private_params = fedmodel.get_splited_params()['private']
        
    def fit(
        self, 
        fedmodel: FedRecModel,
        server_params: dict, 
        local_epochs: int,
        config: Dict[str, str], 
        device, 
        stats_logger: TimeStats,
        **forward_kwargs
    ) -> Tuple[dict, int, Dict]:
        # Preparing train dataloader
        if self.train_loader is None:
            raise RuntimeError("Please prepare train dataloader first. CID: %d" % self._cid)
        # Set model parameters, train model, return updated model parameters
        with stats_logger.timer('set_parameters'):
            fedmodel.set_state_from_splited_params([self._private_params, server_params])

        optimizer = fedmodel.get_optimizer(config)
        with stats_logger.timer('fit'):
            metrics = fedmodel.local_train(self.train_loader, 
                                optimizer, 
                                num_epochs=local_epochs, 
                                device=device, 
                                base_lr=config.TRAIN.lr, 
                                wd=config.TRAIN.weight_decay, 
                                **forward_kwargs)
        
        with stats_logger.timer('get_parameters'):
            splitted_params = fedmodel.get_splited_params()
            self._private_params = splitted_params['private']
            sharable_param = splitted_params['public']
            sharable_param.sub_(server_params)
            sharable_param.pre_transfer(config)

        # stats_logger.stats_transfer_params(cid=self._cid, stat_dict=self._model.stat_transfered_params(update_tree))
        return sharable_param, len(self.train_loader.dataset), metrics
    



class ClientSampler:
    def __init__(self, num_users, prepare_dataloader_fn, n_workers=1) -> None:
        # self._client_set = client_set
        self.num_users = num_users
        self._round_count = 0
        self._client_count = 0
        self._n_workers = 1 # Currently only support 1 worker
        self.prepare_dataloader_fn = prepare_dataloader_fn
    
    def initialize_clients(self, client_cls: BaseClient, fedmodel: FedRecModel) -> None:
        """
        creates `Client` instance for each `client_id` in dataset
        :param cfg: configuration dict
        :return: list of `Client` objects
        """
        clients = list()
        for client_id in range(self.num_users):
            c = client_cls(client_id)
            c.init_private_params(fedmodel)
            clients.append(c)
        self._client_set = clients
    
    def suffle_client_set(self, shuffle_seed):
        random.seed(shuffle_seed)
        random.shuffle(self._client_set)
        self.sorted_client_set = sorted(self._client_set, key=lambda t: t.cid)

    def next_round(self, num_clients) -> List[BaseClient]:
        participants = self._client_set[:num_clients]
        # rotate the list by `num_clients`
        self._client_set =  self._client_set[num_clients:] + participants
        # self._client_count += num_clients
        self._round_count += 1

        total_ds_sizes = 0
        for i in range(len(participants)):
            worker_id = self._client_count % self._n_workers
            client_permuted_index, cid, train_loader = self.queue[worker_id].get()
            assert participants[i].cid == cid
            train_loader = pickle.loads(train_loader)
            participants[i].set_trainloader(train_loader)
            total_ds_sizes += len(train_loader.dataset)
            self._client_count += 1
            # yield participants[i]
        return participants, total_ds_sizes

    def prepare_dataloader(self, n_clients_per_round) -> None:
        self.processors = []
        self._n_workers = 1
        self.queue = [Queue(maxsize=n_clients_per_round) for _ in range(self._n_workers)]
        # for i in range(self._n_workers):
        for i in range(self._n_workers):
            print(f'Starting worker {i}')
            process = Process(
                target=self.prepare_dataloader_fn,
                args=(self._client_set, i, self._n_workers, self.queue[i])
            )
            self.processors.append(process)
            process.daemon = True
            process.start()
            # process.join()
        # total_ds_sizes = 0
        # for i in range(len(participants)):
        #     train_loader = queue.get()
        #     participants[i].train_loader = train_loader
        #     total_ds_sizes += len(train_loader.dataset)
    
    def close(self):
        for process in self.processors:
            process.terminate()
            process.join()