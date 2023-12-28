class FedRecAvgParameters:
    @classmethod
    def zero_like(cls, sample: dict):
        raise NotImplementedError
    
    @property
    def meta(self):
        raise NotImplementedError

    def add_(self, other, weight=1.):
        raise NotImplementedError

    def div_(self, other):
        raise NotImplementedError
    
    def numel(self):
        raise NotImplementedError
    
    def pre_transfer(self, config):
        pass

    def post_transfer(self, config):
        pass


class FedRecAvg:
    def __init__(self, sample_params: FedRecAvgParameters) -> None:
        self.aggregated_params = sample_params.zero_like(sample_params)
        self.total_weight = 0.
    
    def collect(self, params: FedRecAvgParameters, weight):
        self.aggregated_params.add_(params, weight)
        self.total_weight += weight

    def finallize(self):
        # if 'private_inter_mask' in self.aggregated_param_tree:
        #     interaction_mask = self.aggregated_param_tree['private_inter_mask']
        # else:
        #     interaction_mask = None
        self.aggregated_params.div_(self.total_weight)
        # tree_div_(self.aggregated_param_tree, interaction_mask, self.count)
        return self.aggregated_params
    
    def step_server_optim(self, server_params, delta_params, step_size=1.):
        server_params.add_(delta_params, weight=step_size)
        # optree.tree_map_(lambda x, y: tree_server_add_(x, y, None, 1.), self.server_params, delta_params)
        #     # self.server_params.server_step_(delta_params)
        # with self._timestats.timer("server_unroleAB_time"):
        #     self.fedmodel._set_state_from_splited_params([self._dummy_private_params, treedict2statedict(self.server_params)])