import torch


class CPR(torch.optim.Optimizer):
    def __init__(self, optimizer: torch.optim.Optimizer, kappa_init_param: float, kappa_init_method: str = 'warm_start',
                 reg_function: str = 'l2', kappa_adapt: bool = False, kappa_update: float = 1.0, apply_lr=False):
        """
        Args:
            optimizer (torch.optim.Optimizer): The original optimizer (e.g., SGD, Adam).
            kappa_init_param (float): The initial value of kappa.
            kappa_init_method (str): The method to initialize kappa. Options: 'warm_start', 'uniform', 'dependent'
            reg_function (str): The function to regularize the parameters. Options: 'l2', 'std'
            kappa_adapt (bool): Whether to adapt kappa during training.
            kappa_update (float): The update rate of kappa (mu).

        """
        self.base_optimizer = optimizer

        self.kappa_init_param = kappa_init_param
        self.kappa_init_method = kappa_init_method
        self.reg_function = reg_function
        self.kappa_adapt = kappa_adapt
        self.kappa_update = kappa_update
        self.apply_lr = apply_lr

        assert self.kappa_init_method in ['warm_start', 'uniform', 'dependent']
        assert self.reg_function in ['l2', 'std']

        # Ensure internal optimizer's weight decay is set to 0
        for group in self.base_optimizer.param_groups:
            if 'weight_decay' in group and group['weight_decay'] != 0:
                group['weight_decay'] = 0

        # Initialize CPR states
        self.initilize_CPR_states()

    def initilize_CPR_states(self):

        self.cpr_states = []

        for group in self.base_optimizer.param_groups:
            group_state = []
            if 'weight_decay' in group and group['weight_decay'] != 0:
                group['weight_decay'] = 0

            if 'apply_decay' in group and group['apply_decay'] is True:

                for p in group['params']:
                    state = {}
                    state["lagmul"] = torch.tensor(0, dtype=torch.float, device=p.device)
                    state["step"] = torch.tensor(0, dtype=torch.int32, device=p.device)

                    if self.kappa_adapt:
                        state["adapt_flag"] = torch.tensor(False, dtype=torch.bool, device=p.device)

                    if self.kappa_init_method == 'uniform':
                        state["kappa"] = torch.tensor(self.kappa_init_param, dtype=torch.float, device=p.device)
                    elif self.kappa_init_method == 'warm_start':
                        state["kappa"] = torch.tensor(10, dtype=torch.float, device=p.device)
                    elif self.kappa_init_method == 'dependent':
                        if self.reg_function == 'std':
                            state["kappa"] = self.kappa_init_param * torch.std(p).detach()
                        elif self.reg_function == 'l2':
                            state["kappa"] = self.kappa_init_param * p.square().mean().detach()
                    group_state.append(state)
            self.cpr_states.append(group_state)

    def zero_grad(self):
        """Clears the gradients of all optimized parameters."""
        self.base_optimizer.zero_grad()

    def state_dict(self):
        """Returns the state of the optimizer as a dict."""
        state_dict = self.base_optimizer.state_dict()
        state_dict['cpr_states'] = self.cpr_states
        return state_dict

    def load_state_dict(self, state_dict):
        """Loads the optimizer state."""
        if 'cpr_states' in state_dict:
            self.cpr_states = state_dict['cpr_states']
            del state_dict['cpr_states']
        self.base_optimizer.load_state_dict(state_dict)

    def __getattr__(self, name):
        """Redirect unknown attribute requests to the original optimizer."""
        return getattr(self.base_optimizer, name)

    def step(self, closure=None):
        """Performs a single optimization step."""
        self.base_optimizer.step(closure)

        with torch.no_grad():

            # Apply constrained parameter regularization (CPR)
            for group, group_states in zip(self.base_optimizer.param_groups, self.cpr_states):

                if 'apply_decay' in group and group['apply_decay'] is True:
                    assert len(group['params']) == len(group_states)
                    for param, state in zip(group['params'], group_states):

                        lagmul = state['lagmul']
                        kappa = state['kappa']
                        step = state['step']

                        if self.reg_function == 'l2':

                            n = float(param.numel())
                            half_sum_l2norm = param.square().sum()  # reg function

                            param_specific_lagmul_rate = self.kappa_update / n
                            param_specific_kappa = kappa * n

                            constraint_value = half_sum_l2norm - param_specific_kappa
                            grad_c = 2 * param

                            lagmul.add_(param_specific_lagmul_rate * constraint_value).clip_(min=0.)
                            if self.apply_lr:
                                param.add_(-grad_c * lagmul * group['lr'])
                            else:
                                param.add_(-grad_c * lagmul)

                        elif self.reg_function == 'std':

                            n = float(param.numel())
                            std_dev = param.std()

                            constraint_value = std_dev - kappa

                            mean = param.mean()
                            norm_param = param.sub(mean)
                            grad_std_dev = norm_param.mul_(2).sub_(2 * norm_param.mean()).div_(n - 1)
                            grad_std_dev.div_(std_dev.mul_(2))
                            grad_c = grad_std_dev

                            lagmul.add_(self.kappa_update * constraint_value).clip_(min=0.)
                            if self.apply_lr:
                                param.add_(-grad_c * lagmul * group['lr'])
                            else:
                                param.add_(-grad_c * lagmul)

                        if self.kappa_adapt and not (
                                self.kappa_init_method == 'warm_start' and self.kappa_init_param >= step):
                            adapt_flag = state['adapt_flag']

                            if True == adapt_flag and lagmul == 0:
                                if self.reg_function == 'l2':
                                    new_kappa = param.square().mean()
                                    kappa.clamp_max_(new_kappa)

                                elif self.reg_function == 'std':
                                    new_kappa = param.std()
                                    kappa.clamp_max_(new_kappa)

                            if lagmul > 0 and False == adapt_flag:
                                adapt_flag.add_(True)

                        if self.kappa_init_method == 'warm_start' and self.kappa_init_param == step:
                            if self.reg_function == 'std':
                                new_kappa = param.std()
                            elif self.reg_function == 'l2':
                                new_kappa = param.square().mean()
                            kappa.clamp_max_(new_kappa)

                        state['step'] += 1
