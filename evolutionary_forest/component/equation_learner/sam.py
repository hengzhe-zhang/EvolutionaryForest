import torch


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        """
        SAM optimizer implementation based on the paper
        "Sharpness-Aware Minimization for Efficiently Improving Generalization"

        Args:
            params: model parameters
            base_optimizer: the optimizer to wrap (e.g., Adam)
            rho: neighborhood size for computing the sharpness (default: 0.05)
            **kwargs: keyword arguments passed to the base_optimizer
        """
        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """
        Compute and apply the perturbation to the parameters
        """
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)  # perturb the parameter
                self.state[p]["e_w"] = e_w

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """
        Perform the actual parameter update using the gradients from the perturbed parameters
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or "e_w" not in self.state[p]:
                    continue
                p.sub_(self.state[p]["e_w"])  # revert to the original parameter

        self.base_optimizer.step()  # do the actual update

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        """
        Obsolete method, only for compatibility
        """
        raise NotImplementedError(
            "SAM doesn't work like standard optimizers. Please use the first_step and second_step methods."
        )

    def _grad_norm(self):
        """
        Compute the gradient norm for all parameters
        """
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack(
                [
                    p.grad.norm(p=2).to(shared_device)
                    for group in self.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ]
            ),
            p=2,
        )
        return norm
