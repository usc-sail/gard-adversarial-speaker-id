import torch
import torch.nn.functional as F


def stable_norm(x, order=2):
    # return tf.norm(tf.contrib.layers.flatten(x), ord=ord, axis=1, keepdims=True)
    x = x.view(x.size(0), -1)
    # return F.normalize(x, p=ord, dim=1)
    return torch.norm(x, p=order, dim=1)

class ALR(object):
    """
    Adversarial Lipschitz Regularization

    Paper: https://arxiv.org/abs/1907.05681
    Code reference: [the original author's repo](https://github.com/dterjek/adversarial_lipschitz_regularization/blob/master/wgan_alp.py)
    
    The default configs expect a scalar Y and a
    """
    def __init__(
        self, 
        d_X=lambda x, x_hat: stable_norm(x - x_hat),
        d_Y=lambda y, y_hat: torch.abs(y - y_hat),
        eps_min=0.1, eps_max=10, xi=10, ip=1, K=1
    ):
        super(ALR, self).__init__()
        self.d_X = d_X
        self.d_Y = d_Y
        self.eps_min = eps_min
        self.eps_max = eps_max
        if eps_min == eps_max:
            self.eps = lambda x: eps_min * torch.ones(x.size(0), 1, device=x.device)
        else:
            self.eps = lambda x: torch.distributions.uniform.Uniform(
                low=eps_min, high=eps_max
            ).sample(
                sample_shape=(x.size(0), 1)
            ).cuda()
        self.xi = xi
        self.ip = ip
        self.K = K
    
    def adversarial_direction(self, f, x, label):
        batch_size = x.shape[0]
        y = f(x).detach()
        y = F.log_softmax(y, dim=-1)
        i = torch.arange(batch_size)
        y = y[i, label]
        
        dim = list(range(len(x.shape)))[1:]
        shape = [-1] + [1 for _ in dim]
        normalize = lambda vector: F.normalize(vector, p=2, dim=dim)  # ad hoc dims
        d = torch.rand_like(x) - 0.5
        for _ in range(self.ip):
            d = normalize(d)
            d.requires_grad_()
            x_hat = x + self.xi * d  #.view(x.shape)

            y_hat = f(x_hat)
            y_hat = y_hat[i, label]

            y_diff = self.d_Y(y, y_hat)
            y_diff = torch.mean(y_diff)

            grads = torch.autograd.grad(y_diff, d)
            d = grads[0].detach()

        r_adv = normalize(d) * self.eps(x).view(shape)  # ad hoc
        r_adv_mask = torch.lt(torch.norm(r_adv, p=2, dim=dim, keepdim=True), self.eps_min).float()
        r_adv = (1 - r_adv_mask) * r_adv + r_adv_mask * normalize(torch.rand_like(d) - 0.5)
        return r_adv
        
    def get_adversarial_perturbations(self, f, x, label):
        r_adv = self.adversarial_direction(f=f, x=x.detach(), label=label)
        # FIXME: the 0.1 prevents overshoot but is ad-hoc.
        x_hat = x + 0.1 * r_adv
        x_hat = x_hat.clamp(-1, 1)
        return x_hat

    def get_alp(self, x, x_hat, y, y_hat, label):
        
        i = torch.arange(0, int(y.shape[0]))
        y = y[i, label]

        y_hat = y_hat[i, label]

        y_diff = self.d_Y(y, y_hat)
        x_diff = self.d_X(x, x_hat)
        lip_ratio = y_diff / x_diff
        alp = (lip_ratio - self.K).clamp(min=0)
        # nonzeros = torch.nonzero(alp)
        # alp_count = nonzeros.size(0)
        alp_count = len(alp.nonzero())
        return (
            alp, lip_ratio, x_diff, y_diff, alp_count
        )
