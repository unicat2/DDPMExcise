import torch
from torch import nn

# class Test(nn.Module):
#     def __init__(self):
#         super(Test, self).__init__()
#         self.linear = nn.Linear(28*28, 28*28)
#
#     def forward(self, x):
#         x = x.view(x.size(0), -1)
#         x = self.linear(x)
#         x = x.view(x.size(0), 1, 28, 28)
#         return x


class DDPM:
    def __init__(self, model, n_timesteps: int, device):
        self.net = model
        self.n_timesteps = n_timesteps
        self.device = device

        # beta
        self.betas = torch.linspace(0.0001, 0.02, n_timesteps).to(device)

        # alpha
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, 0)
        # self.alpha_bars = torch.tensor([torch.prod(self.alphas[:t+1]) for t in range(len(self.alphas))]).to(device)

    def p_forward(self, x0, t, noise):
        alpha_bar = self.alpha_bars[t].reshape(-1, 1, 1, 1)
        # noise = torch.randn_like(x0)
        if noise is None:
            noise = torch.randn_like(x0).to(self.device)
        mean = torch.sqrt(alpha_bar) * x0
        var = 1 - alpha_bar

        return mean + torch.sqrt(var) * noise

    def q_backward(self, xt_shape, net, device):

        xt = torch.randn(xt_shape).to(device)
        net = net.to(device)

        for t in range(self.n_timesteps-1, -1, -1):
            timestep = torch.tensor([t] * xt.shape[0], dtype=torch.long).to(xt.device)
            noise_pred = net(xt, timestep, return_dict=False)[0]

            mean = (xt - (1 - self.alphas[t]) / torch.sqrt(1 - self.alpha_bars[t]) *
                    noise_pred) / torch.sqrt(self.alphas[t])

            if t == 0:
                x0 = mean
            else:
                var = self.betas[t]
                eps = torch.randn_like(xt).to(xt.device)
                xt = mean + torch.sqrt(var) * eps

        return x0
