import torch
from tqdm import tqdm
from ddpm import DDPM


class DDIM(DDPM):
    def __init__(self, model, n_timesteps: int, device):
        super().__init__(model, n_timesteps, device)

    def q_backward(self, xt_shape, net, device, ddim_step=20, eta=0):
        xt = torch.randn(xt_shape).to(device)
        timesteps = torch.linspace(self.n_timesteps, 0, (ddim_step + 1)).long().to(device)
        net = net.to(device)
        x = xt
        for i in tqdm(range(1, ddim_step + 1)):
            timestep_cur = timesteps[i - 1] - 1
            timestep_pre = timesteps[i] - 1

            alpha_bar_cur = self.alpha_bars[timestep_cur]
            alpha_bar_prev = self.alpha_bars[timestep_pre] if timestep_pre >= 0 else 1

            timestep = torch.tensor([timestep_cur] * xt.shape[0], dtype=torch.long).to(device)

            noise_pred = net(x, timestep, return_dict=False)[0]

            var = eta * (1 - alpha_bar_prev) / (1 - alpha_bar_cur) * (1 - alpha_bar_cur / alpha_bar_prev)
            eps = torch.randn_like(x)

            x = (alpha_bar_prev / alpha_bar_cur) ** 0.5 * x \
                + ((1 - alpha_bar_prev - var) ** 0.5 -
                           (alpha_bar_prev * (1 - alpha_bar_cur) / alpha_bar_cur) ** 0.5) * noise_pred \
                + var ** 0.5 * eps

        return x
