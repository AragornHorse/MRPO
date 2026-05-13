import torch
import math


def ratio(logp1, logp2, clip=None):
    """
        \frac{p1}{p2}
    """
    if clip is not None:
        rate = torch.exp(torch.clamp(logp1 - logp2, math.log(clip[0]), math.log(clip[1])))
    else:
        rate = torch.exp(logp1 - logp2)
    if clip is not None:
        rate = torch.clamp(rate, clip[0], clip[1])
    return rate


def log_sump(logp1, logp2, alpha):
    """
        log(a p1 + (1-a) p2)
    """
    if alpha == 0:
        return logp2
    if alpha == 1:
        return logp1
    loga = math.log(alpha) + logp1
    logb = math.log(1 - alpha) + logp2
    return loga - torch.nn.functional.logsigmoid(loga - logb)


if __name__ == '__main__':
    a = torch.tensor([-517.3308, -443.5132, -352.4223, -492.8773, -583.0779, -522.0129,
        -525.1516, -417.8176], device='cuda:0', requires_grad=True)
    advantages = torch.tensor([-1., -2, -3, 1, 2, 3, 0, 0], device=a.device)

    # print(log_qs(
    #     a, advantages,
    #     beta=0.1, max_improve=5., max_reduce=0.9, max_improve_others=0.5, max_reduce_others=0.5, advantage_others=None,
    #     ref=True
    # ))


