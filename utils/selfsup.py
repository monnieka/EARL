import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from copy import deepcopy
import math

def add_self_args(parser):
    parser.add_argument('--selfsup', type=str, default='barlow', choices=['simclr', 'simsiam', 'byol', 'barlow'])

def _get_projector_prenet(net, device=None, bn=True):
    device = net.device if hasattr(net, 'device') else device if device is not None else "cpu"
    assert "resnet" in type(net).__name__.lower()

    sizes = [net.nf*8] + [256, net.nf*8]

    layers = []
    for i in range(len(sizes) - 2):
        layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False).to(device))
        if bn:
            layers.append(nn.BatchNorm1d(sizes[i + 1]).to(device))
        layers.append(nn.ReLU(inplace=True).to(device))
    layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False).to(device))
    return nn.Sequential(*layers).to(device)

def init_model(model, args, device=None):
    model.projector = _get_projector_prenet(model, device=device, bn=args.selfsup!='simclr')
    model.predictor = deepcopy(model.projector)

    return model

def get_self_func(args):
    if args.selfsup == 'simclr':
        return SimCLR
    elif args.selfsup == 'simsiam':
        return SimSiam
    elif args.selfsup == 'byol':
        return BYOL
    elif args.selfsup == 'barlow':
        return BarlowTwins
    else:
        raise NotImplementedError

def BarlowTwins(model, y1, y2, compute_logits=True):
    z1 = model.projector(model(y1,returnt="features") if compute_logits else y1)
    z2 = model.projector(model(y2,returnt="features") if compute_logits else y2)
    z_a = (z1 - z1.mean(0)) / z1.std(0)
    z_b = (z2 - z2.mean(0)) / z2.std(0)
    N, D = z_a.size(0), z_a.size(1)
    c_ = torch.mm(z_a.T, z_b) / N
    c_diff = (c_ - torch.eye(D).cuda()).pow(2)
    c_diff[~torch.eye(D, dtype=bool)] *= 2e-3
    loss = c_diff.sum()   
    return loss

def SimCLR(model, y1, y2, temp=100, eps=1e-6, compute_logits=True, filter_positive_scores=False, filter_bs_len=None, correlation_mask=None, distributed=False):
    z1 = model.projector(model(y1,returnt="features") if compute_logits else y1)
    z2 = model.projector(model(y2,returnt="features") if compute_logits else y2)
    z_a = F.normalize(z1, dim=1) # - z1.mean(0)) / z1.std(0)
    z_b = F.normalize(z2, dim=1) # - z2.mean(0)) / z2.std(0)

    if distributed:
        assert False, "Distributed not supported"

    out = torch.cat([z_a, z_b], dim=0)

    if filter_positive_scores:
        cov = F.cosine_similarity(out.unsqueeze(1), out.unsqueeze(0), dim=-1)

        # filter out the scores from the positive samples
        l_pos = torch.diag(cov, filter_bs_len)
        r_pos = torch.diag(cov, -filter_bs_len)

        positives = torch.cat([l_pos, r_pos]).view(2 * filter_bs_len, 1)
        negatives = cov[correlation_mask].view(2 * filter_bs_len, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= temp

        labels = torch.zeros(2 * filter_bs_len).to(cov.device).long()
        loss = F.cross_entropy(logits, labels, reduction='sum') / (2 * filter_bs_len)
    else:
        cov = torch.mm(out, out.t().contiguous())
        
        sim = torch.exp(cov / temp)
        neg = sim.sum(dim=1)
        
        row_sub = torch.Tensor(neg.shape).fill_(math.e**(1/temp)).cuda()
        neg = torch.clamp(neg - row_sub, min=eps)
        pos = torch.exp(torch.sum(z_a * z_b, dim=-1) / temp)
        pos = torch.cat([pos, pos], dim=0)

        loss = -torch.log(pos / (neg + eps)).mean()

    return loss

def SimSiam(model, y1, y2, compute_logits=True):
    def D(p, z):
        return -F.cosine_similarity(p, z.detach(), dim=-1).mean()
    
    z1 = model.projector(model(y1,returnt="features") if compute_logits else y1)
    z2 = model.projector(model(y2,returnt="features") if compute_logits else y2)

    p1 = model.predictor(z1)
    p2 = model.predictor(z2)

    loss = (D(p1, z2).mean() + D(p2, z1).mean()) * 0.5
    return loss

def BYOL(model, y1, y2, compute_logits=True):
    def D(p, z):
        p = F.normalize(p, dim=-1, p=2)
        z = F.normalize(z, dim=-1, p=2)
        return 2 - 2 *(p*z).sum(dim=-1)
    
    z1 = model.projector(model(y1,returnt="features") if compute_logits else y1)
    z2 = model.projector(model(y2,returnt="features") if compute_logits else y2)
    p1, p2 = model.predictor(z1), model.predictor(z2)

    loss = (D(z1, p2.detach()).mean() + D(z2, p1.detach()).mean() ) *0.5
    return loss