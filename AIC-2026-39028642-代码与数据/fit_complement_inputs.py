import argparse
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from model import BASE,load_bundle
def statistics(z, labels, weights, classes):
    mass = np.bincount(labels, weights=weights, minlength=classes)
    assert (mass > 0).all()
    sums = np.zeros((classes, z.shape[1]), np.float64)
    np.add.at(sums, labels, z * weights[:, None])
    centers = sums / mass[:, None]
    overall = sums.sum(0) / mass.sum()
    within = np.zeros(z.shape[1], np.float64)
    for start in range(0, len(z), 8192):
        residual = z[start:start + 8192] - centers[labels[start:start + 8192]]
        within += (residual ** 2 * weights[start:start + 8192, None]).sum(0)
    within /= mass.sum()
    shrunk = overall + (centers - overall) * (mass / (mass + 12))[:, None]
    between = ((shrunk - shrunk.mean(0)) ** 2).mean(0)
    ratio = between / (within + 0.1 * max(float(np.median(within)), 1e-08))
    relative = ratio / max(float(np.median(ratio)), 1e-08)
    prior = np.sqrt(relative).clip(0.6, 1.7).astype(np.float32)
    noise = (1 / (1 + relative)).astype(np.float32)
    assert np.isfinite(prior).all() and np.isfinite(noise).all()
    return (prior, noise, ratio)
def main():
 p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--axis',action='store_true');a=p.parse_args();torch.set_num_threads(6)
 b=load_bundle(a.model)
 with np.load(BASE/'inference_cache/train/support.npz') as d:v=torch.from_numpy(d['visual'].astype(np.float32));labels=d['labels'].astype(np.int64)
 with np.load(BASE/'training_cache/reliability.npz') as d:reliability=d['reliability'].astype(np.float64);density=d['density'].astype(np.float64)
 assert len(labels)==148695 and len(np.unique(labels))==750
 complement=b['visual_state']['base.projection_complement'].float();z=(F.normalize(v,dim=-1)@complement).numpy()
 if a.axis:
  prior,noise,ratio=statistics(z,labels,reliability*density,750);dest=BASE/'training_cache/axis_prior.npz';assert not dest.exists();np.savez(dest,axis_prior=prior,noise=noise,fisher_ratio=ratio)
 else:
  z=F.normalize(torch.from_numpy(z),dim=-1);weight=torch.from_numpy((reliability**2*density).astype(np.float32));center=F.normalize(torch.zeros(750,256).index_add_(0,torch.from_numpy(labels),z*weight[:,None]),dim=-1);dest=BASE/'training_cache/null_inputs.pt';assert not dest.exists();torch.save(dict(complement=complement,null_centers=center),dest)
if __name__=='__main__':main()
