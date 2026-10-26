import argparse
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from model import BASE
from src.training_neighbors import neighbors,label_distribution
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--features',type=Path,default=BASE/'inference_cache/train/support.npz');p.add_argument('--output',type=Path,default=BASE/'training_cache/reliability.npz');a=p.parse_args()
    assert not a.output.exists();torch.set_num_threads(6)
    with np.load(a.features) as d:
        f=F.normalize(torch.from_numpy(d['features'].astype(np.float32)).cuda(),dim=-1);labels=torch.from_numpy(d['labels'].astype(np.int64)).cuda();confidence=torch.from_numpy(d['probabilities'][np.arange(len(labels)),d['labels']].astype(np.float32)).cuda()
    with np.load(BASE/'training/content_groups.npz') as d:assert np.array_equal(d['labels'],labels.cpu().numpy());groups=d['groups']
    _,inverse,counts=np.unique(groups,return_inverse=True,return_counts=True);folds=torch.from_numpy((inverse%5).astype(np.int64)).cuda();density=torch.from_numpy((1/counts[inverse]).astype(np.float32)).cuda()
    similarity,index=neighbors(f,f,k=32,query_folds=folds,key_folds=folds,batch=256);agreement=torch.empty(len(labels),device='cuda')
    for start in range(0,len(labels),4096):
        distribution=label_distribution(similarity[start:start+4096].cuda(),index[start:start+4096].cuda(),labels,density,40)
        agreement[start:start+4096]=distribution.gather(1,labels[start:start+4096,None]).squeeze(1)
    reliability=.1+.9*(confidence*agreement).sqrt();a.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez(a.output,reliability=reliability.cpu().numpy(),density=density.cpu().numpy(),agreement=agreement.cpu().numpy(),folds=folds.cpu().numpy())
