"""Only official training images form semantic/complement neighbor targets."""
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from model import BASE,load_bundle
from src.training_neighbors import neighbors,label_distribution
OUT=BASE/'training_cache'
def sha(p):
 import hashlib
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    torch.set_num_threads(6);started=time.monotonic()
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);a=p.parse_args();assert not (OUT/'train_teacher.npz').exists()
    cfg={'baseline':a.model,'baseline_sha256':sha(a.model)}
    support=BASE/'inference_cache/train'
    assert sha(cfg['baseline'])==cfg['baseline_sha256']
    with np.load(support/'support.npz') as d:
        labels=torch.from_numpy(d['labels'].astype(np.int64)).cuda()
        main_features=F.normalize(torch.from_numpy(d['features'].astype(np.float32)).cuda(),dim=-1)
        visual=F.normalize(torch.from_numpy(d['visual'].astype(np.float32)).cuda(),dim=-1)
    with np.load(BASE/'training/content_groups.npz') as d:
        assert np.array_equal(d['labels'],labels.cpu().numpy());groups=d['groups'].copy()
    paths=np.array([r[0] for r in json.loads((BASE/'data/train_index.json').read_text())])
    with np.load(OUT/'reliability.npz') as d:
        weights=torch.from_numpy((d['density']*d['reliability']).astype(np.float32)).cuda()
        folds=torch.from_numpy(d['folds'].astype(np.int64)).cuda()
    _,inverse=np.unique(groups,return_inverse=True);assert np.array_equal(inverse%5,folds.cpu().numpy())
    ck=load_bundle(a.model)
    basis=ck['visual_state']['base.projection_complement'].float().cuda()
    axis=ck['visual_state']['axis_prior'].float().cuda()
    detail_features=F.normalize((visual@basis)*axis,dim=-1)
    distributions=[]
    for features in [main_features,detail_features]:
        values,index=neighbors(features,features,k=32,query_folds=folds,key_folds=folds)
        probability=label_distribution(values.cuda(),index.cuda(),labels,weights,40)
        assert torch.allclose(probability.sum(-1),torch.ones(len(labels),device='cuda'),atol=1e-5)
        distributions.append(probability)
    main_prob,detail_prob=distributions
    agreement=(main_prob.sqrt()*detail_prob.sqrt()).sum(-1).clamp(0,1)
    teacher=.75*main_prob+.25*detail_prob
    confidence=teacher.max(-1).values
    strength=(.15+.85*confidence)*(.3+.7*agreement)
    teacher=teacher.sqrt();teacher/=teacher.sum(-1,keepdim=True)
    assert len(labels)==148695 and len(labels.unique())==750 and torch.isfinite(teacher).all()
    np.savez(OUT/'train_teacher.npz',probabilities=teacher.cpu().numpy().astype(np.float16),
             weights=strength.cpu().numpy(),labels=labels.cpu().numpy(),paths=paths)
    metadata=dict(images=148695,classes=750,checkpoint_sha256=cfg['baseline_sha256'],reference_labels_read=False,
                  test_images_read=False,content_groups_excluded_by_fold=True,folds=5,k=32,beta=40,
                  semantic_fraction=.75,complement_fraction=.25,teacher_temperature=2,
                  agreement_mean=float(agreement.mean()),weight_min=float(strength.min()),weight_mean=float(strength.mean()),
                  source_sha256=sha(__file__),teacher_sha256=sha(OUT/'train_teacher.npz'),seconds=time.monotonic()-started,
                  limitation='Noisy training support produces soft targets, not recovered clean labels or unbiased validation.')
    (OUT/'train_teacher.json').write_text(json.dumps(metadata,indent=2));print(json.dumps(metadata),flush=True)


if __name__=='__main__':main()