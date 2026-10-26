import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from model import BASE,load_bundle,load_null,load_cached
from src.readout_math import balance,affinities,floor_propagate,final_balance
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,default=BASE/'inference_cache/test');p.add_argument('--output',type=Path,required=True);p.add_argument('--graph',type=Path);p.add_argument('--skip-sha',action='store_true');a=p.parse_args()
    assert not a.output.exists();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    cfg=json.loads((BASE/'inference_policy.json').read_text())
    with (BASE.parent/'AIC-2026-39028642-模型文件.pt').open('rb') as file:assert hashlib.file_digest(file,'sha256').hexdigest()==cfg['model_sha256']
    bundle=load_bundle();model=load_null(bundle)
    names=json.loads((BASE/'data/test_names.json').read_text());v=[]
    for i in range(len(cfg['classification_views'])):
        with np.load(a.cache/('view_'+str(i)+'.npz')) as d:assert d['filenames'].tolist()==names;v.append(d['probabilities'])
    v=torch.from_numpy(np.stack(v)).float().cuda()
    with np.load(a.cache/'geometry.npz') as d:assert d['filenames'].tolist()==names;other=torch.from_numpy(d['probabilities']).float().cuda()
    temperature=cfg['posterior_temperature']
    if temperature!=1:
        v=v.clamp_min(1e-9).pow(1/temperature);v/=v.sum(-1,keepdim=True);other=other.clamp_min(1e-9).pow(1/temperature);other/=other.sum(-1,keepdim=True)
    confidence=v.max(-1).values;nw=other.max(-1).values;f=cfg['additional_probability_view_weight'];posterior=((v*confidence[...,None]).sum(0)+f*nw[:,None]*other)/(confidence.sum(0)+f*nw).clamp_min(1e-9)[:,None]
    with np.load(a.cache/'support.npz') as d:assert d['filenames'].tolist()==names;raw=torch.from_numpy(d['visual'].astype(np.float32)).cuda();features=torch.from_numpy(d['features'].astype(np.float32)).cuda()
    old=load_cached(bundle);parts=[]
    for start in range(0,len(names),96):
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):parts.append(old.correction(raw[start:start+96],posterior[start:start+96]).float())
    posterior=(posterior.clamp_min(1e-9).log()+torch.cat(parts)/temperature).softmax(-1);del old,parts
    with torch.inference_mode():change=torch.cat([model(raw[s:s+384],features[s:s+384],posterior[s:s+384]) for s in range(0,len(names),384)])
    posterior=(posterior.clamp_min(1e-9).log()+cfg['learned_correction_scale']*change/cfg['posterior_temperature']).softmax(-1);prior=(F.normalize(raw,dim=-1)@bundle['train_centers'].cuda().T*15).softmax(-1)
    seed=balance((1-cfg['prototype_fraction'])*posterior+cfg['prototype_fraction']*prior,cfg['epsilon'])
    del model,v,confidence,posterior,prior,features,raw,change
    if a.graph:
        with np.load(a.graph) as d:assert d['filenames'].tolist()==names;index=torch.from_numpy(d['index'].copy()).cuda();sim=torch.from_numpy(d['similarity'].copy()).cuda()
    else:
        with np.load(a.cache/'support.npz') as d:assert d['filenames'].tolist()==names;raw448=F.normalize(torch.from_numpy(d['visual'].astype(np.float32)).cuda(),dim=-1)
        with np.load(a.cache/'geometry.npz') as d:assert d['filenames'].tolist()==names;raw512=F.normalize(torch.from_numpy(d['visual'].astype(np.float32)).cuda(),dim=-1)
        embedding=torch.cat([raw448*.5**.5,raw512*.5**.5],-1)
        keys=embedding.half().T;index=torch.empty((len(names),32),dtype=torch.long,device='cuda');sim=torch.empty((len(names),32),device='cuda')
        for start in range(0,len(names),256):
            stop=min(start+256,len(names));cosine=embedding[start:stop].half()@keys;cosine[torch.arange(stop-start,device='cuda'),torch.arange(start,stop,device='cuda')]=-torch.inf
            value,ids=cosine.topk(32,-1);index[start:stop]=ids;sim[start:stop]=value.float()
        del embedding,keys,cosine
    index=index[:,:cfg['neighbors']];sim=sim[:,:cfg['neighbors']]
    logweight=(sim-sim.max(-1,keepdim=True).values)*cfg['sharpness'];local=cfg['local_fraction']
    if local:
        dist=(1-sim).clamp_min(1e-5);radius=dist[:,cfg['local_rank']-1].sqrt();metric=-dist/(radius[:,None]*radius[index]).clamp_min(1e-8);metric-=metric.max(-1,keepdim=True).values;logweight=(1-local)*logweight+local*metric
    weights=logweight.exp()*seed.max(-1).values[index].pow(cfg['confidence_power'])*affinities(seed,index).pow(cfg['compatibility_power'])
    mutual=(index[index]==torch.arange(len(names),device='cuda')[:,None,None]).any(-1);weights*=cfg['nonmutual_fraction']+(1-cfg['nonmutual_fraction'])*mutual.float()
    q=floor_propagate(seed,index,weights,cfg['alpha'],cfg['iterations'],cfg['minimum']);labels=final_balance(q,cfg['assignment_temperature'],cfg['balance_strength'],cfg['assignment_iterations']).argmax(-1).cpu().tolist()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x',newline='') as f:csv.writer(f).writerows((n,f'{y:04d}') for n,y in zip(names,labels))
    actual=hashlib.sha256(a.output.read_bytes()).hexdigest();print(json.dumps(dict(rows=len(names),sha256=actual)),flush=True)
    if not a.skip_sha:assert actual==cfg['prediction_sha256'],'Prediction differs from the provided result: retain the files and inspect the environment and cache.'
