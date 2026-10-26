import argparse,json,os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from model import BASE,DEFAULT_MODEL,load_bundle,load_cached
from src.null_prior import OwnPrior
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=BASE/'runs/null_reproduction');p.add_argument('--model',type=Path,default=DEFAULT_MODEL);p.add_argument('--steps',type=int,default=512);a=p.parse_args();assert 0<a.steps<=512;assert not a.output.exists();torch.set_num_threads(6);torch.use_deterministic_algorithms(True)
    cfg=json.loads((BASE/'training/null_parameters.json').read_text());torch.manual_seed(cfg['seed']);bundle=load_bundle(a.model);buffers=torch.load(BASE/'training_cache/null_inputs.pt',map_location='cpu',weights_only=False);model=OwnPrior(buffers).cuda()
    with np.load(BASE/'inference_cache/train/support.npz') as d:
        v=torch.from_numpy(d['visual'].astype(np.float32)).cuda();f=torch.from_numpy(d['features'].astype(np.float32)).cuda();q=torch.from_numpy(d['probabilities'].astype(np.float32)).cuda();y=torch.from_numpy(d['labels'].astype(np.int64)).cuda()
    assert len(y)==148695 and len(y.unique())==750;q/=q.sum(-1,keepdim=True).clamp_min(1e-9)
    old=load_cached(bundle)
    with torch.inference_mode():
        for start in range(0,len(y),512):
            with torch.autocast('cuda',dtype=torch.bfloat16):change=old.correction(v[start:start+512],q[start:start+512]).float()
            q[start:start+512]=(q[start:start+512].clamp_min(1e-9).log()+change).softmax(-1)
    del old
    q=q.clamp_min(1e-9).pow(1/cfg['temperature']);q/=q.sum(-1,keepdim=True)
    with np.load(BASE/'training_cache/reliability.npz') as d:reliability=torch.from_numpy(d['reliability'].astype(np.float32)).cuda();density=torch.from_numpy(d['density'].astype(np.float32)).cuda()
    weights=(cfg['reliability_floor']+(1-cfg['reliability_floor'])*reliability)*density
    with torch.inference_mode():assert torch.equal(model(v[:16],f[:16],q[:16]),torch.zeros_like(q[:16]))
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    def multiplier(step):
        if step<cfg['warmup']:return (step+1)/cfg['warmup']
        return .5*(1+np.cos(np.pi*min(1,(step-cfg['warmup'])/(512-cfg['warmup']))))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,multiplier);history=[]
    for step in range(a.steps):
        generator=torch.Generator(device='cuda').manual_seed(cfg['seed']+step*1009);ids=torch.randint(0,len(y),(cfg['batch'],),device='cuda',generator=generator);change=model(v[ids],f[ids],q[ids]);logits=q[ids].clamp_min(1e-9).log()*cfg['temperature']+change;pred=logits.softmax(-1);true=pred.gather(1,y[ids,None]).flatten().clamp_min(1e-7)
        per=(1-true.pow(cfg['gce_q']))/cfg['gce_q'];loss=(per*weights[ids]).sum()/weights[ids].sum()+cfg['regularization']*change.square().mean();assert torch.isfinite(loss)
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();scheduler.step();history.append(dict(step=step+1,loss=float(loss.detach())))
        if step<3 or (step+1)%128==0:print(json.dumps(history[-1]),flush=True)
    a.output.mkdir(parents=True);torch.save(dict(state=model.cpu().state_dict(),parameters=cfg,history=history),a.output/'last.pt')
