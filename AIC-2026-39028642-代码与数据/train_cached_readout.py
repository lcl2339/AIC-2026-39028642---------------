import argparse,json
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from model import BASE,load_bundle,CachedCorrection
def main():
 p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=512);a=p.parse_args();assert 0<a.steps<=512 and not a.output.exists();cfg=json.loads((BASE/'training/cached_parameters.json').read_text());torch.set_num_threads(6);torch.manual_seed(cfg['seed']);torch.use_deterministic_algorithms(True);b=load_bundle(a.model);m=CachedCorrection(cfg).cuda();m.complement.copy_(b['visual_state']['base.projection_complement']);m.axis_prior.copy_(b['visual_state']['axis_prior']);m.axis_delta.copy_(b['visual_state']['axis_delta']);torch.nn.init.zeros_(m.readout_up.weight)
 with np.load(BASE/'inference_cache/train/support.npz') as d:y=torch.from_numpy(d['labels'].astype(np.int64)).cuda();q=torch.from_numpy(d['probabilities'].astype(np.float32)).cuda();v=torch.from_numpy(d['visual'].astype(np.float32)).cuda()
 with np.load(BASE/'training_cache/reliability.npz') as d:weights=torch.from_numpy(((cfg['reliability_floor']+(1-cfg['reliability_floor'])*d['reliability'])*d['density']).astype(np.float32)).cuda()
 with np.load(BASE/'training_cache/train_teacher.npz') as d:assert np.array_equal(d['labels'],y.cpu().numpy());teacher=torch.from_numpy(d['probabilities'].astype(np.float32)).cuda();strength=torch.from_numpy(d['weights'].astype(np.float32)).cuda()
 teacher/=teacher.sum(-1,keepdim=True).clamp_min(1e-9);assert len(y)==148695
 optimizer=torch.optim.AdamW(m.parameters(),lr=cfg['cached_lr'],weight_decay=cfg['cached_weight_decay']);warm=cfg['cached_warmup'];total=cfg['cached_schedule_steps'];scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda s:(s+1)/warm if s<warm else .5*(1+np.cos(np.pi*min(1,(s-warm)/(total-warm)))));m.train();history=[]
 for step in range(a.steps):
  generator=torch.Generator(device='cuda').manual_seed(cfg['seed']+step*1009);ids=torch.randint(0,len(y),(cfg['cached_batch'],),device='cuda',generator=generator)
  with torch.autocast('cuda',dtype=torch.bfloat16):
   change=m.correction(v[ids],q[ids]);logits=q[ids].clamp_min(1e-9).log()+change;true=logits.float().softmax(-1).gather(1,y[ids,None]).flatten().clamp_min(1e-7);cls=(((1-true.pow(cfg['gce_q']))/cfg['gce_q'])*weights[ids]).sum()/weights[ids].sum();kl=F.kl_div(F.log_softmax(logits.float()/2,dim=-1),teacher[ids],reduction='none').sum(-1)*4;aux=(kl*strength[ids]).sum()/strength[ids].sum();loss=cls+cfg['cached_aux_weight']*aux+cfg['cached_regularization']*change.float().square().mean()
  assert torch.isfinite(loss);optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);optimizer.step();scheduler.step();history.append(dict(step=step+1,loss=float(loss.detach())))
  if step<3 or (step+1)%100==0:print(json.dumps(history[-1]),flush=True)
 b['readout_state']={k:v.cpu() for k,v in m.state_dict().items()};a.output.parent.mkdir(parents=True,exist_ok=True);torch.save(b,a.output);torch.save(dict(config=cfg,history=history,optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict()),a.output.with_name('cached_training.pt'))
if __name__=='__main__':main()
