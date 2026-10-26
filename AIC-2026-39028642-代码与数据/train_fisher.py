import argparse,json,os,random,time
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from model import BASE,load_bundle,load_visual
from src.data.image_cache import CachedImageDataset
class Frames(torch.utils.data.Dataset):
 def __init__(self,center,full,groups,areas,cfg):self.center,self.full,self.groups,self.areas,self.cfg=center,full,groups,areas,cfg
 def __len__(self):return len(self.groups)
 def __getitem__(self,i):
  c=self.cfg;p=min(c['full_area_max'],c['full_area_min']+c['full_area_gain']*(1-self.areas[i]));full=np.random.default_rng(c['seed']+int(self.groups[i])*1009).random()<p;row=(self.full if full else self.center)[i];row['full_frame']=int(full);return row
def main():
 p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--steps',type=int,default=400);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert 0<a.steps<=400 and not a.output.exists();assert ',' not in os.environ.get('CUDA_VISIBLE_DEVICES','0')
 cfg=json.loads((BASE/'training/fisher_parameters.json').read_text());torch.set_num_threads(6);random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed']);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
 b=load_bundle(a.model)
 with np.load(BASE/'training_cache/axis_prior.npz') as d:b['visual_state']['axis_prior']=torch.from_numpy(d['axis_prior'].copy());b['visual_state']['axis_noise']=torch.from_numpy(d['noise'].copy())
 model=load_visual(a.model).base;model.load_state_dict(b['visual_state'],strict=True);model.requires_grad_(False)
 for i in cfg['train_layers']:model.base.clip.vision_model.encoder.layers[i].requires_grad_(True)
 model.base.clip.vision_model.post_layernorm.requires_grad_(True)
 for name,param in model.base.named_parameters():
  if '.adapter_' in name or name in ['head','log_scale','null_head']:param.requires_grad_(True)
 model.axis_delta.requires_grad_(True)
 with np.load(BASE/'inference_cache/train/support.npz') as d:reference=torch.from_numpy(d['features'].astype(np.float32)).cuda();labels=d['labels'].astype(np.int64)
 with np.load(BASE/'training_cache/reliability.npz') as d:reliability=torch.from_numpy(d['reliability'].astype(np.float32)).cuda();density=torch.from_numpy(d['density'].astype(np.float32)).cuda()
 with np.load(BASE/'training/content_groups.npz') as d:groups=d['groups'];assert np.array_equal(d['labels'],labels)
 records=json.loads((BASE/'data/train_index.json').read_text());assert len(labels)==148695;areas=[]
 from PIL import Image
 for name,label in records:
  with Image.open(BASE/'data/official/train'/name) as im:w,h=im.size
  areas.append(min(w,h)/max(w,h))
 center=CachedImageDataset(BASE/'image_cache/semi_train_512.npy',np.arange(len(labels)),labels,size=448,train=True,scale=tuple(cfg['crop_scale']),wideshot_p=cfg['wideshot_probability'],wideshot_scale=(.4,.83),wideshot_feather=8.,randaug_n=1,randaug_m=cfg['randaug_m'],seed_mode='worker',base_seed=cfg['seed'])
 full=CachedImageDataset(BASE/'image_cache/semi_train_complete_512.npy',np.arange(len(labels)),labels,size=448,train=True,scale=(1.,1.),ratio=(1.,1.),jitter=.1,seed_mode='hash',base_seed=cfg['seed'])
 data=torch.utils.data.DataLoader(Frames(center,full,groups,areas,cfg),batch_size=cfg['batch'],shuffle=True,num_workers=6,pin_memory=True,drop_last=True)
 named=[(n,p) for n,p in model.named_parameters() if p.requires_grad];head=lambda n:n.endswith(('.head','.log_scale'));adapt=lambda n:any(k in n for k in cfg['adapt_parameter_patterns']);head_params=[p for n,p in named if head(n)];backbone=[p for n,p in named if '.vision_model.' in n and not adapt(n)];extra=[p for n,p in named if not head(n) and ('.vision_model.' not in n or adapt(n))];assert len(named)==len(head_params)+len(backbone)+len(extra)
 optimizer=torch.optim.AdamW([dict(params=head_params,lr=cfg['head_lr']),dict(params=backbone,lr=cfg['backbone_lr']),dict(params=extra,lr=cfg['adapt_lr'])],weight_decay=cfg['weight_decay']);total=len(data);warm=max(1,int(.03*total));scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda s:(s+1)/warm if s<warm else .5*(1+np.cos(np.pi*min(1,(s-warm)/(total-warm)))))
 model.train();history=[]
 for step,batch in enumerate(data):
  x=batch['image'].cuda(non_blocking=True);y=batch['label'].cuda();ids=batch['index'].cuda()
  with torch.autocast('cuda',dtype=torch.bfloat16):
   r=model(x);true=r['logits'].float().softmax(-1).gather(1,y[:,None]).flatten().clamp_min(1e-7);weight=(cfg['reliability_floor']+(1-cfg['reliability_floor'])*reliability[ids])*density[ids];classification=(((1-true.pow(cfg['gce_q']))/cfg['gce_q'])*weight).sum()/weight.sum();anchor=(1-F.cosine_similarity(r['feat'].float(),reference[ids],dim=-1)).mean();aux=(model.axis_delta.float().tanh().square()*model.axis_noise).mean();loss=classification+cfg['feature_anchor']*anchor+cfg['aux_weight']*aux
  assert torch.isfinite(loss);optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_([p for _,p in named],1.,error_if_nonfinite=True);optimizer.step();scheduler.step();history.append(dict(step=step+1,loss=float(loss.detach())))
  if step<3 or (step+1)%100==0:print(json.dumps(history[-1]),flush=True)
  if step+1>=a.steps:break
 b['visual_state']={k:v.cpu() for k,v in model.state_dict().items()};a.output.parent.mkdir(parents=True,exist_ok=True);torch.save(b,a.output);torch.save(dict(config=cfg,history=history,optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict()),a.output.with_name('fisher_training.pt'))
if __name__=='__main__':main()
