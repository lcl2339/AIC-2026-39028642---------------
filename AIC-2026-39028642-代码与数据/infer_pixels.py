import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from model import BASE,load_visual,DEFAULT_MODEL
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=1);p.add_argument('--split',choices=['train','test'],default='test');p.add_argument('--model',type=Path,default=DEFAULT_MODEL);a=p.parse_args()
    torch.set_num_threads(6);torch.backends.cuda.matmul.allow_tf32=False
    policy=json.loads((BASE/'inference_policy.json').read_text());names=json.loads((BASE/'data'/('test_names.json' if a.split=='test' else 'train_index.json')).read_text());names=names if a.split=='test' else [r[0] for r in names]
    array=np.load(BASE/'image_cache'/('semi_'+a.split+'_512.npy'),mmap_mode='r');assert len(array)==len(names)
    complete=np.load(BASE/'image_cache'/('semi_'+a.split+'_complete_512.npy'),mmap_mode='r') if a.split=='test' else None
    with a.model.open('rb') as file:actual_model_sha=hashlib.file_digest(file,'sha256').hexdigest()
    if a.split=='test':assert actual_model_sha==policy['model_sha256'],'Model checksum differs from the inference policy'
    else:assert a.model.resolve().is_relative_to(BASE.parent)
    model=load_visual(a.model);out=BASE/'inference_cache'/a.split;out.mkdir(parents=True,exist_ok=True)
    mean=torch.tensor([.48145466,.4578275,.40821073]).view(1,3,1,1);std=torch.tensor([.26862954,.26130258,.27577711]).view(1,3,1,1)
    jobs=[('view_'+str(i),v,96,False) for i,v in enumerate(policy['classification_views'])] if a.split=='test' else []
    jobs+=[('support',dict(source=0,crop=1.,flip=False,py=.5,px=.5,resolution=448),96,True),('geometry',policy['geometry_view'],96,True)] if a.split=='test' else [('support',dict(source=0,crop=1.,flip=False,py=.5,px=.5,resolution=448),96,True)]
    for number,(name,view,batch,collect) in enumerate(jobs):
        if number%a.workers!=a.worker:continue
        target=out/(name+'.npz')
        binding=dict(model_sha256=actual_model_sha,view=view,batch=batch,split=a.split,filenames_sha256=hashlib.sha256('\n'.join(names).encode()).hexdigest())
        metadata=target.with_suffix('.json')
        if target.exists():
            assert metadata.exists() and json.loads(metadata.read_text())==binding,'An existing feature cache has a different or missing binding; inspect it before rebuilding'
            continue
        probabilities=[];features=[];visual=[]
        for start in range(0,len(names),batch):
            if collect:
                x=torch.from_numpy(np.array((complete if view.get('source')==1 else array)[start:start+batch],copy=True)).cuda().permute(0,3,1,2).float()/255
                if view['resolution']!=512:x=F.interpolate(x,(view['resolution'],)*2,mode='bilinear',align_corners=False,antialias=True)
                x=(x-mean.cuda())/std.cuda()
            else:
                x=torch.from_numpy(np.array((complete if view.get('source')==1 else array)[start:start+batch])).cuda().permute(0,3,1,2).float().div_(255)
                width=int(512*view['crop']);y=int((512-width)*view['py']);z=int((512-width)*view['px']);x=x[:,:,y:y+width,z:z+width]
                x=F.interpolate(x,(view['resolution'],)*2,mode='bilinear',align_corners=False,antialias=True)
                if view['flip']:x=x.flip(3)
                x=(x-mean.cuda())/std.cuda()
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):r=model(x)
            probabilities.append(r['logits'].float().softmax(-1).cpu().numpy().astype(np.float16))
            if collect:
                features.append(r['feat'].float().cpu().numpy().astype(np.float16));visual.append(model.base.visual.float().cpu().numpy().astype(np.float16))
            if start%(batch*100)==0:print(json.dumps(dict(job=name,processed=start+len(x),total=len(names))),flush=True)
        values=dict(filenames=np.array(names),probabilities=np.concatenate(probabilities))
        if collect:values.update(features=np.concatenate(features),visual=np.concatenate(visual))
        if a.split=='train':values['labels']=np.array([r[1] for r in json.loads((BASE/'data/train_index.json').read_text())],np.int16)
        np.savez(target,**values);metadata.write_text(json.dumps(binding,indent=2)+'\n');print(json.dumps(dict(job=name,status='complete')),flush=True)
