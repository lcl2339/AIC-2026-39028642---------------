import argparse,subprocess,sys,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent;V=ROOT/'training/visual'
def run(script,*args,cwd=ROOT):subprocess.run([sys.executable,str(script),*map(str,args)],cwd=cwd,check=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('--max-steps',type=int,default=0);p.add_argument('--initial-only',action='store_true');a=p.parse_args();run(ROOT/'validate_reproduction.py','--split','train');run(ROOT/'prepare_initial_training.py')
 initial=V/'runs/initial';extra=['--max-steps',a.max_steps] if a.max_steps else [];run(V/'tools/train.py','--config',V/'initial_config.json','--out',initial,*extra,cwd=V)
 if a.initial_only:return
 import torch,numpy as np
 from model import load_bundle,load_visual
 b=load_bundle();ck=torch.load(initial/'last.pt',map_location='cpu',weights_only=False);state={k:v for k,v in b['visual_state'].items() if not k.startswith('base.')};state.update({'base.'+k:v for k,v in ck['state_dict'].items() if 'base.'+k in b['visual_state']});state['axis_delta']=torch.zeros(256);state['axis_prior']=torch.ones(256);state['axis_noise']=torch.zeros(256);b['visual_state']=state
 output=ROOT/'runs/full_training';output.mkdir(parents=True,exist_ok=True);initial_model=output/'initial_visual.pt';torch.save(b,initial_model);run(ROOT/'validate_training.py','--model',initial_model)
 if a.max_steps:return
 run(ROOT/'infer_pixels.py','--split','train','--model',initial_model);run(ROOT/'fit_train_statistics.py');run(ROOT/'fit_complement_inputs.py','--model',initial_model,'--axis');fisher=output/'fisher_visual.pt';run(ROOT/'train_fisher.py','--model',initial_model,'--output',fisher)
 # Bind subsequent statistics to the newly generated Fisher visual model.
 shutil.rmtree(ROOT/'inference_cache/train');(ROOT/'training_cache/reliability.npz').unlink();run(ROOT/'infer_pixels.py','--split','train','--model',fisher);run(ROOT/'fit_train_statistics.py');run(ROOT/'generate_train_teacher.py','--model',fisher)
 cached=output/'cached_readout.pt';run(ROOT/'train_cached_readout.py','--model',fisher,'--output',cached);run(ROOT/'fit_complement_inputs.py','--model',cached);run(ROOT/'train_null.py','--model',cached)
 b=load_bundle(cached);b['null_state']=torch.load(ROOT/'runs/null_reproduction/last.pt',map_location='cpu',weights_only=False)['state']
 weights=np.load(ROOT/'training_cache/reliability.npz');data=np.load(ROOT/'inference_cache/train/support.npz');v=torch.nn.functional.normalize(torch.from_numpy(data['visual'].astype(np.float32)),dim=-1);y=torch.from_numpy(data['labels'].astype(np.int64));w=torch.from_numpy((weights['reliability']**2*weights['density']).astype(np.float32));b['train_centers']=torch.nn.functional.normalize(torch.zeros(750,768).index_add_(0,y,v*w[:,None]),dim=-1)
 for key in ['official_score','score','score_text','official_result']:b.pop(key,None)
 result=output/'retrained_model.pt';torch.save(b,result);run(ROOT/'validate_training.py','--model',result);print('Full training model:',result)
if __name__=='__main__':main()
