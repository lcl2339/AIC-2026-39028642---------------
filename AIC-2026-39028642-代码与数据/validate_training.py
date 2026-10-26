import argparse,json,hashlib
from pathlib import Path
import torch
from model import BASE,DEFAULT_MODEL,load_bundle,load_visual
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--model',type=Path,default=DEFAULT_MODEL);p.add_argument('--compare',type=Path);a=p.parse_args();d=load_bundle(a.model);m=load_visual(a.model,device='cpu');state=m.state_dict()
 assert len(d['class_names'])==750 and all(torch.isfinite(x).all() for x in state.values() if x.is_floating_point())
 report=dict(strict_load=True,classes=750,tensors=len(state),finite=True,independent_validation=False)
 if a.compare:
  other=load_visual(a.compare,device='cpu').state_dict();assert set(other)==set(state);diff=[k for k in state if not torch.equal(state[k],other[k])];report['different_tensors']=diff;report['parameter_equal']=not diff
 with a.model.open('rb') as f:report['model_sha256']=hashlib.file_digest(f,'sha256').hexdigest()
 print(json.dumps(report,ensure_ascii=False))
