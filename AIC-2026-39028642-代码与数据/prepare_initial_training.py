import argparse,json,shutil,subprocess,sys,hashlib
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from transformers import CLIPImageProcessor,CLIPVisionModelWithProjection
ROOT=Path(__file__).resolve().parent;VISUAL=ROOT/'training/visual'
def main():
 p=argparse.ArgumentParser();p.add_argument('--features-only',action='store_true');a=p.parse_args()
 records=json.loads((ROOT/'data/train_index.json').read_text());paths=[ROOT/'data/official/train'/r[0] for r in records];assert all(p.is_file() for p in paths)
 subprocess.run([sys.executable,str(ROOT/'fetch_initialization.py')],check=True)
 if not a.features_only:
  subprocess.run([sys.executable,str(ROOT/'build_cache.py'),'--split','train'],check=True)
  subprocess.run([sys.executable,str(ROOT/'generate_train_groups.py')],check=True)
  cache=VISUAL/'cache';cache.mkdir(exist_ok=True);shutil.copy2(ROOT/'training/content_groups.npz',cache/'duplicate_groups.npz')
  with np.load(cache/'duplicate_groups.npz') as d:assert hashlib.sha256(d['groups'].tobytes()+d['labels'].tobytes()).hexdigest()==json.loads((VISUAL/'initial_config.json').read_text())['group_index_sha256']
  (ROOT/'image_cache/semi_train_complete_512.json').write_text(json.dumps(dict(status='complete',split='train',path_order_sha256=hashlib.sha256('\n'.join(map(str,paths)).encode()).hexdigest())))
 out=VISUAL/'features/official_train_features.npz';out.parent.mkdir(parents=True,exist_ok=True)
 if out.exists():raise FileExistsError('Official TRAIN features already exist; inspect the previous generation before replacing them')
 folder=ROOT/'resources/weights/clip-vit-base-patch32';processor=CLIPImageProcessor.from_pretrained(folder,local_files_only=True);model=CLIPVisionModelWithProjection.from_pretrained(folder,local_files_only=True).cuda().eval();torch.set_num_threads(6);torch.backends.cuda.matmul.allow_tf32=False
 from PIL import Image
 class Images(torch.utils.data.Dataset):
  def __len__(self):return len(paths)
  def __getitem__(self,i):
   with Image.open(paths[i]) as im:return processor(images=im.convert('RGB'),return_tensors='pt')['pixel_values'][0]
 data=torch.utils.data.DataLoader(Images(),batch_size=256,num_workers=8,pin_memory=True);features=[]
 with torch.inference_mode():
  for step,pixels in enumerate(data):
   features.append(F.normalize(model(pixels.cuda()).image_embeds,dim=-1).half().cpu().float().numpy())
   if step%100==0:print('official TRAIN features',step*256,len(paths),flush=True)
 np.savez(out,features=np.concatenate(features),ok=np.ones(len(paths),dtype=bool),paths=np.array([r[0] for r in records]));print('Official initialization TRAIN features generated')
if __name__=='__main__':main()
