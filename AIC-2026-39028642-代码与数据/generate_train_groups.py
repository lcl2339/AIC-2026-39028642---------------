"""Rebuild exact-byte TRAIN groups using only official images and given labels."""
import argparse,hashlib,json,os,tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
ROOT=Path(__file__).resolve().parent
EXPECTED_ARRAY_SHA='65f3513e3e0c4028cb4da0660a55466c0f55b44e339532dbd3e37cb7cad9f252'
DESTINATION='training/content_groups.npz'
def sha_array(x):return hashlib.sha256(x.tobytes(order='C')).hexdigest()
def generate(paths):
    def key(p):
        with p.open('rb') as f:return hashlib.file_digest(f,'sha256').digest()
    mapping={};groups=[]
    with ThreadPoolExecutor(max_workers=16) as pool:
        for n,k in enumerate(pool.map(key,paths)):
            if k not in mapping:mapping[k]=len(mapping)
            groups.append(mapping[k])
            if (n+1)%10000==0:print('TRAIN content groups',n+1,len(paths),flush=True)
    return np.asarray(groups,dtype=np.int64)
def main():
    p=argparse.ArgumentParser();p.add_argument('--verify-only',action='store_true');a=p.parse_args()
    records=json.loads((ROOT/'data/train_index.json').read_text());assert len(records)==148695
    labels=np.asarray([r[1] for r in records],dtype=np.int64);destination=ROOT/DESTINATION
    if a.verify_only:
        with np.load(destination) as d:groups=d['groups'];assert np.array_equal(d['labels'],labels)
    else:
        paths=[ROOT/'data/official/train'/r[0] for r in records];missing=[p for p in paths if not p.is_file()]
        if missing:raise FileNotFoundError(f'Missing {len(missing)} TRAIN images; see data/数据集放置说明.md')
        groups=generate(paths)
    assert groups.shape==labels.shape and groups.dtype==np.int64 and sha_array(groups)==EXPECTED_ARRAY_SHA,'Training group membership differs from historical input'
    if not a.verify_only:
        destination.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=destination.parent,suffix='.npz',delete=False) as f:temporary=Path(f.name)
        try:np.savez(temporary,groups=groups,labels=labels);os.replace(temporary,destination)
        finally:temporary.unlink(missing_ok=True)
    print('TRAIN group identities and label order checked:',len(np.unique(groups)))
if __name__=='__main__':main()
