"""Check package identity, official image placement and reproduced prediction bytes."""
import argparse,csv,hashlib,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
MODEL_SHA='f34b1432526ed4c144ec2853f4d064ca26570af450cd3ba8867cb6ebb13e9b50'
CSV_SHA='14776ac27392e34f167d70e6ce3ba7b1d977d767349f529b969c6a53b92bb8e1'
def digest(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['train','test']);p.add_argument('--prediction',type=Path);a=p.parse_args()
    assert len(list(ROOT.parent.iterdir()))==4,'Expected exactly four root items'
    weights=list(ROOT.parent.glob('*-模型文件.pt'));assert len(weights)==1 and digest(weights[0])==MODEL_SHA,'Model identity mismatch'
    with zipfile.ZipFile(next(ROOT.parent.glob('*-赛题结果文档.zip'))) as z:
        assert z.namelist()==['pred_results.csv'];reference=z.read('pred_results.csv')
    assert hashlib.sha256(reference).hexdigest()==CSV_SHA
    expected=list(csv.reader(reference.decode().splitlines()));assert len(expected)==37444 and len({r[0] for r in expected})==37444
    assert all(len(r)==2 and len(r[1])==4 and r[1].isdigit() and 0<=int(r[1])<750 for r in expected)
    if a.split:
        if a.split=='test':paths=[ROOT/'data/official/test'/r[0] for r in expected]
        else:
            index=json.loads((ROOT/'data/train_index.json').read_text());assert len(index)==148695 and len(set(r[0] for r in index))==148695
            assert {r[1] for r in index}==set(range(750)) and all(Path(r[0]).parts[0]==f'{r[1]:04d}' for r in index)
            paths=[ROOT/'data/official/train'/r[0] for r in index]
        missing=[str(x.relative_to(ROOT)) for x in paths if not x.is_file()]
        if missing:raise FileNotFoundError(f'Missing {len(missing)} images; first missing: {missing[:5]}. See data/数据集放置说明.md')
        print(a.split,'original image paths checked:',len(paths))
    if a.prediction:
        actual=list(csv.reader(a.prediction.read_text().splitlines()));assert len(actual)==37444 and [r[0] for r in actual]==[r[0] for r in expected],'Prediction shape/order mismatch'
        assert all(len(r)==2 and len(r[1])==4 and r[1].isdigit() and 0<=int(r[1])<750 for r in actual),'Bad prediction class'
        assert digest(a.prediction)==CSV_SHA,'Prediction differs from supplied official result; retain the output and check numerical environment'
        print('All 37444 predictions match the supplied official result exactly')
    print('Package identity and CSV format checked')
if __name__=='__main__':main()
