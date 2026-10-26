import argparse,json
from pathlib import Path
from src.data.image_cache import build_cache
from model import BASE
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['train','test'],default='test');a=p.parse_args()
    names=json.loads((BASE/'data'/('test_names.json' if a.split=='test' else 'train_index.json')).read_text())
    names=names if a.split=='test' else [r[0] for r in names]
    paths=[BASE/'data/official'/a.split/n for n in names]
    missing=[str(p.relative_to(BASE)) for p in paths if not p.is_file()]
    if missing:raise FileNotFoundError(f'Missing {len(missing)} images; first: {missing[:5]}')
    build_cache(paths,BASE/'image_cache'/('semi_'+a.split+'_512.npy'),size=512,workers=24)

    failed=BASE/'image_cache'/('semi_'+a.split+'_512.failed.txt')
    if failed.exists() and failed.read_text().strip():raise RuntimeError('Image decoding failed; inspect the failure list and rebuild the cache after correcting the images')

    if a.split=='test':
        from concurrent.futures import ThreadPoolExecutor
        from PIL import Image,ImageOps,ImageFile
        import numpy as np
        ImageFile.LOAD_TRUNCATED_IMAGES=True
        destination=BASE/'image_cache/semi_test_complete_512.npy'
        if not destination.exists():
            temporary=destination.with_suffix('.building.npy')
            if temporary.exists():raise FileExistsError('Inspect the incomplete full-frame cache before restarting')
            def decode_complete(path):
                with Image.open(path) as im:return np.asarray(ImageOps.pad(im.convert('RGB'),(512,512),method=Image.Resampling.BICUBIC,color=(123,117,104)))
            array=np.lib.format.open_memmap(temporary,mode='w+',dtype=np.uint8,shape=(len(paths),512,512,3))
            with ThreadPoolExecutor(max_workers=24) as pool:
                for index,pixels in enumerate(pool.map(decode_complete,paths)):
                    array[index]=pixels
                    if (index+1)%10000==0:print('full-frame',index+1,flush=True)
            array.flush();del array;temporary.rename(destination)
