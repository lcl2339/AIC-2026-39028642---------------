from pathlib import Path
import numpy as np
def scan_class_folders(root):
    classes=sorted(p for p in Path(root).iterdir() if p.is_dir())
    paths=[];labels=[]
    for i,p in enumerate(classes):
        images=sorted(q for q in p.rglob('*') if q.suffix.lower() in ['.jpg','.jpeg','.png','.bmp','.webp','.tif','.tiff'])
        paths.extend(images);labels.extend([i]*len(images))
    return paths,np.array(labels,dtype=np.int64),[p.name for p in classes]
