"""Fetch the pinned official CLIP initialization for training."""
from pathlib import Path
from huggingface_hub import hf_hub_download
ROOT=Path(__file__).resolve().parent
REVISION='3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'
if __name__=='__main__':
    folder=ROOT/'resources/weights/clip-vit-base-patch32';folder.mkdir(parents=True,exist_ok=True)
    for name in ['config.json','preprocessor_config.json','pytorch_model.bin','tokenizer_config.json','special_tokens_map.json','vocab.json','merges.txt']:
        path=folder/name
        if not path.exists():hf_hub_download('openai/clip-vit-base-patch32',name,revision=REVISION,local_dir=folder)
    print('Pinned official initialization present')
