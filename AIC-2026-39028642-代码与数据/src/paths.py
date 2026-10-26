from pathlib import Path
BASE=Path(__file__).resolve().parents[1]
CACHE=BASE/'image_cache'
def find_train_dir(stage):
    assert stage=='semi'
    return BASE/'data/official/train'
