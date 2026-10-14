"""Last adjustments before the full sequential run.

Transfer learning: cats_vs_dogs cannot be prepared on Windows (tfds builds the
archive keys with backslashes), so part 1 uses cifar10 filtered to its cat and
dog classes - the alternative the workshop itself suggests - and part 2 reads
the COCO data from the course Data folder. Captioning: a safe attention grid,
a checkpoint folder outside git, and 10 epochs so the CPU run finishes.
"""
import ast
import re
import shutil
import nbformat
from pathlib import Path

CESI = Path(r"C:\my\CESI\A5\Data Science")
PROJECT = Path(r"C:\my\Claude\cesi-data-science")

TRANSFER = {
8: '''# Load a tensorflow dataset

import tensorflow_datasets as tfds

tfds.disable_progress_bar()

# Solution. The workshop loads "cats_vs_dogs", but that dataset cannot be
# prepared on Windows with tensorflow_datasets 4.9 (it builds the archive keys
# with backslashes and never finds PetImages/Cat/0.jpg). We take the
# alternative the workshop names on the next line, cifar10, and keep it a
# two-class problem by filtering its cat (3) and dog (5) classes. Everything
# downstream stays identical: label 0 = cat, 1 = dog.
raw_train, raw_val, raw_test = tfds.load(
    "cifar10",
    # Reserve 10% for validation and 10% for test
    split=["train[:40%]", "train[40%:50%]", "train[50%:60%]"],
    as_supervised=True,  # Include labels
)

CAT, DOG = 3, 5
keep = lambda image, label: tf.logical_or(tf.equal(label, CAT), tf.equal(label, DOG))
relabel = lambda image, label: (image, tf.cast(tf.equal(label, DOG), tf.int64))

train_ds = raw_train.filter(keep).map(relabel)
validation_ds = raw_val.filter(keep).map(relabel)
test_ds = raw_test.filter(keep).map(relabel)

# filter() makes the size unknown, so we count once (the images are 32x32, it is quick)
count = lambda ds: sum(1 for _ in ds)
print("Number of training samples: %d" % count(train_ds))
# Nb of validation samples
print("Number of validation samples: %d" % count(validation_ds))
# Nb of test samples
print("Number of test samples: %d" % count(test_ds))''',

28: '''# Annotation file path
# Solution: the COCO data lives in the course Data folder; the split present
# there is used (val2014 is enough, the workshop keeps 2,000 images).
from pathlib import Path
COCO_ROOT = Path(r"C:\\my\\CESI\\A5\\Data Science\\Data")
SPLIT = "train2014" if (COCO_ROOT / "train2014").is_dir() else "val2014"

annotation_folder = str(COCO_ROOT / "annotations") + "/"
annotation_file = annotation_folder + f"captions_{SPLIT}.json"

# Path to folder containing images to be annotated
PATH = str(COCO_ROOT / SPLIT) + "/"

print("annotation file ", annotation_file)
print("PATH path ", PATH)''',
}

TRANSFER_SUBS = {
32: ("image_path = PATH + 'COCO_train2014_' + '%012d.jpg' % (val['image_id'])",
     "image_path = PATH + f'COCO_{SPLIT}_' + '%012d.jpg' % (val['image_id'])"),
}

CAPTIONING_SUBS = {
24: ('checkpoint_path = "./checkpoints/train"',
     'checkpoint_path = "./checkpoints/captioning"   # kept out of git'),
30: ("EPOCHS = 20",
     "EPOCHS = 10   # 20 in the workshop; 10 keeps the CPU run under a few hours"),
32: ('''    len_result = len(result)
    for l in range(len_result):
        temp_att = np.resize(attention_plot[l], (8, 8))
        ax = fig.add_subplot(len_result//2, len_result//2, l+1)''',
     '''    len_result = len(result)
    # Fix: the original grid (len//2 x len//2) has fewer cells than words for
    # most lengths and raises; a square grid that always fits is used instead.
    side = int(np.ceil(np.sqrt(len_result)))
    for l in range(len_result):
        temp_att = np.resize(attention_plot[l], (8, 8))
        ax = fig.add_subplot(side, side, l+1)'''),
}


def patch(name, replace_cells, subs):
    path = CESI / name
    nb = nbformat.read(path, as_version=4)
    for idx, source in replace_cells.items():
        assert nb.cells[idx].cell_type == "code", (name, idx)
        ast.parse(source)
        nb.cells[idx].source = source
        nb.cells[idx].outputs, nb.cells[idx].execution_count = [], None
    for idx, (old, new) in subs.items():
        cell = nb.cells[idx]
        assert old in cell.source, f"{name} cell {idx}: expected text not found"
        cell.source = cell.source.replace(old, new)
        ast.parse(cell.source)
        cell.outputs, cell.execution_count = [], None
    nbformat.write(nb, path)
    shutil.copy2(path, PROJECT / name)
    bad = [i for i, c in enumerate(nb.cells) if c.cell_type == "code" and not _parses(c.source)]
    print(f"{name}: {len(replace_cells)} cells replaced, {len(subs)} edited | broken: {bad}")


def _parses(src):
    try:
        ast.parse(src)
        return True
    except SyntaxError:
        return False


patch("WS_Transfer_Learning_EN_relu.ipynb", TRANSFER, TRANSFER_SUBS)
patch("WS_Captioning_EN_relu.ipynb", {}, CAPTIONING_SUBS)
