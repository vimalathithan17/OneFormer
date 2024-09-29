# ------------------------------------------------------------------------------
# Reference: https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/data/datasets/register_custom_coco_panoptic.py
# Modified by Jitesh Jain (https://github.com/praeclarumjj3)
# ------------------------------------------------------------------------------

import json
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.file_io import PathManager

COCO_CUSTOM_CATEGORIES = [
    {'id': 1, 'color': [220, 20, 60], 'name': 'person', 'isthing': 1},
    {'id': 2, 'color': [119, 11, 32], 'name': 'bicycle', 'isthing': 1},
    {'id': 3, 'color': [0, 0, 142], 'name': 'car', 'isthing': 1},
    {'id': 4, 'color': [0, 0, 230], 'name': 'motorcycle', 'isthing': 1},
    {'id': 6, 'color': [0, 60, 100], 'name': 'bus', 'isthing': 1},
    {'id': 7, 'color': [0, 80, 100], 'name': 'train', 'isthing': 1},
    {'id': 8, 'color': [0, 0, 70], 'name': 'truck', 'isthing': 1},
    {'id': 10, 'color': [250, 170, 30], 'name': 'traffic light', 'isthing': 1},
    {'id': 13, 'color': [220, 220, 0], 'name': 'stop sign', 'isthing': 1},
    {'id': 128, 'color': [70, 70, 70], 'name': 'house', 'isthing': 0},
    {'id': 149, 'color': [128, 64, 128], 'name': 'road', 'isthing': 0},
    {'id': 184, 'color': [107, 142, 35], 'name': 'tree-merged', 'isthing': 0},
    {'id': 185, 'color': [190, 153, 153], 'name': 'fence-merged', 'isthing': 0},
    {'id': 187,
  'color': [70, 130, 180],
  'name': 'sky-other-merged',
  'isthing': 0},
    {'id': 193, 'color': [152, 251, 152], 'name': 'grass-merged', 'isthing': 0},
    {'id': 199,
  'color': [102, 102, 156],
  'name': 'wall-other-merged',
  'isthing': 0},
    {'color': [255, 180, 195], 'isthing': 0, 'id': 144, 'name': 'platform'},
    {'color': [96, 96, 96], 'isthing': 0, 'id': 191, 'name': 'pavement-merged'}
]

COCO_CUSTOM_COLORS = [k["color"] for k in COCO_CUSTOM_CATEGORIES]

MetadataCatalog.get("ade20k_sem_seg_train").set(
    stuff_colors=COCO_CUSTOM_COLORS[:],
)

MetadataCatalog.get("ade20k_sem_seg_val").set(
    stuff_colors=COCO_CUSTOM_COLORS[:],
)


def load_custom_coco_panoptic_json(json_file, image_dir, gt_dir, semseg_dir, meta):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/coco/train2017".
        gt_dir (str): path to the raw annotations. e.g., "~/coco/panoptic_train2017".
        json_file (str): path to the json file. e.g., "~/coco/annotations/panoptic_train2017.json".
    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    """

    def _convert_category_id(segment_info, meta):
        if segment_info["category_id"] in meta["thing_dataset_id_to_contiguous_id"]:
            segment_info["category_id"] = meta["thing_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
            segment_info["isthing"] = True
        else:
            segment_info["category_id"] = meta["stuff_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
            segment_info["isthing"] = False
        return segment_info

    with PathManager.open(json_file) as f:
        json_info = json.load(f)

    ret = []
    for ann in json_info["annotations"]:
        image_id = ann["image_id"]
        # TODO: currently we assume image and label has the same filename but
        # different extension, and images have extension ".jpg" for COCO. Need
        # to make image extension a user-provided argument if we extend this
        # function to support other COCO-like datasets.
        image_file = os.path.join(image_dir, os.path.splitext(ann["file_name"])[0] + ".jpg")
        label_file = os.path.join(gt_dir, ann["file_name"])
        sem_label_file = os.path.join(semseg_dir, ann["file_name"])
        segments_info = [_convert_category_id(x, meta) for x in ann["segments_info"]]
        ret.append(
            {
                "file_name": image_file,
                "image_id": image_id,
                "pan_seg_file_name": label_file,
                "sem_seg_file_name": sem_label_file,
                "segments_info": segments_info,
            }
        )
    assert len(ret), f"No images found in {image_dir}!"
    assert PathManager.isfile(ret[0]["file_name"]), ret[0]["file_name"]
    assert PathManager.isfile(ret[0]["pan_seg_file_name"]), ret[0]["pan_seg_file_name"]
    assert PathManager.isfile(ret[0]["sem_seg_file_name"]), ret[0]["sem_seg_file_name"]
    return ret


def register_custom_coco_panoptic(
    name, metadata, image_root, panoptic_root, semantic_root, panoptic_json, instances_json=None,
):
    """
    Register a "standard" version of ADE20k panoptic segmentation dataset named `name`.
    The dictionaries in this registered dataset follows detectron2's standard format.
    Hence it's called "standard".
    Args:
        name (str): the name that identifies a dataset,
            e.g. "ade20k_panoptic_train"
        metadata (dict): extra metadata associated with this dataset.
        image_root (str): directory which contains all the images
        panoptic_root (str): directory which contains panoptic annotation images in COCO format
        panoptic_json (str): path to the json panoptic annotation file in COCO format
        sem_seg_root (none): not used, to be consistent with
            `register_coco_panoptic_separated`.
        instances_json (str): path to the json instance annotation file
    """
    panoptic_name = name
    DatasetCatalog.register(
        panoptic_name,
        lambda: load_custom_coco_panoptic_json(
            panoptic_json, image_root, panoptic_root, semantic_root, metadata
        ),
    )
    MetadataCatalog.get(panoptic_name).set(
        panoptic_root=panoptic_root,
        image_root=image_root,
        panoptic_json=panoptic_json,
        json_file=instances_json,
        evaluator_type="ade20k_panoptic_seg",
        ignore_label=255,
        label_divisor=1000,
        **metadata,
    )


_PREDEFINED_SPLITS_CUSTOM_COCO_PANOPTIC = {
    "coco_city500_train": (
        "coco_city500/train2017",
        "coco_city500/panoptic_train2017",
        "coco_city500/annotations/panoptic_train2017.json",
        None,
        "coco_city500/annotations/instances_train2017.json",
    ),
    "coco_city500_val": (
        "coco_city500/val2017",
        "coco_city500/panoptic_val2017",
        "coco_city500/annotations/panoptic_val2017.json",
        None,
        "coco_city500/annotations/instances_val2017.json",
    ),
}


def get_metadata():
    meta = {}
    # The following metadata maps contiguous id from [0, #thing categories +
    # #stuff categories) to their names and colors. We have to replica of the
    # same name and color under "thing_*" and "stuff_*" because the current
    # visualization function in D2 handles thing and class classes differently
    # due to some heuristic used in Panoptic FPN. We keep the same naming to
    # enable reusing existing visualization functions.
    thing_classes = [k["name"] for k in COCO_CUSTOM_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in COCO_CUSTOM_CATEGORIES if k["isthing"] == 1]
    stuff_classes = [k["name"] for k in COCO_CUSTOM_CATEGORIES]
    stuff_colors = [k["color"] for k in COCO_CUSTOM_CATEGORIES]

    meta["thing_classes"] = thing_classes
    meta["thing_colors"] = thing_colors
    meta["stuff_classes"] = stuff_classes
    meta["stuff_colors"] = stuff_colors

    # Convert category id for training:
    #   category id: like semantic segmentation, it is the class id for each
    #   pixel. Since there are some classes not used in evaluation, the category
    #   id is not always contiguous and thus we have two set of category ids:
    #       - original category id: category id in the original dataset, mainly
    #           used for evaluation.
    #       - contiguous category id: [0, #classes), in order to train the linear
    #           softmax classifier.
    thing_dataset_id_to_contiguous_id = {}
    stuff_dataset_id_to_contiguous_id = {}

    for i, cat in enumerate(COCO_CUSTOM_CATEGORIES):
        if cat["isthing"]:
            thing_dataset_id_to_contiguous_id[cat["id"]] = i
        # else:
        #     stuff_dataset_id_to_contiguous_id[cat["id"]] = i

        # in order to use sem_seg evaluator
        stuff_dataset_id_to_contiguous_id[cat["id"]] = i

    meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id

    return meta


def register_all_custom_coco_panoptic(root):
    metadata = get_metadata()
    for (
        prefix,
        (image_root, panoptic_root, panoptic_json, semantic_root, instance_json),
    ) in _PREDEFINED_SPLITS_CUSTOM_COCO_PANOPTIC.items():
        # The "standard" version of COCO panoptic segmentation dataset,
        # e.g. used by Panoptic-DeepLab
        register_custom_coco_panoptic(
            prefix,
            metadata,
            os.path.join(root, image_root),
            os.path.join(root, panoptic_root),
            os.path.join(root, semantic_root),
            os.path.join(root, panoptic_json),
            os.path.join(root, instance_json),
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_custom_coco_panoptic(_root)
