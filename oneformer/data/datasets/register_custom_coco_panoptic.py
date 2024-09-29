import os
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_panoptic



# Define the paths
image_root = os.path.join(BASE_PATH, "train2017")
panoptic_root = os.path.join(BASE_PATH, "panoptic_train2017")
panoptic_json = os.path.join(BASE_PATH, "annotations", "panoptic_train2017.json")
instances_json = os.path.join(BASE_PATH, "annotations", "instances_train2017.json")

# Metadata for the dataset with ids, names, and colors
metadata = {
    'thing_classes': [
        {'id': 1, 'name': 'person', 'color': [220, 20, 60]},
        {'id': 2, 'name': 'bicycle', 'color': [119, 11, 32]},
        {'id': 3, 'name': 'car', 'color': [0, 0, 142]},
        {'id': 4, 'name': 'motorcycle', 'color': [0, 0, 230]},
        {'id': 6, 'name': 'bus', 'color': [0, 60, 100]},
        {'id': 7, 'name': 'train', 'color': [0, 80, 100]},
        {'id': 8, 'name': 'truck', 'color': [0, 0, 70]},
        {'id': 10, 'name': 'traffic light', 'color': [250, 170, 30]},
        {'id': 13, 'name': 'stop sign', 'color': [220, 220, 0]},
    ],
    'stuff_classes': [
        {'id': 128, 'name': 'house', 'color': [70, 70, 70]},
        {'id': 149, 'name': 'road', 'color': [128, 64, 128]},
        {'id': 184, 'name': 'tree-merged', 'color': [107, 142, 35]},
        {'id': 185, 'name': 'fence-merged', 'color': [190, 153, 153]},
        {'id': 187, 'name': 'sky-other-merged', 'color': [70, 130, 180]},
        {'id': 193, 'name': 'grass-merged', 'color': [152, 251, 152]},
        {'id': 199, 'name': 'wall-other-merged', 'color': [102, 102, 156]},
        {'id': 144, 'name': 'platform', 'color': [255, 180, 195]},
        {'id': 191, 'name': 'pavement-merged', 'color': [96, 96, 96]},
    ],
}

# Register the dataset for panoptic segmentation
dataset_name_train = "coco_city500_train"
register_coco_panoptic(
    name=dataset_name_train,
    metadata=metadata,
    image_root=image_root,
    panoptic_root=panoptic_root,
    panoptic_json=panoptic_json,
    instances_json=instances_json
)

# Register the metadata catalog
MetadataCatalog.get(dataset_name_train).set(thing_classes=[cls['name'] for cls in metadata['thing_classes']],
                                             stuff_classes=[cls['name'] for cls in metadata['stuff_classes']],
                                             thing_colors=[cls['color'] for cls in metadata['thing_classes']],
                                             stuff_colors=[cls['color'] for cls in metadata['stuff_classes']])

# Repeat for validation set
image_root_val = os.path.join(BASE_PATH, "val2017")
panoptic_root_val = os.path.join(BASE_PATH, "panoptic_val2017")
panoptic_json_val = os.path.join(BASE_PATH, "annotations", "panoptic_val2017.json")
instances_json_val = os.path.join(BASE_PATH, "annotations", "instances_val2017.json")

dataset_name_val = "coco_city500_val"
register_coco_panoptic(
    name=dataset_name_val,
    metadata=metadata,
    image_root=image_root_val,
    panoptic_root=panoptic_root_val,
    panoptic_json=panoptic_json_val,
    instances_json=instances_json_val
)

# Register the metadata catalog for validation
MetadataCatalog.get(dataset_name_val).set(thing_classes=[cls['name'] for cls in metadata['thing_classes']],
                                            stuff_classes=[cls['name'] for cls in metadata['stuff_classes']],
                                            thing_colors=[cls['color'] for cls in metadata['thing_classes']],
                                            stuff_colors=[cls['color'] for cls in metadata['stuff_classes']])
