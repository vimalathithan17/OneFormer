import pickle
import sys
# Replace 'your_file.pkl' with the path to your .pkl file
file_path = sys.argv[1] 

# Open the file in 'rb' (read binary) mode and load the data
# with open(file_path, 'rb') as file:
#     data = pickle.load(file)

# # Now, you can inspect the data
# import pprint

# pprint.pprint(data)

import torch

# Load the checkpoint file (path to the .pth or .ckpt file)
checkpoint_path = file_path
checkpoint = torch.load(checkpoint_path)

# Explore the structure of the checkpoint
print(checkpoint.keys())  # This will help you find where the model weights are stored

# Assuming the backbone is stored under 'model' or similar
# The key might vary, so check the structure first
state_dict = checkpoint['model']

# Explore the state_dict to locate the ConvNeXt backbone
# For example, it might be stored under 'backbone'
print(state_dict.keys())

# Now, filter out only the ConvNeXt backbone weights
# convnext_backbone = {k: v for k, v in state_dict.items() if 'backbone' in k}

# # Save the extracted backbone weights to a new checkpoint file
# backbone_checkpoint_path = "path/to/save/convnext_backbone.pth"
# torch.save(convnext_backbone, backbone_checkpoint_path)

# print(f"ConvNeXt backbone saved to {backbone_checkpoint_path}")
