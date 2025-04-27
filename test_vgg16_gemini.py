import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from PIL import Image
import os
from tqdm import tqdm # For progress bar
from collections import OrderedDict # To handle 'module.' prefix removal

# --- Configuration ---
# --- Configuration ---
VAL_DATA_DIR = "/home/wbc/vgg/dataset/val" # Your validation dataset path
CUSTOM_WEIGHTS_PATH = "/home/wbc/vgg/vgg_model/vgg16_imagenet_final.pth" # Your weights path
# BATCH_SIZE = 32  # <<--- CHANGE THIS
BATCH_SIZE = 1     # Process one image at a time to handle variable sizes
# NUM_WORKERS = 4 # <<--- CHANGE THIS (often better to use 0 with batch_size=1)
NUM_WORKERS = 4    # Or maybe 1, test what works best. 0 is safest.
TEST_SCALE_Q = 256 # The smallest side dimension for resizing during testing

# Function to convert VGG classifier to fully convolutional (same as before)
def convert_vgg16_to_fully_conv(vgg_model: models.VGG):
    # ... (keep the exact same function definition as before) ...
    if not isinstance(vgg_model, models.VGG):
        raise TypeError("Input model must be an instance of torchvision.models.VGG")
    features = vgg_model.features
    fully_conv_classifier = nn.Sequential()
    in_channels = 512
    in_height, in_width = 7, 7 # Expected feature map size before classifier

    for layer in vgg_model.classifier:
        if isinstance(layer, nn.Linear):
            if fully_conv_classifier.__len__() == 0:
                kernel_size = (in_height, in_width)
                padding = 0
                out_channels = layer.out_features
                conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
                conv_layer.weight.data.copy_(layer.weight.data.view(out_channels, in_channels, *kernel_size))
                conv_layer.bias.data.copy_(layer.bias.data)
                in_channels = out_channels
            else:
                kernel_size = 1
                padding = 0
                out_channels = layer.out_features
                conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
                conv_layer.weight.data.copy_(layer.weight.data.view(out_channels, in_channels, 1, 1))
                conv_layer.bias.data.copy_(layer.bias.data)
                in_channels = out_channels
            fully_conv_classifier.add_module(str(len(fully_conv_classifier)), conv_layer)
        elif isinstance(layer, (nn.ReLU, nn.Dropout)):
            fully_conv_classifier.add_module(str(len(fully_conv_classifier)), layer)
        else:
             print(f"Warning: Skipping unexpected layer type in classifier: {type(layer)}")

    fully_conv_vgg = nn.Sequential(
        features,
        fully_conv_classifier
    )
    return fully_conv_vgg


# --- Main Execution ---
if __name__ == "__main__":
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.isdir(VAL_DATA_DIR):
        print(f"Error: Validation directory not found at {VAL_DATA_DIR}")
        exit()

    if not os.path.isfile(CUSTOM_WEIGHTS_PATH):
        print(f"Error: Custom weights file not found at {CUSTOM_WEIGHTS_PATH}")
        exit()

    # --- Load Model Structure and Custom Weights ---
    print("Creating VGG16 model structure...")
    # Create the model architecture WITHOUT loading default weights
    original_vgg16 = models.vgg16(weights=None)

    print(f"Loading custom weights from: {CUSTOM_WEIGHTS_PATH}")
    # Load checkpoint to CPU first to avoid potential GPU memory issues
    checkpoint = torch.load(CUSTOM_WEIGHTS_PATH, map_location='cpu')

    # Extract the state dictionary - check for common keys
    state_dict_to_load = None
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict_to_load = checkpoint['state_dict']
            print("Loaded 'state_dict' from checkpoint.")
        elif 'model_state_dict' in checkpoint:
            state_dict_to_load = checkpoint['model_state_dict']
            print("Loaded 'model_state_dict' from checkpoint.")
        elif 'model' in checkpoint: # Less common, but possible
            state_dict_to_load = checkpoint['model']
            print("Loaded 'model' from checkpoint.")
        else:
            # If no standard keys found, assume the dictionary *is* the state_dict
            print("Warning: Could not find standard state_dict keys ('state_dict', 'model_state_dict', 'model'). Assuming the loaded dictionary is the state_dict itself.")
            state_dict_to_load = checkpoint
    else:
         # If it's not a dict, assume the loaded object is the state_dict directly
         state_dict_to_load = checkpoint
         print("Loaded checkpoint directly as state_dict.")

    # --- Handle 'module.' prefix (often from nn.DataParallel/nn.DistributedDataParallel) ---
    # Check if keys start with 'module.'
    has_module_prefix = any(key.startswith('module.') for key in state_dict_to_load.keys())

    if has_module_prefix:
        print("Detected 'module.' prefix in saved weights. Removing prefix...")
        # Create a new state dict without the prefix
        new_state_dict = OrderedDict()
        for k, v in state_dict_to_load.items():
            name = k[7:] # remove 'module.'
            new_state_dict[name] = v
        state_dict_to_load = new_state_dict # Use the cleaned state dict
    # ----------------------------------------------------------------------------

    # Load the weights into the model
    try:
        original_vgg16.load_state_dict(state_dict_to_load)
        print("Custom weights loaded successfully into the model.")
    except RuntimeError as e:
        print("\nError loading state_dict:")
        print(e)
        print("\nThis might be due to:")
        print("1. Mismatched keys between the checkpoint and the model architecture.")
        print("2. The checkpoint being for a different VGG variant or having extra/missing layers.")
        print("3. Not correctly handling the 'module.' prefix (though the script tries).")
        exit()

    original_vgg16.eval() # Set to evaluation mode AFTER loading weights

    # --- Convert Model ---
    print("Converting VGG16 with custom weights to fully convolutional...")
    fcn_vgg16 = convert_vgg16_to_fully_conv(original_vgg16)
    fcn_vgg16.to(device) # Move the final converted model to the device
    fcn_vgg16.eval()
    print("Model converted and moved to device.")

    # --- Prepare Validation Dataset and DataLoader ---
    preprocess_val = transforms.Compose([
        transforms.Resize(TEST_SCALE_Q),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print(f"Loading validation dataset from: {VAL_DATA_DIR}")
    val_dataset = datasets.ImageFolder(VAL_DATA_DIR, transform=preprocess_val)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True if device.type == 'cuda' else False)
    print(f"Found {len(val_dataset)} validation images belonging to {len(val_dataset.classes)} classes.")

    # --- Evaluation Loop ---
    running_corrects_top1 = 0
    running_corrects_top5 = 0
    total_samples = 0

    print("Starting evaluation...")
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Evaluating Batches"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs_map = fcn_vgg16(inputs)
            scores = torch.mean(outputs_map, dim=[2, 3])
            _, preds_top5 = torch.topk(scores, 5, dim=1)
            preds_top1 = preds_top5[:, 0]
            total_samples += inputs.size(0)
            running_corrects_top1 += torch.sum(preds_top1 == labels.data)
            running_corrects_top5 += torch.sum(preds_top5 == labels.data.unsqueeze(1))


    # --- Calculate Final Accuracy AND Error Rates --- # MODIFIED SECTION
    if total_samples > 0:
        # Calculate accuracies first
        top1_accuracy = (running_corrects_top1.double() / total_samples) * 100
        top5_accuracy = (running_corrects_top5.double() / total_samples) * 100

        # Calculate error rates from accuracies
        top1_error_rate = 100.0 - top1_accuracy
        top5_error_rate = 100.0 - top5_accuracy
    else:
        # Handle case with no samples
        top1_accuracy = 0.0
        top5_accuracy = 0.0
        top1_error_rate = 100.0
        top5_error_rate = 100.0
    # --- END OF MODIFIED SECTION --- #

    # --- Display Results --- # MODIFIED SECTION
    print("\n--- Evaluation Results (Using Custom Weights) ---")
    print(f"Weights Path: {CUSTOM_WEIGHTS_PATH}")
    print(f"Test Scale (Q, smallest side): {TEST_SCALE_Q}")
    print(f"Total Validation Samples: {total_samples}")
    # print(f"Top-1 Accuracy: {top1_accuracy:.4f}%") # Optional: Keep if you want accuracy too
    # print(f"Top-5 Accuracy: {top5_accuracy:.4f}%") # Optional: Keep if you want accuracy too
    print(f"Top-1 Error Rate: {top1_error_rate:.4f}%") # REPORT ERROR RATE
    print(f"Top-5 Error Rate: {top5_error_rate:.4f}%") # REPORT ERROR RATE
    print("-------------------------------------------------")
    # --- END OF MODIFIED SECTION --- #