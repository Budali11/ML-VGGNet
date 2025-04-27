import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
import math
import signal # For Ctrl+C handling
import sys    # For exiting cleanly

# --- Configuration ---
# !!! MODIFY THESE PATHS !!!
DATA_DIR = '/home/wbc/dataset'             # Path to ImageNet dataset (train/val folders)
VGG11_WEIGHTS_PATH = './vgg11_imagenet_best.pth'      # Path to your TRAINED VGG11 weights file
VGG16_MODEL_SAVE_PATH = './vgg_model/vgg16_imagenet_final.pth' # Path to save the FINAL BEST VGG16 model
VGG16_CHECKPOINT_PATH = './vgg_model/vgg16_imagenet_checkpoint.pth' # Path for VGG16 checkpoints (for resuming)
# !!! END MODIFY PATHS !!!

# --- Hyperparameters ---
NUM_CLASSES = 1000
# Consider reducing BATCH_SIZE for VGG16 due to larger memory footprint
BATCH_SIZE = 128 # Adjust based on your GPU memory (e.g., 128, 64)
NUM_EPOCHS = 74  # Total epochs for training VGG16
# Initial LR might need tuning when starting from pre-initialized weights
INITIAL_LR = 0.01 # Paper default, maybe try 0.005 or 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
NUM_WORKERS = 8  # Adjust based on your system
LR_DECAY_FACTOR = 0.1
LR_PATIENCE = 10 # Epochs to wait for improvement before reducing LR

# --- VGG16 Model Definition ---
class VGG16(nn.Module):
    def __init__(self, num_classes=1000, init_weights=True):
        super(VGG16, self).__init__()
        # Based on Configuration 'D' in the VGG paper (16 weight layers)
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1), # 0: Conv 1
            nn.ReLU(inplace=True),                     # 1
            nn.Conv2d(64, 64, kernel_size=3, padding=1),# 2: Conv 2 * (New in VGG16)
            nn.ReLU(inplace=True),                     # 3
            nn.MaxPool2d(kernel_size=2, stride=2),     # 4
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),# 5: Conv 3
            nn.ReLU(inplace=True),                     # 6
            nn.Conv2d(128, 128, kernel_size=3, padding=1),# 7: Conv 4 * (New in VGG16)
            nn.ReLU(inplace=True),                     # 8
            nn.MaxPool2d(kernel_size=2, stride=2),     # 9
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),# 10: Conv 5
            nn.ReLU(inplace=True),                     # 11
            nn.Conv2d(256, 256, kernel_size=3, padding=1),# 12: Conv 6
            nn.ReLU(inplace=True),                     # 13
            nn.Conv2d(256, 256, kernel_size=3, padding=1), # 14: Conv 7 * (New in VGG16)
            nn.ReLU(inplace=True),                     # 15
            nn.MaxPool2d(kernel_size=2, stride=2),     # 16
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),# 17: Conv 8
            nn.ReLU(inplace=True),                     # 18
            nn.Conv2d(512, 512, kernel_size=3, padding=1),# 19: Conv 9
            nn.ReLU(inplace=True),                     # 20
            nn.Conv2d(512, 512, kernel_size=3, padding=1), # 21: Conv 10 * (New in VGG16)
            nn.ReLU(inplace=True),                     # 22
            nn.MaxPool2d(kernel_size=2, stride=2),     # 23
            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),# 24: Conv 11
            nn.ReLU(inplace=True),                     # 25
            nn.Conv2d(512, 512, kernel_size=3, padding=1),# 26: Conv 12
            nn.ReLU(inplace=True),                     # 27
            nn.Conv2d(512, 512, kernel_size=3, padding=1), # 28: Conv 13 * (New in VGG16)
            nn.ReLU(inplace=True),                     # 29
            nn.MaxPool2d(kernel_size=2, stride=2),     # 30
        )
        # Classifier structure is the same as VGG11
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096), # 0: FC 1
            nn.ReLU(True),                # 1
            nn.Dropout(p=0.5),            # 2
            nn.Linear(4096, 4096),      # 3: FC 2
            nn.ReLU(True),                # 4
            nn.Dropout(p=0.5),            # 5
            nn.Linear(4096, num_classes), # 6: FC 3
        )
        if init_weights:
            self._initialize_weights() # Standard VGG initialization

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        # Standard initialization (intermediate layers will use this)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d): # VGG doesn't use BN by default
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

# --- Partial Weight Loading Function ---
def load_partial_weights(vgg16_model, vgg11_state_dict_path, device):
    """
    Initializes VGG16 layers based on VGG11 weights as described in the paper.
    - First 4 conv layers from VGG11 -> corresponding VGG16 conv layers.
    - Last 3 FC layers from VGG11 -> corresponding VGG16 FC layers.
    - Intermediate layers keep their random initialization.

    Args:
        vgg16_model (nn.Module): The VGG16 model instance (already initialized).
        vgg11_state_dict_path (str): Path to the saved VGG11 state_dict (.pth file).
        device: The device to load the VGG11 state_dict onto ('cpu' recommended here).
    """
    if not os.path.exists(vgg11_state_dict_path):
        print(f"Warning: VGG11 weights file not found at {vgg11_state_dict_path}. VGG16 will use random initialization.")
        return vgg16_model

    print(f"Loading partial weights from VGG11: {vgg11_state_dict_path}")
    vgg11_state_dict = torch.load(vgg11_state_dict_path, map_location=device)

    # --- Handle potential 'module.' prefix from DataParallel saving in VGG11 ---
    if list(vgg11_state_dict.keys())[0].startswith('module.'):
        print("Removing 'module.' prefix from VGG11 state_dict keys")
        vgg11_state_dict = {k.replace('module.', '', 1): v for k, v in vgg11_state_dict.items()}

    vgg16_dict = vgg16_model.state_dict() # Get current VGG16 state

    # --- Define Layer Mapping ---
    # Paper: "initialised the first four convolutional layers... with the layers of net A"
    # Mapping VGG11 (Net A) conv layers to VGG16 (Net D) conv layers based on structure
    # VGG11 Feature Idx -> VGG16 Feature Idx
    conv_mapping = {
        0: 0,  # VGG11 Conv 1 (idx 0) -> VGG16 Conv 1 (idx 0)
        3: 5,  # VGG11 Conv 2 (idx 3) -> VGG16 Conv 3 (idx 5)
        6: 10, # VGG11 Conv 3 (idx 6) -> VGG16 Conv 5 (idx 10)
        8: 12, # VGG11 Conv 4 (idx 8) -> VGG16 Conv 6 (idx 12)
        # VGG11 Conv 5,6,7,8 map to VGG16 Conv 8,9,11,12 if needed, but paper only mentions first 4
        # 11: 17, # VGG11 Conv 5 -> VGG16 Conv 8
        # 13: 19, # VGG11 Conv 6 -> VGG16 Conv 9
        # 16: 24, # VGG11 Conv 7 -> VGG16 Conv 11
        # 18: 26  # VGG11 Conv 8 -> VGG16 Conv 12
    }

    print("Copying convolutional layers (first 4 as per paper):")
    copied_conv_keys = set()
    for vgg11_idx, vgg16_idx in conv_mapping.items():
        for suffix in ['.weight', '.bias']:
            vgg11_key = f'features.{vgg11_idx}{suffix}'
            vgg16_key = f'features.{vgg16_idx}{suffix}'

            if vgg11_key in vgg11_state_dict and vgg16_key in vgg16_dict:
                if vgg16_dict[vgg16_key].shape == vgg11_state_dict[vgg11_key].shape:
                    vgg16_dict[vgg16_key] = vgg11_state_dict[vgg11_key]
                    copied_conv_keys.add(vgg16_key)
                    # print(f"  Copied: {vgg11_key} -> {vgg16_key}")
                else:
                    print(f"  Skipped (shape mismatch): {vgg11_key} vs {vgg16_key}")
            # else:
                # print(f"  Key missing: {vgg11_key} in VGG11 or {vgg16_key} in VGG16")
    print(f"Copied {len(copied_conv_keys)} conv layer parameter tensors.")


    # Paper: "and the last three fully-connected layers"
    # Classifier structure is identical
    # VGG11 Classifier Idx -> VGG16 Classifier Idx
    fc_mapping = {
        0: 0, # FC 1
        3: 3, # FC 2
        6: 6, # FC 3
    }

    print("Copying fully-connected layers:")
    copied_fc_keys = set()
    for vgg11_idx, vgg16_idx in fc_mapping.items():
         for suffix in ['.weight', '.bias']:
            vgg11_key = f'classifier.{vgg11_idx}{suffix}'
            vgg16_key = f'classifier.{vgg16_idx}{suffix}'

            if vgg11_key in vgg11_state_dict and vgg16_key in vgg16_dict:
                 if vgg16_dict[vgg16_key].shape == vgg11_state_dict[vgg11_key].shape:
                    vgg16_dict[vgg16_key] = vgg11_state_dict[vgg11_key]
                    copied_fc_keys.add(vgg16_key)
                    # print(f"  Copied: {vgg11_key} -> {vgg16_key}")
                 else:
                    print(f"  Skipped (shape mismatch): {vgg11_key} vs {vgg16_key}")
            # else:
                # print(f"  Key missing: {vgg11_key} in VGG11 or {vgg16_key} in VGG16")
    print(f"Copied {len(copied_fc_keys)} FC layer parameter tensors.")


    # Load the modified state dict into VGG16
    try:
        vgg16_model.load_state_dict(vgg16_dict)
        print("Partial weights loaded into VGG16 model successfully.")
    except RuntimeError as e:
        print(f"Error loading partial weights into VGG16: {e}")
        print("VGG16 will proceed with its initial random weights for potentially conflicting layers.")

    return vgg16_model

# --- Data Loading and Augmentation ---
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ]),
}
print("Initializing Datasets and Dataloaders...")
try:
    image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE,
                                                     shuffle=True if x=='train' else False,
                                                     num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True if NUM_WORKERS > 0 else False)
                      for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    print(f"Found {len(class_names)} classes.")
    print(f"Training set size: {dataset_sizes['train']}")
    print(f"Validation set size: {dataset_sizes['val']}")
except FileNotFoundError:
    print(f"ERROR: ImageNet dataset not found at '{DATA_DIR}'. Please set the correct path.")
    sys.exit(1)
except Exception as e:
    print(f"An error occurred during dataset loading: {e}")
    sys.exit(1)


# --- Device Setup ---
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Training Function (with Checkpointing and Ctrl+C handling) ---
def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25,
                model_save_path='model_final.pth', checkpoint_path='model_checkpoint.pth'):
    since = time.time()
    start_epoch = 0
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_model_wts_state_dict = None # Store the best state_dict

    # --- Checkpoint Loading ---
    if os.path.isfile(checkpoint_path):
        print(f"=> Loading checkpoint '{checkpoint_path}'")
        # Load checkpoint onto the same device the model will eventually be on
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # Use .get() for resilience against missing keys in older checkpoints
        start_epoch = checkpoint.get('epoch', -1) + 1 # Start from the next epoch

        # Load model state: Handle DataParallel prefix difference carefully
        model_state_dict_from_checkpoint = checkpoint.get('model_state_dict')
        if model_state_dict_from_checkpoint:
            is_model_dataparallel = isinstance(model, nn.DataParallel)
            # Check if the saved state dict has the 'module.' prefix
            saved_with_dataparallel = list(model_state_dict_from_checkpoint.keys())[0].startswith('module.')

            try:
                if is_model_dataparallel and not saved_with_dataparallel:
                    print("Adding 'module.' prefix to checkpoint keys for DataParallel model.")
                    new_state_dict = {'module.' + k: v for k, v in model_state_dict_from_checkpoint.items()}
                    model.load_state_dict(new_state_dict)
                elif not is_model_dataparallel and saved_with_dataparallel:
                    print("Removing 'module.' prefix from checkpoint keys for non-DataParallel model.")
                    new_state_dict = {k.replace('module.', '', 1): v for k, v in model_state_dict_from_checkpoint.items()}
                    model.load_state_dict(new_state_dict)
                else: # Prefixes match or neither uses 'module.'
                     model.load_state_dict(model_state_dict_from_checkpoint)
                print("Loaded model state from checkpoint.")
            except RuntimeError as e:
                print(f"Error loading model state dict from checkpoint, possibly due to architecture mismatch: {e}")
                print("Model weights will remain as initialized before checkpoint loading attempt.")
        else:
             print("Warning: 'model_state_dict' not found in checkpoint.")


        # Load optimizer and scheduler state carefully
        optimizer_state_dict = checkpoint.get('optimizer_state_dict')
        if optimizer_state_dict:
            try:
                optimizer.load_state_dict(optimizer_state_dict)
                print("Loaded optimizer state from checkpoint.")
            except ValueError as e:
                 print(f"Warning: Could not load optimizer state, possibly due to parameter mismatch or change: {e}")
                 print("Optimizer state will be re-initialized.")
        else:
            print("Warning: 'optimizer_state_dict' not found in checkpoint.")


        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler_state_dict = checkpoint.get('scheduler_state_dict')
            if scheduler_state_dict:
                try:
                    scheduler.load_state_dict(scheduler_state_dict)
                    print("Loaded scheduler state from checkpoint.")
                except Exception as e:
                     print(f"Warning: Could not load scheduler state: {e}")
                     print("Scheduler state will be re-initialized.")
            else:
                 print("Warning: 'scheduler_state_dict' not found in checkpoint.")


        best_acc = checkpoint.get('best_acc', 0.0)
        history = checkpoint.get('history', history)
        # Load the best model weights found so far if they were saved
        if 'best_model_wts_state_dict' in checkpoint:
             best_model_wts_state_dict = checkpoint.get('best_model_wts_state_dict')
             print(f"Restored previous best accuracy: {best_acc:.4f}")

        print(f"=> Checkpoint loaded. Resuming training from epoch {start_epoch}")

    else:
        print(f"=> No checkpoint found at '{checkpoint_path}', starting training from epoch 0.")
        # Initialize best_model_wts only if not loading checkpoint
        # Get state dict correctly handling DataParallel
        # best_model_wts_state_dict = copy.deepcopy(model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict())


    # --- Graceful Shutdown Handler ---
    interrupted = False
    def signal_handler(sig, frame):
        nonlocal interrupted
        if not interrupted: # Prevent multiple prints if Ctrl+C is held
            print("\nCtrl+C detected! Finishing current epoch and saving checkpoint...")
            interrupted = True
        else:
            print("\nMultiple Ctrl+C detected. Attempting forceful exit after checkpoint save...")
            # Optionally add a more forceful exit here if needed, but usually letting the epoch finish is best.

    prev_handler = signal.signal(signal.SIGINT, signal_handler) # Register handler

    # --- Main Training Loop ---
    for epoch in range(start_epoch, num_epochs):
        print(f'\nEpoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            running_top5_corrects = 0
            num_batches = len(dataloaders[phase])
            batch_times = []

            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                batch_start_time = time.time()
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    _, top5_preds = torch.topk(outputs, 5, dim=1)
                    top5_corrects_batch = torch.sum(top5_preds == labels.view(-1, 1).expand_as(top5_preds))

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                running_top5_corrects += top5_corrects_batch.item()
                batch_times.append(time.time() - batch_start_time)

                # Print progress less frequently for large datasets
                if i % 200 == 0 and i > 0:
                    avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0
                    eta_seconds = avg_batch_time * (num_batches - (i + 1))
                    eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
                    print(f'  {phase} Batch {i}/{num_batches} Loss: {loss.item():.4f} - Avg Batch Time: {avg_batch_time:.3f}s - ETA: {eta_str}')

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_top5_acc = running_top5_corrects / dataset_sizes[phase]

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Top-1 Acc: {epoch_acc:.4f} Top-5 Acc: {epoch_top5_acc:.4f}')

            # Update history and handle scheduler/best model saving during validation phase
            if phase == 'val':
                 history['val_loss'].append(epoch_loss)
                 history['val_acc'].append(epoch_acc.item()) # Store as float
                 if scheduler:
                     if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                         scheduler.step(epoch_loss) # ReduceLROnPlateau needs the metric
                     # elif isinstance(scheduler, lr_scheduler.StepLR):
                     #    scheduler.step() # StepLR steps per epoch without metric
                     else:
                          # Handle other potential schedulers if needed
                          pass


                 if epoch_acc > best_acc:
                     best_acc = epoch_acc
                     # Store the state_dict of the best model, handling DataParallel
                     best_model_wts_state_dict = copy.deepcopy(model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict())
                     print(f"*** New best Top-1 accuracy: {best_acc:.4f} (Epoch {epoch}) ***")
            else: # Training phase
                 history['train_loss'].append(epoch_loss)
                 history['train_acc'].append(epoch_acc.item()) # Store as float

        # --- Checkpoint Saving (End of Epoch) ---
        print("Saving checkpoint...")
        # Ensure we save the unwrapped state dict if using DataParallel
        current_model_state_to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        checkpoint_state = {
            'epoch': epoch,
            'model_state_dict': current_model_state_to_save,
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc.item() if torch.is_tensor(best_acc) else best_acc, # Ensure best_acc is float
            'history': history,
            'best_model_wts_state_dict': best_model_wts_state_dict # Save the best state dict found so far
        }
        if scheduler:
             checkpoint_state['scheduler_state_dict'] = scheduler.state_dict()

        try:
            torch.save(checkpoint_state, checkpoint_path)
            print(f"Checkpoint saved to '{checkpoint_path}' after epoch {epoch}")
        except Exception as e:
            print(f"Error saving checkpoint: {e}")


        # --- Check for interruption AFTER saving checkpoint ---
        if interrupted:
            print("Interruption signal processed. Exiting training loop.")
            break # Exit the main epoch loop cleanly

    # --- End of Training ---
    time_elapsed = time.time() - since
    print(f'\nTraining {("completed normally" if not interrupted else "interrupted")} in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    if interrupted:
         print(f"Latest checkpoint saved at '{checkpoint_path}' (reflects state after epoch {epoch}).")
         print(f"Resume training by running the script again.")
         signal.signal(signal.SIGINT, prev_handler) # Restore previous handler
         sys.exit(0) # Exit cleanly after saving checkpoint on interrupt

    # --- Actions after Normal Completion ---
    print(f'Best validation Top-1 Acc: {best_acc:4f}')

    # Save the best model weights separately if training completed normally
    if best_model_wts_state_dict:
         try:
            torch.save(best_model_wts_state_dict, model_save_path)
            print(f"Best performing model weights saved to {model_save_path}")
         except Exception as e:
             print(f"Error saving best model weights: {e}")
    else:
         print("No best model weights recorded (perhaps training was too short or validation accuracy didn't improve).")
         # Optionally save the final epoch's weights as the final model if no improvement occurred
         final_model_state_to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
         alt_save_path = model_save_path.replace('.pth','_lastepoch.pth')
         try:
            torch.save(final_model_state_to_save, alt_save_path)
            print(f"Saved last epoch model weights to {alt_save_path}")
         except Exception as e:
             print(f"Error saving last epoch model weights: {e}")


    # Load the best weights into the model for return if completed normally
    if best_model_wts_state_dict:
        print("Loading best model weights into model for return.")
        try:
            # Need to handle DataParallel loading again based on the *current* model state (is_model_dataparallel)
            is_model_dataparallel = isinstance(model, nn.DataParallel)
            # Check prefix in the saved best_model_wts_state_dict
            saved_with_dataparallel = list(best_model_wts_state_dict.keys())[0].startswith('module.')

            if is_model_dataparallel and not saved_with_dataparallel:
                 new_state_dict = {'module.' + k: v for k, v in best_model_wts_state_dict.items()}
                 model.load_state_dict(new_state_dict)
            elif not is_model_dataparallel and saved_with_dataparallel:
                 new_state_dict = {k.replace('module.', '', 1): v for k, v in best_model_wts_state_dict.items()}
                 model.load_state_dict(new_state_dict)
            else: # Prefixes match or neither has prefix
                 model.load_state_dict(best_model_wts_state_dict)
        except RuntimeError as e:
             print(f"Error loading best model weights back into model: {e}")


    signal.signal(signal.SIGINT, prev_handler) # Restore default handler
    return model, history


# ===========================================
# --- Main Execution Block ---
# ===========================================
if __name__ == "__main__":

    print("--- Initializing VGG16 ---")
    # Initialize VGG16 with its default random weights first
    vgg16_model = VGG16(num_classes=NUM_CLASSES, init_weights=True)

    # --- Load Partial Weights from VGG11 ---
    # Load onto CPU first before potentially wrapping with DataParallel
    vgg16_model = load_partial_weights(vgg16_model, VGG11_WEIGHTS_PATH, device='cpu')

    # --- Setup Multi-GPU (DataParallel) if available ---
    # Wrap the *initialized* model
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        vgg16_model = nn.DataParallel(vgg16_model)

    # --- Move Model to Target Device ---
    vgg16_model.to(device)
    print(f"VGG16 model (potentially wrapped) moved to device: {device}")

    # --- Optimizer, Loss, Scheduler ---
    # Ensure optimizer gets parameters from the model *after* potential DP wrapping
    optimizer_ft = optim.SGD(vgg16_model.parameters(), lr=INITIAL_LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    lr_scheduler_ft = lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', factor=LR_DECAY_FACTOR, patience=LR_PATIENCE, verbose=True)

    # --- Start Training ---
    print("\n--- Starting VGG16 Training ---")
    trained_vgg16_model, training_history = train_model(
        model=vgg16_model,
        dataloaders=dataloaders_dict,
        criterion=criterion,
        optimizer=optimizer_ft,
        scheduler=lr_scheduler_ft,
        num_epochs=NUM_EPOCHS,
        model_save_path=VGG16_MODEL_SAVE_PATH,      # Pass VGG16 paths
        checkpoint_path=VGG16_CHECKPOINT_PATH       # Pass VGG16 paths
    )

    # --- Post-Training ---
    print("\n--- VGG16 Training Process Finished ---")
    # Example: Print final validation accuracy history
    val_acc_history = training_history.get('val_acc', [])
    if val_acc_history:
        print("Validation accuracy per epoch:")
        print([f"Epoch {i}: {acc:.4f}" for i, acc in enumerate(val_acc_history)])
