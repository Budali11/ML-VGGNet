import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import os
import time
import copy
import argparse

# --- VGG16 Model Definition ---
# Based on Table 1, Configuration D (16 weight layers)

class VGG16(nn.Module):
    def __init__(self, num_classes=1000, init_weights=True):
        super(VGG16, self).__init__()

        # Convolutional Blocks (Conv + ReLU) followed by MaxPool
        # All conv layers use 3x3 filters, stride 1, padding 1
        # MaxPool uses 2x2 window, stride 2
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Adaptive Average Pooling to handle different input sizes after conv layers
        # Output size is 7x7 for standard 224x224 input after 5 maxpools
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        # Classifier (Fully Connected Layers)
        # Includes ReLU and Dropout(0.5) for first two FC layers
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5), # Dropout specified in paper
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5), # Dropout specified in paper
            nn.Linear(4096, num_classes),
            # Note: Softmax is implicitly handled by nn.CrossEntropyLoss
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1) # Flatten the features
        x = self.classifier(x)
        return x

    # Weight Initialization (Xavier/Glorot uniform for Conv/Linear weights, 0 for biases)
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Xavier initialization for Conv layers
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # Although not in VGG16 paper, good practice if added later
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Xavier initialization for Linear layers
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0) # Initialize biases to 0 as per paper


def load_partial_weights(vgg16_model, vgg11_state_dict_path, device):
    """
    Initializes VGG16 layers based on VGG11 weights as described in the paper.
    - First 4 conv layers from VGG11 -> corresponding VGG16 conv layers.
    - Last 3 FC layers from VGG11 -> corresponding VGG16 FC layers.
    - Intermediate layers keep their random initialization.

    Args:
        vgg16_model (nn.Module): The VGG16 model instance (already initialized).
        vgg11_state_dict_path (str): Path to the saved VGG11 state_dict (.pth file).
        device: The device to load the VGG11 state_dict onto.
    """
    if not os.path.exists(vgg11_state_dict_path):
        print(f"Warning: VGG11 weights file not found at {vgg11_state_dict_path}. VGG16 will use random initialization.")
        return vgg16_model

    print(f"Loading partial weights from VGG11: {vgg11_state_dict_path}")
    vgg11_state_dict = torch.load(vgg11_state_dict_path, map_location=device)

    # --- Handle potential 'module.' prefix from DataParallel saving ---
    # We assume vgg16_model is NOT wrapped in DataParallel yet.
    # If vgg11_state_dict keys start with 'module.', remove it.
    if list(vgg11_state_dict.keys())[0].startswith('module.'):
        print("Removing 'module.' prefix from VGG11 state_dict keys")
        vgg11_state_dict = {k.replace('module.', '', 1): v for k, v in vgg11_state_dict.items()}

    vgg16_dict = vgg16_model.state_dict()

    # --- Mapping based on layer *order* and paper description ---

    # Convolutional Layers (First 4)
    # Map VGG11 Conv layer index to VGG16 Conv layer index within features
    # VGG11 Conv #: VGG11 features index -> VGG16 features index
    conv_mapping = {
        0: 0,  # VGG11 Conv 1 (features[0]) -> VGG16 Conv 1 (features[0])
        3: 5,  # VGG11 Conv 2 (features[3]) -> VGG16 Conv 3 (features[5])
        6: 10, # VGG11 Conv 3 (features[6]) -> VGG16 Conv 5 (features[10])
        8: 12, # VGG11 Conv 4 (features[8]) -> VGG16 Conv 7 (features[12])
        # VGG11 only has 8 conv layers in features
    }

    print("Copying convolutional layers:")
    for vgg11_idx, vgg16_idx in conv_mapping.items():
        # Construct state_dict keys
        vgg11_w_key = f'features.{vgg11_idx}.weight'
        vgg11_b_key = f'features.{vgg11_idx}.bias'
        vgg16_w_key = f'features.{vgg16_idx}.weight'
        vgg16_b_key = f'features.{vgg16_idx}.bias'

        if vgg11_w_key in vgg11_state_dict and vgg16_w_key in vgg16_dict:
            vgg16_dict[vgg16_w_key] = vgg11_state_dict[vgg11_w_key]
            print(f"  Copied: {vgg11_w_key} -> {vgg16_w_key}")
        if vgg11_b_key in vgg11_state_dict and vgg16_b_key in vgg16_dict:
            vgg16_dict[vgg16_b_key] = vgg11_state_dict[vgg11_b_key]
            print(f"  Copied: {vgg11_b_key} -> {vgg16_b_key}")

    # Fully-Connected Layers (Last 3 - structure is identical)
    # Map VGG11 Classifier layer index to VGG16 Classifier layer index
    # VGG11 FC #: VGG11 classifier index -> VGG16 classifier index
    fc_mapping = {
        0: 0, # VGG11 FC 1 (classifier[0]) -> VGG16 FC 1 (classifier[0])
        3: 3, # VGG11 FC 2 (classifier[3]) -> VGG16 FC 2 (classifier[3])
        6: 6, # VGG11 FC 3 (classifier[6]) -> VGG16 FC 3 (classifier[6])
    }

    print("Copying fully-connected layers:")
    for vgg11_idx, vgg16_idx in fc_mapping.items():
        # Construct state_dict keys
        vgg11_w_key = f'classifier.{vgg11_idx}.weight'
        vgg11_b_key = f'classifier.{vgg11_idx}.bias'
        vgg16_w_key = f'classifier.{vgg16_idx}.weight'
        vgg16_b_key = f'classifier.{vgg16_idx}.bias'

        if vgg11_w_key in vgg11_state_dict and vgg16_w_key in vgg16_dict:
            vgg16_dict[vgg16_w_key] = vgg11_state_dict[vgg11_w_key]
            print(f"  Copied: {vgg11_w_key} -> {vgg16_w_key}")
        if vgg11_b_key in vgg11_state_dict and vgg16_b_key in vgg16_dict:
            vgg16_dict[vgg16_b_key] = vgg11_state_dict[vgg11_b_key]
            print(f"  Copied: {vgg11_b_key} -> {vgg16_b_key}")

    # Load the modified state dict into VGG16
    vgg16_model.load_state_dict(vgg16_dict)
    print("Partial weights loaded into VGG16 model.")
    return vgg16_model
# --- Training Script ---

# --- Modified Training Function ---
def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25,
                model_save_path=MODEL_SAVE_PATH, checkpoint_path=CHECKPOINT_PATH):
    since = time.time()

    start_epoch = 0
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    # --- Checkpoint Loading ---
    if os.path.isfile(checkpoint_path):
        print(f"=> Loading checkpoint '{checkpoint_path}'")
        # Load checkpoint onto the same device model will be on
        checkpoint = torch.load(checkpoint_path, map_location=device)
        start_epoch = checkpoint['epoch'] + 1 # Start from the next epoch

        # Handle DataParallel wrapper prefix ('module.') if necessary
        # If the checkpoint was saved with DataParallel but loaded without, or vice-versa
        # state_dict = checkpoint['model_state_dict']
        # new_state_dict = {}
        # if isinstance(model, nn.DataParallel) and not list(state_dict.keys())[0].startswith('module.'):
        #     print("Adding 'module.' prefix to checkpoint keys")
        #     for k, v in state_dict.items():
        #         new_state_dict['module.' + k] = v
        # elif not isinstance(model, nn.DataParallel) and list(state_dict.keys())[0].startswith('module.'):
        #     print("Removing 'module.' prefix from checkpoint keys")
        #     for k, v in state_dict.items():
        #         new_state_dict[k.replace('module.', '', 1)] = v
        # else:
        #      new_state_dict = state_dict # No prefix adjustment needed

        # A simpler way (often works): If saved WITH DP and loading WITH DP OR saved WITHOUT and loading WITHOUT
        # Or if saved WITH DP and loading WITHOUT: model.load_state_dict() might handle it if strict=False
        # Best practice: Save model.module.state_dict() if using DataParallel
        # Let's assume for now the state dict matches or load_state_dict handles it with strict=False
        # If using DP, it's generally safer to save model.module.state_dict()
        if isinstance(model, nn.DataParallel):
             model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
             model.load_state_dict(checkpoint['model_state_dict'])


        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and 'scheduler_state_dict' in checkpoint: # Check if scheduler state exists
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        best_acc = checkpoint.get('best_acc', 0.0) # Load best_acc if saved
        history = checkpoint.get('history', history) # Load history if saved
        print(f"=> Loaded checkpoint '{checkpoint_path}' (epoch {checkpoint['epoch']})")
    else:
        print(f"=> No checkpoint found at '{checkpoint_path}', starting from scratch.")
        # Initialize best_model_wts only if not loading checkpoint
        best_model_wts = copy.deepcopy(model.state_dict()) # Initialize here if starting fresh


    # --- Graceful Shutdown Handler ---
    interrupted = False
    def signal_handler(sig, frame):
        nonlocal interrupted
        print("\nCtrl+C detected! Saving checkpoint before exiting...")
        interrupted = True
        # Saving logic will be handled after the current epoch finishes or immediately if needed

    signal.signal(signal.SIGINT, signal_handler) # Register handler for Ctrl+C


    # --- Main Training Loop ---
    for epoch in range(start_epoch, num_epochs):
        if interrupted: # Check if interrupted flag is set
             break # Exit the loop

        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            running_top5_corrects = 0

            # Iterate over data.
            num_batches = len(dataloaders[phase])
            start_time_batches = time.time()

            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                if interrupted: # Allow faster exit if needed within batch loop
                    break

                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    _, top5_preds = torch.topk(outputs, 5, dim=1)
                    top5_corrects = torch.sum(top5_preds == labels.view(-1, 1).expand_as(top5_preds))

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                batch_loss = loss.item() * inputs.size(0)
                batch_corrects = torch.sum(preds == labels.data)
                batch_top5_corrects = top5_corrects.item()

                running_loss += batch_loss
                running_corrects += batch_corrects
                running_top5_corrects += batch_top5_corrects

                if i % 100 == 0 and i > 0:
                     elapsed_batches = time.time() - start_time_batches
                     batches_per_sec = (i+1) / elapsed_batches if elapsed_batches > 0 else 0
                     eta_seconds = (num_batches - (i+1)) / batches_per_sec if batches_per_sec > 0 else 0
                     eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
                     print(f'  Batch {i+1}/{num_batches} - Loss: {batch_loss/inputs.size(0):.4f} Acc: {batch_corrects.double()/inputs.size(0):.4f} Top5 Acc: {batch_top5_corrects/inputs.size(0):.4f} - ETA: {eta_str}')

            if interrupted: # Exit outer loop too
                break

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_top5_acc = running_top5_corrects / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Top-5 Acc: {epoch_top5_acc:.4f}')

            if phase == 'val':
                 history['val_loss'].append(epoch_loss)
                 history['val_acc'].append(epoch_acc)
                 if scheduler:
                     # Special handling for ReduceLROnPlateau which steps on metrics
                     if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                         scheduler.step(epoch_loss)
                     # Handle other schedulers that step per epoch (might need adjustment if scheduler steps per batch)
                     # else:
                     #    scheduler.step() # Uncomment if using StepLR etc.

                 if epoch_acc > best_acc:
                     best_acc = epoch_acc
                     # Save the best model separately (optional but good practice)
                     # Ensure you save model.module.state_dict if using DataParallel
                     best_model_wts_to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
                     torch.save(best_model_wts_to_save, model_save_path.replace('.pth', '_best.pth'))
                     print(f"*** Best model saved with accuracy: {best_acc:.4f} ***")
            else:
                 history['train_loss'].append(epoch_loss)
                 history['train_acc'].append(epoch_acc)

        # --- Checkpoint Saving (End of Epoch) ---
        print("Saving checkpoint...")
        checkpoint_state = {
            'epoch': epoch,
            # Important: Save model.module.state_dict() if using DataParallel
            'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc,
            'history': history,
        }
        if scheduler: # Save scheduler state if it exists
             checkpoint_state['scheduler_state_dict'] = scheduler.state_dict()

        torch.save(checkpoint_state, checkpoint_path)
        print(f"Checkpoint saved to '{checkpoint_path}' after epoch {epoch}")
        
    # --- End of Training ---
    time_elapsed = time.time() - since
    if not interrupted:
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')
        # Load best model weights at the very end
        # Ensure the path points to the saved best model weights
        best_model_path = model_save_path.replace('.pth', '_best.pth')
        if os.path.exists(best_model_path):
             print(f"Loading best model weights from {best_model_path}")
             best_weights = torch.load(best_model_path, map_location=device)
             if isinstance(model, nn.DataParallel):
                  model.module.load_state_dict(best_weights)
             else:
                  model.load_state_dict(best_weights)

        # Save the final model (optional, could be the last epoch's or the best)
        final_model_state_to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        torch.save(final_model_state_to_save, model_save_path)
        print(f"Final model state saved to {model_save_path}")

    else: # If training was interrupted
         print(f'Training interrupted after {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
         print(f"Resume training by running the script again. Checkpoint saved at '{checkpoint_path}'")
         sys.exit(0) # Exit cleanly after saving checkpoint


    return model, history


# --- Main Execution ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch VGG16 ImageNet Training')
    parser.add_argument('--data', metavar='DIR', required=True,
                        help='path to ImageNet dataset')
    parser.add_argument('--epochs', default=74, type=int, metavar='N',
                        help='number of total epochs to run (default: 74 as per paper reference)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256 as per paper)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate (default: 0.01 as per paper)', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum (default: 0.9 as per paper)')
    parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 5e-4 as per paper)',
                        dest='weight_decay')
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')

    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        # Potentially add more seeding here if using CUDA extensively and need determinism
        # torch.cuda.manual_seed(args.seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False


    # --- Data Loading and Preprocessing ---
    data_dir = args.data
    image_size = 224 # VGG input size
    # S=256 used for training in one of the paper's settings
    train_resize_scale = 256

    # Standard ImageNet normalization values
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    # Note: Paper mentioned only subtracting mean RGB. Standard practice uses
    # full normalization (mean+std) which generally works better.
    # If you MUST only subtract mean, calculate the mean RGB across the training
    # set and use transforms.Normalize(mean=calculated_mean, std=[1.0, 1.0, 1.0]).

    data_transforms = {
        'train': transforms.Compose([
            # Paper: Rescale S, then random 224x224 crop
            transforms.Resize(train_resize_scale), # Rescale smaller side to S=256
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(), # Random horizontal flip
            # transforms.ColorJitter(...) # Optional: Add for RGB color shift
            transforms.ToTensor(),
            normalize,
        ]),
        'val': transforms.Compose([
            # Paper: Rescale Q (often Q=S for validation), then center crop
            transforms.Resize(train_resize_scale), # Typically use same scale as train base
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]),
    }

    print("Initializing Datasets and Dataloaders...")

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size,
                                                 shuffle=(x == 'train'), num_workers=args.workers,
                                                 pin_memory=True) # pin_memory for faster GPU transfer
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    num_classes = len(class_names)

    print(f"Found {num_classes} classes.")
    print(f"Dataset sizes: Train={dataset_sizes['train']}, Val={dataset_sizes['val']}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # --- Model, Loss, Optimizer, Scheduler Setup ---
    print("Initializing VGG16 Model...")
    vgg16_model = VGG16(num_classes=NUM_CLASSES, init_weights=True) # Initialize with random weights first
    print(vgg16_model) # Print model structure

    # Load partial weights from VGG11 *before* moving to GPU or wrapping with DP
    vgg16_model = load_partial_weights(vgg16_model, VGG11_WEIGHTS_PATH, device='cpu') # Load weights on CPU first
    # Move the potentially wrapped model to the target device
    vgg16_model.to(device)
    print("VGG16 model moved to device.")

    # Loss Function
    criterion = nn.CrossEntropyLoss()

    # Optimizer (SGD as per paper)
    optimizer_ft = optim.SGD(vgg16_model.parameters(),
                             lr=args.lr,
                             momentum=args.momentum,
                             weight_decay=args.weight_decay)

    # Learning Rate Scheduler
    # Paper: Decrease LR by factor of 10 when validation accuracy stops improving (3 times total)
    # ReduceLROnPlateau fits this description well.
    # factor=0.1 (decrease by 10), patience=5-10 epochs is common
    scheduler_ft = lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='max', factor=0.1, patience=7, verbose=True)
    # Alternatively, if you know the exact epochs for decay:
    # scheduler_ft = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[30, 50, 65], gamma=0.1)


    # --- Start Training ---
    best_model = train_model(vgg16_model, criterion, optimizer_ft, scheduler_ft,
                             dataloaders, device, num_epochs=args.epochs, dataset_sizes=dataset_sizes)

    # Optional: Save the final best model
    # torch.save(best_model.state_dict(), 'vgg16_imagenet_best.pth')
    print("Training finished.")
