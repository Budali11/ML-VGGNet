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

print(f"PyTorch Version: {torch.__version__}")
print(f"Torchvision Version: {torchvision.__version__}")

# --- Configuration ---
DATA_DIR = '/home/wbc/vgg/dataset' # <<<--- IMPORTANT: SET THIS PATH
MODEL_SAVE_PATH = './vgg11_imagenet.pth'
NUM_CLASSES = 1000  # ImageNet has 1000 classes
BATCH_SIZE = 256  # As per paper (adjust based on your GPU memory)
NUM_EPOCHS = 74   # As per paper (can take a very long time)
INITIAL_LR = 0.01 # As per paper (1e-2)
MOMENTUM = 0.9    # As per paper
WEIGHT_DECAY = 5e-4 # As per paper (L2 penalty)
NUM_WORKERS = 8   # Adjust based on your system's capabilities
LR_DECAY_FACTOR = 0.1 # Decrease LR by factor of 10
LR_PATIENCE = 10  # Number of epochs with no improvement after which LR is reduced
                  # The paper states reduction "when the validation set accuracy stopped improving"
                  # ReduceLROnPlateau simulates this. Paper mentions 3 drops in 74 epochs.
                  # You might need to tune patience or use a StepLR scheduler if preferred.

# --- VGG11 Model Definition ---
class VGG11(nn.Module):
    def __init__(self, num_classes=1000, init_weights=True):
        super(VGG11, self).__init__()
        # Based on Configuration 'A' in the VGG paper (11 weight layers)
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # AdaptiveAvgPool allows different input sizes if needed later,
        # but for fixed 224x224 input, the output feature map size is 7x7
        # self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096), # Input size = 512 channels * 7x7 feature map
            nn.ReLU(True),
            nn.Dropout(p=0.5), # Dropout ratio 0.5 as per paper
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5), # Dropout ratio 0.5 as per paper
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        # x = self.avgpool(x) # Not strictly needed if input is always 224x224
        x = torch.flatten(x, 1) # Flatten the feature map
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        # Initialization as described in the paper:
        # Weights from normal distribution N(0, 0.01), biases initialized to 0
        # Note: Modern practices (like Kaiming init) often work better,
        # but this follows the paper's description.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # The paper uses std=0.01 (variance 1e-4), but common VGG impls use sqrt(2/n)
                # Let's follow the text: N(0, 0.01^2) -> std=0.01
                # nn.init.normal_(m.weight, 0, 0.01)
                # Let's use Kaiming He initialization which is common practice now
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d): # Although VGG11 doesn't use BN by default
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Paper: N(0, 0.01^2) for weights, 0 for biases
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

# --- Data Loading and Augmentation ---

# Mean and std dev for ImageNet
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# Data augmentation and normalization for training
# Following practices from VGG paper and common PyTorch examples
# Paper mentions random crops from rescaled images, random horizontal flip, RGB color shift
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224), # Random crop from rescaled image
        transforms.RandomHorizontalFlip(),  # Random horizontal flip
        # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1), # Simulates RGB color shift (optional)
        transforms.ToTensor(),
        normalize
    ]),
    'val': transforms.Compose([
        transforms.Resize(256), # Rescale smallest side to 256
        transforms.CenterCrop(224), # Take center 224x224 crop
        transforms.ToTensor(),
        normalize
    ]),
}

print("Initializing Datasets and Dataloaders...")

# Create training and validation datasets
image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x])
                  for x in ['train', 'val']}
# Create training and validation dataloaders
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE,
                                                 shuffle=True if x=='train' else False, # Shuffle only training data
                                                 num_workers=NUM_WORKERS, pin_memory=True)
                  for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
print(f"Found {len(class_names)} classes.")
print(f"Training set size: {dataset_sizes['train']}")
print(f"Validation set size: {dataset_sizes['val']}")

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Model Initialization ---
print("Initializing Model...")
model_ft = VGG11(num_classes=NUM_CLASSES, init_weights=True)

# --- Multi-GPU Setup (using the provided snippet) ---
if torch.cuda.device_count() > 1:
   print(f"使用 {torch.cuda.device_count()} 个 GPUs!")
   model_ft = nn.DataParallel(model_ft) # 使用 DataParallel
model_ft.to(device) # Transfer model to GPU(s)

# --- Optimizer and Loss Function ---
# Paper uses SGD with momentum=0.9, weight_decay=5e-4, initial lr=0.01
optimizer_ft = optim.SGD(model_ft.parameters(), lr=INITIAL_LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

# Loss function
criterion = nn.CrossEntropyLoss()

# Learning rate scheduler - Reduce LR when validation loss plateaus
# This aligns with the paper's description "decreased by a factor of 10 when the validation set accuracy stopped improving"
lr_scheduler_ft = lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', factor=LR_DECAY_FACTOR, patience=LR_PATIENCE, verbose=True)
# Alternatively, use StepLR if you prefer fixed epoch drops:
# lr_scheduler_ft = lr_scheduler.StepLR(optimizer_ft, step_size=25, gamma=0.1) # Drop LR every 25 epochs

# --- Training Function ---
def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25, model_save_path=MODEL_SAVE_PATH):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
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
            running_top5_corrects = 0 # Track top-5 accuracy

            # Iterate over data.
            num_batches = len(dataloaders[phase])
            start_time_batches = time.time()

            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # Calculate Top-5 accuracy
                    _, top5_preds = torch.topk(outputs, 5, dim=1)
                    top5_corrects = torch.sum(top5_preds == labels.view(-1, 1).expand_as(top5_preds))

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                batch_loss = loss.item() * inputs.size(0)
                batch_corrects = torch.sum(preds == labels.data)
                batch_top5_corrects = top5_corrects.item() # Use .item()

                running_loss += batch_loss
                running_corrects += batch_corrects
                running_top5_corrects += batch_top5_corrects

                if i % 100 == 0 and i > 0: # Print progress every 100 batches
                     elapsed_batches = time.time() - start_time_batches
                     batches_per_sec = (i+1) / elapsed_batches if elapsed_batches > 0 else 0
                     eta_seconds = (num_batches - (i+1)) / batches_per_sec if batches_per_sec > 0 else 0
                     eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
                     print(f'  Batch {i+1}/{num_batches} - Loss: {batch_loss/inputs.size(0):.4f} Acc: {batch_corrects.double()/inputs.size(0):.4f} Top5 Acc: {batch_top5_corrects/inputs.size(0):.4f} - ETA: {eta_str}')


            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_top5_acc = running_top5_corrects / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Top-5 Acc: {epoch_top5_acc:.4f}')

            # Deep copy the model if best accuracy achieved
            if phase == 'val':
                 history['val_loss'].append(epoch_loss)
                 history['val_acc'].append(epoch_acc)
                 # Adjust LR based on validation loss
                 scheduler.step(epoch_loss)
                 if epoch_acc > best_acc:
                     best_acc = epoch_acc
                     best_model_wts = copy.deepcopy(model.state_dict())
                     # Save the best model
                     torch.save(best_model_wts, model_save_path.replace('.pth', '_best.pth'))
                     print(f"*** Best model saved with accuracy: {best_acc:.4f} ***")
            else:
                 history['train_loss'].append(epoch_loss)
                 history['train_acc'].append(epoch_acc)

        # Save checkpoint every few epochs
        if epoch % 5 == 0:
             checkpoint_path = model_save_path.replace('.pth', f'_epoch{epoch}.pth')
             torch.save(model.state_dict(), checkpoint_path)
             print(f"Checkpoint saved to {checkpoint_path}")

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, history

# --- Start Training ---
print("Starting Training...")
model_ft, history = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, lr_scheduler_ft,
                       num_epochs=NUM_EPOCHS, model_save_path=MODEL_SAVE_PATH)

# Save the final model (might not be the best one)
torch.save(model_ft.state_dict(), MODEL_SAVE_PATH)
print(f"Final model saved to {MODEL_SAVE_PATH}")

# You can analyze the 'history' dictionary for plotting loss/accuracy curves
print("Training History:")
print(history)
