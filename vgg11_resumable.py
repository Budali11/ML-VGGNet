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
import signal # Import signal handling
import sys    # Import sys for exit

# --- Add Checkpoint Path Configuration ---
CHECKPOINT_PATH = './vgg11_imagenet_checkpoint.pth' # Path to save/load checkpoint

# ... (Keep VGG11 class, data loading, model init, multi-gpu setup, optimizer, criterion the same) ...

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

# --- Start Training (no change here) ---
print("Initializing Model...")
model_ft = VGG11(num_classes=NUM_CLASSES, init_weights=True) # Or False if loading weights from checkpoint

# --- Multi-GPU Setup ---
if torch.cuda.device_count() > 1:
   print(f"使用 {torch.cuda.device_count()} 个 GPUs!")
   model_ft = nn.DataParallel(model_ft)
model_ft.to(device) # Transfer model to GPU(s)

# --- Optimizer and Scheduler ---
optimizer_ft = optim.SGD(model_ft.parameters(), lr=INITIAL_LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
criterion = nn.CrossEntropyLoss()
lr_scheduler_ft = lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', factor=LR_DECAY_FACTOR, patience=LR_PATIENCE, verbose=True)

print("Starting Training...")
# Pass the checkpoint path to the training function
model_ft, history = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, lr_scheduler_ft,
                       num_epochs=NUM_EPOCHS, model_save_path=MODEL_SAVE_PATH,
                       checkpoint_path=CHECKPOINT_PATH) # Pass checkpoint path

# --- Post-Training (Optional Analysis) ---
if history: # Only print history if training wasn't interrupted immediately
    print("Training History:")
    print(history)
