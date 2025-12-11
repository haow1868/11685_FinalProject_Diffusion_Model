import torch
import os

def load_checkpoint(unet, scheduler, vae=None, class_embedder=None, optimizer=None, lr_scheduler=None, ema=None, checkpoint_path='checkpoints/checkpoint.pth'):
    
    print("loading checkpoint")
    checkpoint = torch.load(checkpoint_path, weights_only = False)
    
    print("loading unet")
    unet.load_state_dict(checkpoint['unet_state_dict'])
    print("loading scheduler")
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    if vae is not None and 'vae_state_dict' in checkpoint:
        print("loading vae")
        vae.load_state_dict(checkpoint['vae_state_dict'])
    
    if class_embedder is not None and 'class_embedder_state_dict' in checkpoint:
        print("loading class_embedder")
        class_embedder.load_state_dict(checkpoint['class_embedder_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        print("loading optimizer")
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if lr_scheduler is not None:
        if 'lr_scheduler_state_dict' in checkpoint:
            print("loading lr_scheduler")
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        else:
            print("WARNING: lr_scheduler state not found in checkpoint. It will be reset.")

    if ema is not None:
        if 'ema_state_dict' in checkpoint:
            print("loading ema")
            ema.shadow = checkpoint['ema_state_dict']
        else:
            print("WARNING: EMA state not found. Initializing EMA from CURRENT model weights.")

    
    return checkpoint.get('epoch', 0)
        

def save_checkpoint(unet, scheduler, vae=None, class_embedder=None, optimizer=None, epoch=None, ema=None, lr_scheduler=None, save_dir='checkpoints'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Define checkpoint file name
    #checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth')
    checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_last.pth')

    checkpoint = {
        'unet_state_dict': unet.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }
    
    if vae is not None:
        checkpoint['vae_state_dict'] = vae.state_dict()
    
    if class_embedder is not None:
        checkpoint['class_embedder_state_dict'] = class_embedder.state_dict()
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if epoch is not None:
        checkpoint['epoch'] = epoch
    
    if lr_scheduler is not None:
        checkpoint['lr_scheduler_state_dict'] = lr_scheduler.state_dict()

    if ema is not None:
        checkpoint['ema_state_dict'] = ema.shadow

    if epoch is not None:
        checkpoint['epoch'] = epoch
    
    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")
    
    # Manage checkpoint history
    #manage_checkpoints(save_dir, keep_last_n=10)


def manage_checkpoints(save_dir, keep_last_n=10):
    # List all checkpoint files in the save directory
    checkpoints = [f for f in os.listdir(save_dir) if f.startswith('checkpoint_epoch_')]
    checkpoints.sort(key=lambda f: int(f.split('_')[-1].split('.')[0]))  # Sort by epoch number

    # If more than `keep_last_n` checkpoints exist, remove the oldest ones
    if len(checkpoints) > keep_last_n + 1:  # keep_last_n + 1 to account for the latest checkpoint
        for checkpoint_file in checkpoints[:-keep_last_n-1]:
            checkpoint_path = os.path.join(save_dir, checkpoint_file)
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
                print(f"Removed old checkpoint: {checkpoint_path}")