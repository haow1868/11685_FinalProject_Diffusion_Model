import os 
import sys 
import argparse
import numpy as np
import ruamel.yaml as yaml
import torch
import wandb 
import logging 
from logging import getLogger as get_logger
from tqdm import tqdm 
from PIL import Image
import torch.nn.functional as F

from torchvision import datasets, transforms
from torchvision.utils  import make_grid, save_image
from torchvision.models import inception_v3
from torchvision.models import inception_v3, Inception_V3_Weights

from models import UNet, VAE, ClassEmbedder
from schedulers import DDPMScheduler, DDIMScheduler
from pipelines import DDPMPipeline
from utils import seed_everything, load_checkpoint

from train import parse_args

logger = get_logger(__name__)


def select_best_per_class_centroid(all_images, num_classes, images_per_class, device):
    """
    Selects the single best image for each class using the Centroid Method with InceptionV3.
    Returns a tensor of shape [Num_Classes, C, H, W].
    """
    logger.info("Selecting best image per class using Centroid Method (InceptionV3)")
    
    # 1. Load InceptionV3 for feature extraction
    weights = Inception_V3_Weights.DEFAULT
    model = inception_v3(weights=weights).to(device)
    model.eval()
    model.fc = torch.nn.Identity() # Replace classification head with identity
    
    best_images_list = []

    # 2. Iterate over each class chunk
    with torch.no_grad():
        for class_idx in range(num_classes):
            start_idx = class_idx * images_per_class
            end_idx = start_idx + images_per_class
            
            # Get the batch of images for this specific class
            class_imgs = all_images[start_idx:end_idx].to(device)
            
            # Preprocess for Inception (Resize to 299x299)
            # Input needs to be (N, 3, 299, 299)
            imgs_resized = F.interpolate(class_imgs, size=(299, 299), mode='bilinear', align_corners=False)
            
            # Extract features [50, 2048]
            features = model(imgs_resized)
            
            # 3. Calculate Centroid (Mean) for this class
            centroid = features.mean(dim=0, keepdim=True)
            
            # 4. Find the image closest to the centroid
            # Euclidean distance
            distances = torch.cdist(features, centroid).squeeze()
            best_idx_local = torch.argmin(distances).item()
            
            # Select the winner
            best_img = class_imgs[best_idx_local]
            best_images_list.append(best_img)
            
            logger.info(f"Class {class_idx}: Selected image index {best_idx_local} (dist {distances[best_idx_local]:.4f})")

    return torch.stack(best_images_list)

def main():
    # parse arguments
    args = parse_args()
    
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # seed everything
    seed_everything(args.seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    
    # setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    # setup model
    logger.info("Creating model")
    # unet
    unet = UNet(input_size=args.unet_in_size, input_ch=args.unet_in_ch, T=args.num_train_timesteps, ch=args.unet_ch, ch_mult=args.unet_ch_mult, attn=args.unet_attn, num_res_blocks=args.unet_num_res_blocks, dropout=args.unet_dropout, conditional=args.use_cfg, c_dim=args.unet_ch)
    # preint number of parameters
    num_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    logger.info(f"Number of parameters: {num_params / 10 ** 6:.2f}M")
    
    # TODO: ddpm shceduler
    scheduler = DDPMScheduler(beta_start=args.beta_start,
                              beta_end=args.beta_end,
                              num_train_timesteps=args.num_train_timesteps,
                              num_inference_steps = args.num_inference_steps,
                              beta_schedule = args.beta_schedule,
                              prediction_type = args.prediction_type,
                              clip_sample = args.clip_sample,
                              clip_sample_range = args.clip_sample_range)
    # vae 
    vae = None
    if args.latent_ddpm:        
        vae = VAE()
        vae.init_from_ckpt('pretrained/model.ckpt')
        vae.eval()
    # cfg
    class_embedder = None
    if args.use_cfg:
        # TODO: class embeder
        class_embedder = ClassEmbedder(embed_dim=args.unet_ch, n_classes=args.num_classes)
        
    # send to device
    unet = unet.to(device)
    scheduler = scheduler.to(device)
    if vae:
        vae = vae.to(device)
    if class_embedder:
        class_embedder = class_embedder.to(device)
        
    # scheduler
    if args.use_ddim:
        shceduler_class = DDIMScheduler
    else:
        shceduler_class = DDPMScheduler
    # TOOD: scheduler
    scheduler = shceduler_class(beta_start=args.beta_start,
                                beta_end=args.beta_end,
                                num_train_timesteps=args.num_train_timesteps,
                                num_inference_steps=args.num_inference_steps,
                                beta_schedule=args.beta_schedule,
                                prediction_type=args.prediction_type,
                                clip_sample=args.clip_sample,
                                clip_sample_range=args.clip_sample_range)
    scheduler = scheduler.to(device)

    # load checkpoint
    load_checkpoint(unet, scheduler, vae=vae, class_embedder=class_embedder, checkpoint_path=args.ckpt)
    
    # TODO: pipeline
    pipeline = DDPMPipeline(unet=unet, scheduler=scheduler, vae=vae, class_embedder=class_embedder)

    
    logger.info("***** Running Infrence *****")
    
    # TODO: we run inference to generation 5000 images
    # TODO: with cfg, we generate 50 images per class 
    all_images = []
    if args.use_cfg:
        batch_size = 50
        # generate 50 images per class
        for i in tqdm(range(args.num_classes), desc="Generating images"):
            logger.info(f"Generating 50 images for class {i}")
            classes = torch.full((batch_size,), i, dtype=torch.long, device=device)
            gen_images =  pipeline(
                batch_size=batch_size,
                num_inference_steps=args.num_inference_steps,
                classes=classes.tolist(),
                guidance_scale=args.cfg_guidance_scale,
                generator=generator,
                device=device
            )
            gen_images_tensor = torch.stack([transforms.ToTensor()(img) for img in gen_images])
            all_images.append(gen_images_tensor)
    else:
        # generate 5000 images
        batch_size = 50
        for _ in tqdm(range(0, 5000, batch_size), desc="Generating images"):
            gen_images = pipeline(
                batch_size=batch_size,
                num_inference_steps=args.num_inference_steps,
                generator=generator,
                device=device
            )
            gen_images_tensor = torch.stack([transforms.ToTensor()(img) for img in gen_images])
            all_images.append(gen_images_tensor)
    

    all_images = torch.cat(all_images, dim=0)
    logger.info(f"Generated {all_images.shape[0]} images")

    if args.use_cfg and args.ckpt:
        images_per_class = 50
        best_dir = os.path.join(os.path.dirname(args.ckpt), 'best_images')
        os.makedirs(best_dir, exist_ok=True)
        
        # Select best 1 image per class (returns tensor of size [10, C, H, W])
        best_class_images = select_best_per_class_centroid(all_images, args.num_classes, images_per_class, device)
        
        # Save as 2 rows x 5 columns
        save_path = os.path.join(best_dir, 'best_class_centroid.png')
        save_image(best_class_images, save_path, nrow=5, normalize=False) # nrow=5 ensures 2 rows for 10 images
        
        logger.info(f"Saved best centroid images grid to {save_path}")

    if args.ckpt:
        save_dir = os.path.join(os.path.dirname(args.ckpt), 'sample_images')
        os.makedirs(save_dir, exist_ok=True)
    
        logger.info("Saving sample images for visualization")
    
        # Save 100 random samples
        num_samples = min(100, all_images.shape[0])
        sample_indices = torch.randperm(all_images.shape[0])[:num_samples]
    
        for i, idx in enumerate(sample_indices):
            img_pil = transforms.ToPILImage()(all_images[idx])
            img_pil.save(os.path.join(save_dir, f'generated_sample_{i:03d}.png'))
    
        grid_samples = all_images[sample_indices[:81]]  # 8x8 grid
        save_image(grid_samples, os.path.join(save_dir, 'preview_grid.png'), nrow=9, normalize=False)
    
    
    # TODO: load validation images as reference batch
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])

    # Load validation dataset
    if hasattr(args, 'data_valid') and args.data_valid:
        val_dataset = datasets.ImageFolder(args.data_valid, transform=transform)
    else:
        # Fallback to using part of training data if validation not specified
        val_dataset = datasets.ImageFolder(args.data_dir, transform=transform)
    
    # Sample validation images (match number of generated images)
    num_val_images = min(len(val_dataset), all_images.shape[0])
    val_indices = torch.randperm(len(val_dataset))[:num_val_images]
    val_images = torch.stack([val_dataset[i][0] for i in val_indices])
    logger.info(f"Loaded {val_images.shape[0]} validation images")

    # TODO: using torchmetrics for evaluation, check the documents of torchmetrics
    import torchmetrics 
    
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.inception import InceptionScore
    # TODO: compute FID and IS
    logger.info("Computing FID and IS scores")
    
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    inception_score = InceptionScore(normalize=True).to(device)
    
    gen_images_uint8 = (all_images * 255).clamp(0, 255).to(torch.uint8)
    val_images_uint8 = (val_images * 255).clamp(0, 255).to(torch.uint8)
    
    batch_size_eval = 32
    for i in range(0, val_images_uint8.shape[0], batch_size_eval):
        batch = val_images_uint8[i:i+batch_size_eval].to(device)
        fid.update(batch, real=True)
    
    for i in range(0, gen_images_uint8.shape[0], batch_size_eval):
        batch = gen_images_uint8[i:i+batch_size_eval].to(device)
        fid.update(batch, real=False)
        inception_score.update(batch)
    
    # Compute final scores
    fid_score = fid.compute()
    is_mean, is_std = inception_score.compute()

    logger.info(f"FID Score:        {fid_score.item():.2f}")
    logger.info(f"Inception Score:  {is_mean.item():.2f} ± {is_std.item():.2f}")

    results = {
        'FID': fid_score.item(),
        'IS_mean': is_mean.item(),
        'IS_std': is_std.item(),
        'num_generated': all_images.shape[0],
        'num_real': val_images.shape[0],
    }

    if args.ckpt:
        results_path = os.path.join(os.path.dirname(args.ckpt), 'evaluation_results.txt')
        with open(results_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("EVALUATION RESULTS\n")
            f.write("="*60 + "\n")
            f.write(f"FID Score:        {fid_score.item():.2f}\n")
            f.write(f"Inception Score:  {is_mean.item():.2f} ± {is_std.item():.2f}\n")
            f.write(f"Generated images: {all_images.shape[0]}\n")
            f.write(f"Real images:      {val_images.shape[0]}\n")
            f.write("="*60 + "\n")
        logger.info(f"Results saved to {results_path}")
    


if __name__ == '__main__':
    main()