import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from models import *
import argparse
from utils import *


def main(args):
    model = load_ptm(args.ptm_name, args.ptm_path)
    shadow_images = np.load(args.shadow_images)
    reference_embedding = generate_reference_embedding(model, args.reference_images)

    normalize = transforms.Normalize(mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250])

    opt_trigger = torch.tensor(
        np.random.uniform(low=-1, high=1, size=(224, 224, 3)),
        requires_grad=True,
        device='cuda'
    )
    optimizer = torch.optim.Adam([opt_trigger], lr=args.lr)

    for epoch in range(1, args.epochs+1):
        indices = np.random.choice(len(shadow_images), len(shadow_images), replace=False)
        pbar = tqdm(range(len(shadow_images) // args.batch_size))
        for i in pbar:
            images = []
            start = i * args.batch_size
            end = start + args.batch_size
            for idx in indices[start:end]:
                images.append(shadow_images[idx])
            images = [torch.tensor(image, dtype=torch.float, device='cuda') for image in images]

            trigger = args.norm * F.tanh(opt_trigger)
            poisoned_images = []
            for image in images:
                image[:,:,:] = torch.clamp(image[:,:,:] + trigger, 0, 255)
                poisoned_images.append(image)
            poisoned_images = [img.permute(2, 0, 1) for img in poisoned_images]
            poisoned_images = [img / 255. for img in poisoned_images]
            poisoned_images = torch.stack([normalize(img) for img in poisoned_images])
            
            poisoned_embeddings = model(poisoned_images)

            loss = - F.cosine_similarity(poisoned_embeddings, reference_embedding, dim=1).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_description(f"Epoch {epoch}/{args.epochs} Loss={loss.item():.4f}")
    
    trigger = args.norm * F.tanh(opt_trigger)
    np.save(args.output, trigger.detach().cpu().numpy())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trigger optimization')
    parser.add_argument('--ptm_name', default='resnet18', type=str)
    parser.add_argument('--ptm_path', default='', type=str)
    parser.add_argument('--shadow_images', default='', type=str)
    parser.add_argument('--reference_images', default='', type=str)
    parser.add_argument('--norm', default=10, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--output', default='', type=str)
    args = parser.parse_args()
    print_args(args)
    main(args)
