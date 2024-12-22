import torch
import argparse
from torch.utils.data import DataLoader
import torch.optim as optim
from taco_dataset import TACODataset
from model import get_model
import os

def collate_fn(batch):
    return tuple(zip(*batch))

def train(args):
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Dataset
    train_dataset = TACODataset(
        root_dir=args.data_dir,
        annotation_file=os.path.join(args.data_dir, f'annotations_{args.round}_train.json')
    )
    
    test_dataset = TACODataset(
        root_dir=args.data_dir,
        annotation_file=os.path.join(args.data_dir, f'annotations_{args.round}_test.json')
    )
    
    # DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=1, 
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    # Model
    if args.round > 0:
        prev_checkpoint_path = os.path.join(args.output_dir, f'model_round_{args.round-1}_final.pth')
        if os.path.exists(prev_checkpoint_path):
            print(f"Loading previous round model from {prev_checkpoint_path}")
            checkpoint = torch.load(prev_checkpoint_path)
            model = get_model(train_dataset.num_classes)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Successfully loaded model from round {args.round-1}")
        else:
            print(f"No previous round model found at {prev_checkpoint_path}, starting fresh")
            model = get_model(train_dataset.num_classes)
    else:
        print("Starting training from round 0")
        model = get_model(train_dataset.num_classes)
    
    model.to(device)
    
    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)
    
    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                  step_size=3, 
                                                  gamma=0.1)
    
    # Training loop
    for epoch in range(args.epochs):
        # Train for one epoch
        model.train()
        total_loss = 0
        
        for i, (images, targets) in enumerate(train_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            total_loss += losses.item()
            
            if i % 50 == 0:
                print(f'Epoch [{epoch+1}/{args.epochs}], Step [{i}/{len(train_loader)}], '
                      f'Loss: {losses.item():.4f}')
        
        # Update learning rate
        lr_scheduler.step()
        
        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = os.path.join(args.output_dir, 
                                         f'model_round_{args.round}_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss,
            }, checkpoint_path)
            print(f'Checkpoint saved to {checkpoint_path}')

    # Save final model for this round
    final_checkpoint_path = os.path.join(args.output_dir, f'model_round_{args.round}_final.pth')
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': total_loss,
    }, final_checkpoint_path)
    print(f'Final model for round {args.round} saved to {final_checkpoint_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Faster R-CNN on TACO dataset')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to TACO dataset directory')
    parser.add_argument('--round', type=int, required=True,
                        help='Dataset round number')
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Learning rate')
    parser.add_argument('--save_freq', type=int, default=1,
                        help='Frequency of saving checkpoints (epochs)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    train(args)