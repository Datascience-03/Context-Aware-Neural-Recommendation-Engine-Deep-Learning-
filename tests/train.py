import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# Import your model and dataset classes
# from model import RecModel
# from dataset import RecDataset

def parse_args():
    parser = argparse.ArgumentParser(description="RecSys Training & Pipeline Validation")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--dry-run", action="store_true", help="Run 2-3 iterations to test pipeline and OOM")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Directory to save weights")
    return parser.parse_args()

def run_training():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Dataset & Loader
    # full_dataset = RecDataset(...)
    # if args.dry_run:
    #     print(">>> DRY-RUN MODE: Restricting dataset to 500 samples.")
    #     full_dataset = Subset(full_dataset, range(min(500, len(full_dataset))))
    #     args.epochs = 1
    
    # train_loader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=True)

    # 2. Model Initialization
    # model = RecModel(...).to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # criterion = nn.BCEWithLogitsLoss()

    print(">>> Starting Training...")
    # for epoch in range(args.epochs):
    #     model.train()
    #     for step, batch in enumerate(train_loader):
    #         # Forward pass & loss calculation
    #         ...
    #         if args.dry_run and step >= 5:
    #             print("Dry-run validation successful! No shape errors or CUDA OOM.")
    #             break

    # 3. Package Checkpoint for Member 5 (Evaluation)
    checkpoint_path = os.path.join(args.output_dir, "model_handoff.pt")
    handoff_package = {
        # "model_state_dict": model.state_dict(),
        # "config": {
        #     "num_users": ...,
        #     "num_items": ...,
        #     "embedding_dim": 64,
        # },
        "metadata": {
            "trained_epochs": args.epochs,
            "status": "ready_for_eval"
        }
    }
    # torch.save(handoff_package, checkpoint_path)
    print(f"\n[Success] Checkpoint packaged for Member 5 at: {checkpoint_path}")

if __name__ == "__main__":
    run_training()