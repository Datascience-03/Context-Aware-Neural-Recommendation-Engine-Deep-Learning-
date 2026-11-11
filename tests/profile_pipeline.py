import time
import torch
from torch.utils.data import DataLoader

# Import your actual Dataset class here
# from dataset import RecommendationDataset 

def benchmark_dataloader(dataloader, num_batches=100):
    print("--- Starting DataLoader Profiling ---")
    start_time = time.perf_counter()
    total_samples = 0

    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        
        # Check shapes on first batch
        if i == 0:
            print("\n[Shape Verification]")
            if isinstance(batch, dict):
                for k, v in batch.items():
                    if hasattr(v, 'shape'):
                        print(f"  {k}: shape = {v.shape}, dtype = {v.dtype}")
            elif isinstance(batch, (list, tuple)):
                for idx, v in enumerate(batch):
                    if hasattr(v, 'shape'):
                        print(f"  tensor_{idx}: shape = {v.shape}, dtype = {v.dtype}")
            print("\nStreaming batches...")

        # Simulate batch sample count
        batch_size = batch['users'].size(0) if isinstance(batch, dict) else batch[0].size(0)
        total_samples += batch_size

    elapsed = time.perf_counter() - start_time
    fps = total_samples / elapsed
    print(f"\n--- Profiling Results ---")
    print(f"Processed {num_batches} batches ({total_samples} samples) in {elapsed:.2f} seconds.")
    print(f"Throughput: {fps:.2f} samples/sec ({num_batches / elapsed:.2f} batches/sec)")
    
    if torch.cuda.is_available():
        print(f"Max Allocated GPU Memory: {torch.cuda.max_memory_allocated() / 1e6:.2f} MB")
        print(f"Max Cached GPU Memory: {torch.cuda.max_memory_reserved() / 1e6:.2f} MB")

if __name__ == "__main__":
    # TODO: Replace with your actual dataset & loader
    # dataset = RecommendationDataset(...)
    # loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
    # benchmark_dataloader(loader)
    pass