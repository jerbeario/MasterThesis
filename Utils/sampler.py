import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, Dataset
import matplotlib.pyplot as plt

# Load the same data you're using in your model

 # Create a simple synthetic dataset with class imbalance
class SyntheticDataset(Dataset):
    def __init__(self, size=10000, imbalance_ratio=0.1):
        self.size = size
        # Create synthetic data with similar imbalance to wildfire data
        self.features = torch.randn(size, 1, 5, 5)  # Simple 5x5 patches
        self.labels = torch.zeros(size)
        
        # Set positive examples (1% positive cases)
        pos_indices = np.random.choice(size, int(size * imbalance_ratio), replace=False)
        self.labels[pos_indices] = 1.0
    
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Custom balanced sampler
class BalancedBatchSampler:
    def __init__(self, labels, batch_size):
        self.pos_indices = np.where(labels == 1)[0]
        self.neg_indices = np.where(labels == 0)[0]
        self.batch_size = batch_size
        
    def __iter__(self):
        np.random.shuffle(self.pos_indices)
        np.random.shuffle(self.neg_indices)
        
        pos_per_batch = self.batch_size // 2
        neg_per_batch = self.batch_size - pos_per_batch
        
        n_pos_batches = len(self.pos_indices) // pos_per_batch
        n_neg_batches = len(self.neg_indices) // neg_per_batch
        n_batches = min(n_pos_batches, n_neg_batches)
        
        for i in range(n_batches):
            pos_batch = self.pos_indices[i*pos_per_batch:(i+1)*pos_per_batch]
            neg_batch = self.neg_indices[i*neg_per_batch:(i+1)*neg_per_batch]
            batch = np.concatenate([pos_batch, neg_batch])
            np.random.shuffle(batch)
            yield batch.tolist()
            
    def __len__(self):
        return min(len(self.pos_indices) // (self.batch_size // 2), 
                    len(self.neg_indices) // (self.batch_size - self.batch_size // 2))
    

def run_benchmark(dataset_size=10000, batch_size=32, num_trials=5, num_workers=0):
    """Benchmark BalancedBatchSampler vs standard DataLoader"""
    

    # Create dataset
    dataset = SyntheticDataset(size=dataset_size)
    
    # Create balanced sampler
    balanced_sampler = BalancedBatchSampler(dataset.labels, batch_size)
    
    # Times for each approach
    balanced_times = []
    standard_times = []
    weighted_times = []
    
    print(f"Dataset size: {dataset_size}, Batch size: {batch_size}")
    print(f"Positive samples: {sum(dataset.labels)}, Negative samples: {dataset_size - sum(dataset.labels)}")
    
    # Run multiple trials
    for trial in range(num_trials):
        print(f"\nTrial {trial+1}/{num_trials}")
        
        # 1. Test with balanced batch sampler
        start_time = time.time()
        loader_balanced = DataLoader(dataset, batch_sampler=balanced_sampler, num_workers=num_workers)
        
        num_batches = 0
        for batch_idx, (inputs, targets) in enumerate(loader_balanced):
            num_batches += 1
            # Count positive samples in each batch as a sanity check
            if batch_idx == 0:
                pos_count = torch.sum(targets).item()
                print(f"  Balanced Sampler - First batch: {len(targets)} samples, {pos_count} positive ({pos_count/len(targets)*100:.1f}%)")
        
        balanced_time = time.time() - start_time
        balanced_times.append(balanced_time)
        print(f"  Balanced sampler: {balanced_time:.4f}s for {num_batches} batches")
        
        # 2. Test with standard random sampler
        start_time = time.time()
        loader_standard = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        
        num_batches = 0
        for batch_idx, (inputs, targets) in enumerate(loader_standard):
            num_batches += 1
            # Count positive samples in each batch as a sanity check
            if batch_idx == 0:
                pos_count = torch.sum(targets).item()
                print(f"  Standard Sampler - First batch: {len(targets)} samples, {pos_count} positive ({pos_count/len(targets)*100:.1f}%)")
        
        standard_time = time.time() - start_time
        standard_times.append(standard_time)
        print(f"  Standard sampler: {standard_time:.4f}s for {num_batches} batches")
        
        # 3. Test with weighted random sampler
        # Calculate sample weights
        class_counts = np.bincount(dataset.labels.numpy().astype(int))
        class_weights = 1.0 / class_counts
        sample_weights = np.array([class_weights[int(label)] for label in dataset.labels])
        
        # Create weighted sampler
        weighted_sampler = torch.utils.data.WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights).float(),
            num_samples=len(dataset),
            replacement=True
        )
        
        start_time = time.time()
        loader_weighted = DataLoader(dataset, batch_size=batch_size, sampler=weighted_sampler, num_workers=num_workers)
        
        num_batches = 0
        for batch_idx, (inputs, targets) in enumerate(loader_weighted):
            num_batches += 1
            # Count positive samples in each batch as a sanity check
            if batch_idx == 0:
                pos_count = torch.sum(targets).item()
                print(f"  Weighted Sampler - First batch: {len(targets)} samples, {pos_count} positive ({pos_count/len(targets)*100:.1f}%)")
        
        weighted_time = time.time() - start_time
        weighted_times.append(weighted_time)
        print(f"  Weighted sampler: {weighted_time:.4f}s for {num_batches} batches")
    
    # Calculate average times
    avg_balanced = sum(balanced_times) / len(balanced_times)
    avg_standard = sum(standard_times) / len(standard_times)
    avg_weighted = sum(weighted_times) / len(weighted_times)
    
    # Print summary results
    print("\nSummary Results:")
    print(f"  Average time with balanced sampler: {avg_balanced:.4f}s")
    print(f"  Average time with standard sampler: {avg_standard:.4f}s")
    print(f"  Average time with weighted sampler: {avg_weighted:.4f}s")
    print(f"  Balanced/Standard ratio: {avg_balanced/avg_standard:.2f}x")
    print(f"  Weighted/Standard ratio: {avg_weighted/avg_standard:.2f}x")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.bar(['Balanced Sampler', 'Standard Sampler', 'Weighted Sampler'], 
            [avg_balanced, avg_standard, avg_weighted])
    plt.ylabel('Average Time (seconds)')
    plt.title('DataLoader Iteration Time Comparison')
    
    # Add time values on top of bars
    for i, v in enumerate([avg_balanced, avg_standard, avg_weighted]):
        plt.text(i, v + 0.02, f"{v:.4f}s", ha='center')
    
    plt.savefig('dataloader_benchmark.png')
    plt.close()
    
    return {
        'balanced': avg_balanced,
        'standard': avg_standard,
        'weighted': avg_weighted,
        'balanced_ratio': avg_balanced/avg_standard,
        'weighted_ratio': avg_weighted/avg_standard
    }

if __name__ == "__main__":
    # Run with different dataset sizes
    print("Running small dataset benchmark")
    # small_results = run_benchmark(dataset_size=10000, batch_size=32, num_trials=3)
    

    print("\nRunning with multiple workers")

    results = run_benchmark(dataset_size=100000, batch_size=64, num_trials=3, num_workers=0)
    results = run_benchmark(dataset_size=100000, batch_size=64, num_trials=3, num_workers=1)
    results = run_benchmark(dataset_size=100000, batch_size=64, num_trials=3, num_workers=2)
    results = run_benchmark(dataset_size=100000, batch_size=64, num_trials=3, num_workers=3)
    results = run_benchmark(dataset_size=100000, batch_size=64, num_trials=3, num_workers=4)
