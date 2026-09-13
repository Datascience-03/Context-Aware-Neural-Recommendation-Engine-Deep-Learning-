def estimate_redis_ram(num_items: int, dim: int, key_name_len: int = 16):
    # FP32 = 4 bytes per dimension
    raw_vector_bytes = num_items * dim * 4
    
    # Redis overhead per key: dictEntry (~24-32B) + robj (~16B) + SDS strings + key name
    redis_overhead_per_key = 100 + key_name_len
    total_overhead_bytes = num_items * redis_overhead_per_key
    
    # Base memory
    base_ram_bytes = raw_vector_bytes + total_overhead_bytes
    base_ram_gb = base_ram_bytes / (1024 ** 3)
    
    # Production safety buffer (1.35x for jemalloc fragmentation & peak operations)
    recommended_ram_gb = base_ram_gb * 1.35
    
    print(f"--- Sizing for {num_items:,} vectors (Dimension: {dim}) ---")
    print(f"Raw Vector Data:     {raw_vector_bytes / (1024 ** 3):.2f} GB")
    print(f"Redis Key Overhead:  {total_overhead_bytes / (1024 ** 3):.2f} GB")
    print(f"Base Memory Needed:  {base_ram_gb:.2f} GB")
    print(f"Recommended RAM:     {recommended_ram_gb:.2f} GB (with 1.35x buffer)")

# Example: 1 Million vectors with 768 dimensions (standard BERT/all-mpnet size)
estimate_redis_ram(num_items=1_000_000, dim=768)