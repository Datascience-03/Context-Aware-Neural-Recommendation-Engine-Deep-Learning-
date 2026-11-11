import pprint
import redis

# Connect to your local Redis instance
# Adjust host/port if you use non-default settings
r = redis.Redis(host="localhost", port=6379, decode_responses=True)

print("=" * 50)
print("1. MEMORY PROFILING (Global)")
print("=" * 50)
info_mem = r.info("memory")
print(f"Used Memory:            {info_mem.get('used_memory_human')}")
print(f"Peak Memory:            {info_mem.get('used_memory_peak_human')}")
print(
    f"Fragmentation Ratio:    {info_mem.get('mem_fragmentation_ratio')}"
)

print("\n" + "=" * 50)
print("2. MEMORY USAGE FOR A SINGLE VECTOR KEY")
print("=" * 50)
# Find keys matching your vector prefix
keys = r.keys("item:vec:*")

if keys:
    sample_key = keys[0]
    # In redis-py, execute raw commands via r.execute_command
    usage_bytes = r.execute_command("MEMORY", "USAGE", sample_key)
    print(f"Sample Key:  {sample_key}")
    print(
        f"Memory:      {usage_bytes} bytes (~{usage_bytes / 1024:.2f} KB)"
    )
else:
    # If using a different prefix, list any 5 keys to see what's in your DB:
    all_keys = r.keys("*")[:5]
    print(f"No keys matching 'item:vec:*'. Available sample keys: {all_keys}")
    if all_keys:
        usage_bytes = r.execute_command("MEMORY", "USAGE", all_keys[0])
        print(f"Memory for '{all_keys[0]}': {usage_bytes} bytes")

print("\n" + "=" * 50)
print("3. PERSISTENCE CONFIGURATION (Non-blocking)")
print("=" * 50)
# Configure AOF and RDB dynamically to satisfy the task requirement
r.config_set("appendonly", "yes")
r.config_set("appendfsync", "everysec")
r.config_set("no-appendfsync-on-rewrite", "yes")
r.config_set("save", "900 1 300 10 60 10000")

# Check persistence configuration
persistence_info = r.info("persistence")
print(f"RDB Last Save Status:  {persistence_info.get('rdb_last_bgsave_status')}")
print(f"AOF Enabled:           {persistence_info.get('aof_enabled')}")
print(f"AOF Rewrite in Prog:   {persistence_info.get('aof_rewrite_in_progress')}")
print(
    "\nPersistence successfully tuned (everysec background sync + RDB snapshots)!"
)