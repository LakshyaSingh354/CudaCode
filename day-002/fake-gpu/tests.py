def test_parallel_execution(start_times):
    thread_start_times = start_times["threads"]
    warp_start_times = start_times["warps"]
    block_start_times = start_times["blocks"]

    # Check SIMT Execution (All 32 threads in a warp should start together)
    for (block_id, warp_id), warp_time in warp_start_times.items():
        thread_times = [thread_start_times[(block_id, warp_id, t)] for t in range(32)]
        max_diff = max(thread_times) - min(thread_times)
        assert max_diff < 0.02, f"Warp {warp_id} in Block {block_id} has thread timing divergence: {max_diff:.6f} sec"

    print("✅ SIMT execution: Threads in a warp start together")

    # Check Warp-Level Parallelism (All 4 warps in a block should start together)
    for block_id in range(4):
        warp_times = [warp_start_times[(block_id, w)] for w in range(4)]
        max_diff = max(warp_times) - min(warp_times)
        assert max_diff < 0.02, f"Block {block_id} has warp timing divergence: {max_diff:.6f} sec"

    print("✅ Warp-level parallelism: Warps in a block start together")

    # Check Block-Level Parallelism (All 4 blocks should start together)
    block_times = [block_start_times[b] for b in range(4)]
    max_diff = max(block_times) - min(block_times)
    assert max_diff < 0.006, f"Grid has block timing divergence: {max_diff:.6f} sec"

    print("✅ Block-level parallelism: Blocks start together")