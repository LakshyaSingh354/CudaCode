from concurrent.futures import ThreadPoolExecutor
import time
from tests import test_parallel_execution

class Thread:
    def __init__(self, thread_id):
        self.thread_id = thread_id
        self.status = "idle"
        self.registers = [0] * 32

    def run(self, kernel, block_id, warp_id, start_times, *args):
        start_time = time.time()
        self.status = "running"
        start_times["threads"][(block_id, warp_id, self.thread_id)] = start_time  # start time
        kernel()
        self.status = "done"


class Warp:
    def __init__(self, warp_id):
        self.warp_id = warp_id
        self.threads = [Thread(i) for i in range(32)]

    def run(self, kernel, block_id, start_times, *args):
        start_time = time.time()
        start_times["warps"][(block_id, self.warp_id)] = start_time  # warp start time
        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = [executor.submit(thread.run, kernel, block_id, self.warp_id, start_times, *args) for thread in self.threads]
            for future in futures:
                future.result()


class Block:
    def __init__(self, block_id):
        self.block_id = block_id
        self.warps = [Warp(warp_id=i) for i in range(4)]

    def run(self, kernel, start_times, *args):
        start_time = time.time()
        start_times["blocks"][self.block_id] = start_time  # block start time
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(warp.run, kernel, self.block_id, start_times, *args) for warp in self.warps]
            for future in futures:
                future.result()


class Grid:
    def __init__(self):
        self.blocks = [Block(block_id=i) for i in range(4)]

    def run(self, kernel, *args):
        start_times = {
            "blocks": {},
            "warps": {},
            "threads": {}
        }
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(block.run, kernel, start_times, *args) for block in self.blocks]
            for future in futures:
                future.result()

        test_parallel_execution(start_times)


def simple_kernel():
    pass

grid = Grid()
grid.run(simple_kernel)
