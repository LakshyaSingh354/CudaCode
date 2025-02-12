from typing import List
from concurrent.futures import ThreadPoolExecutor
import time

class Thread:
    def __init__(self, thread_id):
        self.thread_id = thread_id
        self.status = "idle"
        self.registers = [0] * 32

    def run(self, kernel, block_id, warp_id, *args):
        start_time = time.time()
        self.status = "running"
        print(f"Thread {self.thread_id} in Warp {warp_id} of Block {block_id} is running...")
        kernel(self.thread_id, warp_id, block_id, start_time, *args)
        self.status = "done"
        print(f"Thread {self.thread_id} in Warp {warp_id} of Block {block_id} is done.")



class Warp:
    warp_id : int
    threads : List[Thread]

    def __init__(self, warp_id):
        self.warp_id = warp_id
        self.threads = [Thread(i) for i in range(32)]


    def run(self, kernel, block_id, *args):
        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = [executor.submit(thread.run, block_id, self.warp_id, kernel, *args) for thread in self.threads]
            for future in futures:
                future.result()

class Block:
    block_id : int
    warps : List[Warp]

    def __init__(self, block_id):
        self.block_id = block_id
        self.warps = [Warp(warp_id=i) for i in range(4)]

    def run(self, kernel, *args):
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(warp.run, self.block_id, kernel, *args) for warp in self.warps]
            for future in futures:
                future.result()

class Grid:
    blocks : List[Block]

    def __init__(self):
        self.blocks = [Block(block_id=i) for i in range(4)]

    def run(self, kernel, *args):
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(block.run, kernel, *args) for block in self.blocks]
            for future in futures:
                future.result()



def simple_kernel(thread_id, warp_id, block_id, start_time):
    print(f"Thread {thread_id} in Warp {warp_id} of Block {block_id} started at {start_time:.7f} seconds")


grid = Grid()
grid.run(simple_kernel)