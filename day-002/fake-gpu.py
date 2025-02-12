from typing import List


class Thread:
    thread_id : int
    registers : List[int]
    status : str

class Warp:
    warp_id : int
    threads : List[Thread]

class Block:
    block_id : int
    warps : List[Warp]

class Grid:
    blocks : List[Block]

