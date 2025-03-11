import numpy as np
import itertools

class Cache:
    def __init__(self, size, block_size, associativity):
        self.size = size          # in bytes
        self.block_size = block_size  # in bytes
        self.associativity = associativity
        self.sets = size // (block_size * associativity)
        self.cache = [[{'valid': False, 'tag': -1, 'lru_counter': 0} 
                      for _ in range(associativity)] 
                      for _ in range(self.sets)]
        self.hits = 0
        self.misses = 0
        self.time = 0

    def access(self, address):
        set_index = (address // self.block_size) % self.sets
        tag = (address // self.block_size) // self.sets
        self.time += 1

        # Check for hit
        for block in self.cache[set_index]:
            if block['valid'] and block['tag'] == tag:
                self.hits += 1
                block['lru_counter'] = self.time
                return

        # Handle miss
        self.misses += 1
        # Find LRU block
        lru_block = min(self.cache[set_index], key=lambda x: x['lru_counter'])
        lru_block['valid'] = True
        lru_block['tag'] = tag
        lru_block['lru_counter'] = self.time

class CacheHierarchy:
    def __init__(self):
        self.l1 = Cache(16384, 64, 8)  # 16KB L1, 64B blocks, 8-way
        self.l2 = Cache(1048576, 64, 16)  # 1MB L2, 64B blocks, 16-way

    def access(self, address):
        # Access L1
        l1_hit = any(block['valid'] and block['tag'] == (address//64) for block in 
                    self.l1.cache[(address//64) % (16384//(64*8))])
        
        if not l1_hit:
            # Access L2 on L1 miss
            self.l1.access(address)
            self.l2.access(address)
        else:
            self.l1.hits += 1

def generate_optimization_combinations():
    tiling_sizes = [16, 32, 64]
    data_layouts = ['NCHW', 'NHWC']
    use_fusion = [True, False]
    return list(itertools.product(tiling_sizes, data_layouts, use_fusion))

def simulate_optimized_conv(tiling_size, data_layout, use_fusion):
    cache_system = CacheHierarchy()
    H, W = 256, 256
    K = 3
    
    for y_tile in range(0, H, tiling_size):
        for x_tile in range(0, W, tiling_size):
            for y in range(y_tile, min(y_tile + tiling_size, H)):
                for x in range(x_tile, min(x_tile + tiling_size, W)):
                    # Generate addresses
                    if data_layout == 'NHWC':
                        addr = y * W * 3 + x * 3  # NHWC format
                    else:
                        addr = 3 * (y * W + x)    # NCHW format
                    
                    # Simulate read
                    cache_system.access(addr)
                    
                    # Simulate write if not fused
                    if not use_fusion:
                        cache_system.access(addr)

    # Calculate hit rates
    l1_hr = cache_system.l1.hits / (cache_system.l1.hits + cache_system.l1.misses)
    l2_hr = cache_system.l2.hits / (cache_system.l2.hits + cache_system.l2.misses)
    return l1_hr, l2_hr

def find_best_optimization():
    combinations = generate_optimization_combinations()
    best_l1 = -1
    best_config = None

    for config in combinations:
        tiling, layout, fusion = config
        l1_hr, l2_hr = simulate_optimized_conv(tiling, layout, fusion)
        print(f"Config: Tile={tiling}, Layout={layout}, Fusion={fusion} => L1={l1_hr:.2f}, L2={l2_hr:.2f}")
        if l1_hr > best_l1:
            best_l1 = l1_hr
            best_config = config

    print(f"\nBest Config: Tile={best_config[0]}, Layout={best_config[1]}, Fusion={best_config[2]}")
    print(f"L1 Hit Rate: {best_l1:.2f}")

if __name__ == "__main__":
    find_best_optimization()