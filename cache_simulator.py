class Cache:
    def __init__(self, name, size, block_size, associativity, policy='LRU', write_policy='write-back'):
        self.name = name
        self.size = size  # Bytes
        self.block_size = block_size  # Bytes per block
        self.associativity = associativity
        self.sets = size // (block_size * associativity)
        self.policy = policy
        self.write_policy = write_policy
        self.stats = {'hits': 0, 'misses': 0, 'writebacks': 0}
        
        # Initialize cache: list of sets, each set is a list of blocks
        self.cache = [
            [{'valid': False, 'tag': -1, 'lru_counter': 0, 'dirty': False}
             for _ in range(associativity)]
            for _ in range(self.sets)
        ]
        self.time = 0

    def access(self, address, is_write=False):
        set_index = (address // self.block_size) % self.sets
        tag = (address // self.block_size) // self.sets
        self.time += 1

        # Check for hit
        for block in self.cache[set_index]:
            if block['valid'] and block['tag'] == tag:
                self.stats['hits'] += 1
                block['lru_counter'] = self.time
                if is_write and self.write_policy == 'write-back':
                    block['dirty'] = True
                return True  # Hit

        # Miss: handle replacement
        self.stats['misses'] += 1
        lru_block = min(self.cache[set_index], key=lambda x: x['lru_counter'])
        
        # Writeback if dirty
        if lru_block['dirty'] and self.write_policy == 'write-back':
            self.stats['writebacks'] += 1
        
        # Replace block
        lru_block['valid'] = True
        lru_block['tag'] = tag
        lru_block['lru_counter'] = self.time
        lru_block['dirty'] = is_write if self.write_policy == 'write-back' else False
        return False  # Miss

class GPUCacheHierarchy:
    def __init__(self):
        self.l1 = Cache('L1', 16384, 64, 8)  # 16KB L1, 64B blocks, 8-way
        self.l2 = Cache('L2', 1048576, 64, 16)  # 1MB L2, 64B blocks, 16-way

    def access_memory(self, address, is_write=False):
        # Access L1 first
        hit = self.l1.access(address, is_write)
        if not hit:
            # On L1 miss, access L2
            hit = self.l2.access(address, is_write)
        return hit
    

def simulate_conv_layer(cache_system, C_in, C_out, H, W, K, stride=1, padding=0):
    # Simulate memory accesses for a convolution layer
    for c_out in range(C_out):  # Output channels
        for c_in in range(C_in):  # Input channels
            for y in range(H):  # Output height
                for x in range(W):  # Output width
                    # Input window coordinates
                    y_start = y * stride - padding
                    x_start = x * stride - padding
                    for ky in range(K):  # Kernel height
                        for kx in range(K):  # Kernel width
                            y_in = y_start + ky
                            x_in = x_start + kx
                            if 0 <= y_in < H and 0 <= x_in < W:
                                # Calculate global memory address for input pixel
                                # Assumes NCHW layout: input[c_in][y_in][x_in]
                                input_addr = c_in * (H * W) + y_in * W + x_in
                                cache_system.access_memory(input_addr)
                    
                    # Simulate kernel weight access (c_out, c_in, ky, kx)
                    kernel_addr = c_out * (C_in * K * K) + c_in * (K * K) + ky * K + kx
                    cache_system.access_memory(kernel_addr)

                    # Simulate output write (c_out, y, x)
                    output_addr = c_out * (H * W) + y * W + x
                    cache_system.access_memory(output_addr, is_write=True)

# Initialize cache hierarchy
gpu_cache = GPUCacheHierarchy()

# Simulate a Conv2D layer: 3 input channels, 64 output channels, 256x256 input, 3x3 kernel
simulate_conv_layer(gpu_cache, C_in=3, C_out=64, H=256, W=256, K=3, stride=1, padding=1)

# Print results
print("L1 Cache Stats:")
print(f"Hit Rate: {gpu_cache.l1.stats['hits'] / (gpu_cache.l1.stats['hits'] + gpu_cache.l1.stats['misses']):.2f}")
print("L2 Cache Stats:")
print(f"Hit Rate: {gpu_cache.l2.stats['hits'] / (gpu_cache.l2.stats['hits'] + gpu_cache.l2.stats['misses']):.2f}")


import matplotlib.pyplot as plt

def plot_cache_stats(cache):
    labels = ['Hits', 'Misses']
    values = [cache.stats['hits'], cache.stats['misses']]
    plt.bar(labels, values)
    plt.title(f"{cache.name} Cache Performance")
    plt.show()

plot_cache_stats(gpu_cache.l1)
plot_cache_stats(gpu_cache.l2)