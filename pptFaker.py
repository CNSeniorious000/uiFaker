from faker import *

# 120 for FHD; 160 for QHD; 240 for UHD
block_size = 240
screenx, screeny = 16 * block_size, 9 * block_size


def get_anchor(*numbers):
    if len(numbers) == 0:
        return block_size * numbers[0]
    else:
        return (block_size * n for n in numbers)
