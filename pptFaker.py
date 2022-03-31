from enum import IntEnum, auto
from itertools import pairwise
import numpy as np, random

from faker import *

# 120 for FHD; 160 for QHD; 240 for UHD
block_size = 240
block_shape = block_size, block_size
screenx, screeny = 16 * block_size, 9 * block_size


def parse_grid_coordination(*numbers):
    if len(numbers) == 0:
        return block_size * numbers[0]
    else:
        return (block_size * n for n in numbers)


class AbstractSlide:
    def __init__(self, last_: "AbstractSlide" = None, next_: "AbstractSlide" = None):
        self.last_ = last_
        self.next_ = next_


class Motion(IntEnum):
    idle = 0
    right_to_left = auto()
    left_to_right = auto()
    top_to_button = auto()
    button_to_top = auto()


class RandomFlipper(AbstractLayer):
    semaphore = 4
    duration = 40

    @cached_property
    def pos_map(self):
        """from rightmost to leftmost"""
        return [round(self.faker.at(block_size, 0, i)) for i in np.linspace(0, 1, self.duration)]

    @property
    def next_pair(self):
        return next(self.pos_it)

    @cached_property
    def pos_it(self):
        return pairwise(self.pos_map)

    @cached_property
    def name_next(self):
        while (name_next := random.choice(self.names)) == self.name_last:
            pass  # retry
        return name_next

    def __init__(self, surface_names: list[str], grid_x, grid_y):
        self.anchor = self.x, self.y = parse_grid_coordination(grid_x, grid_y)

        self.names = surface_names
        self.motion = Motion.idle
        self.chromatic = False
        self.name_last = random.choice(surface_names)

        self.pair = None

    @property
    def dirty(self):
        if self.motion:
            try:
                self.pair = self.next_pair
                return True
            except StopIteration:
                self.__delattr__("pos_it")  # reset iterator
                self.name_last = self.name_next  # set last to next
                self.__delattr__("name_next")  # reset next
                return False
        else:
            if random.randrange(60 * 3) == 0:  # 进入working状态
                self.motion = Motion(random.randint(1, 4))
                if self.semaphore:
                    self.chromatic = True
                    self.semaphore -= 1
                else:
                    self.chromatic = False

                return True

            return False

    def draw(self) -> pg.Rect:
        return self.screen.blit(
            self.get_buffer_in(self.name_last, self.name_next, *self.pair, self.motion), self.anchor
        )

    @staticmethod
    @cache
    @surfcache(block_shape)
    def get_buffer_in(name_last, name_next, x_from, x_to, direction):
        self: RandomFlipper = Faker.instance
        assert x_from != x_to

        image = np.zeros(shape := (block_size, block_size, 3), np.float_)
        for x in range(x_from, x_to, x_from < x_to or -1):
            image += np.frombuffer(
                self.get_buffer_at(name_last, name_next, x, direction), np.uint8
            ).reshape(shape)
        return pg.image.frombuffer((image / abs(x_from - x_to)).astype(np.uint8), block_shape, "BGR")

    @staticmethod
    @surfcache(block_shape)
    def get_buffer_at(name_last, name_next, shift, direction):
        self: RandomFlipper = Faker.instance
        self.shadow.set_alpha(int(self.faker.at(64, 0, (shift / block_size) ** 2.5)))

        if direction is Motion.right_to_left:  # baseline
            last_anchor = self.anchor
            last_rect = (block_size - shift) // 2, 0, shift, block_size
            next_anchor = self.x + shift, self.y
            next_rect = 0, 0, block_size - shift, block_size

        elif direction is Motion.left_to_right:
            last_anchor = self.x + block_size - shift, self.y
            last_rect = (block_size - shift) // 2, 0, shift, block_size
            next_anchor = self.anchor
            next_rect = shift, 0, block_size - shift, block_size

        elif direction is Motion.button_to_top:
            last_anchor = self.anchor
            last_rect = 0, (block_size - shift) // 2, block_size, shift
            next_anchor = self.x, self.y + shift
            next_rect = 0, 0, block_size, block_size - shift

        elif direction is Motion.top_to_button:
            last_anchor = self.x, self.y + block_size - shift
            last_rect = 0, (block_size - shift) // 2, block_size, shift
            next_anchor = self.anchor
            next_rect = 0, shift, block_size, block_size - shift

        else:
            raise ValueError(direction)

        self.buffer.blits(((Asset(name_last), last_anchor, last_rect),
                           (Asset(name_next), next_anchor, next_rect),
                           (self.shadow, last_anchor, last_rect)), False)

        return self.buffer.copy()

    @cached_property
    def buffer(self):
        return pg.Surface(block_shape, pg.HWSURFACE, 24)

    @cached_property
    def shadow(self):
        return pg.Surface(block_shape, pg.HWSURFACE, 24)
