from itertools import pairwise, permutations
from enum import IntEnum, auto
import numpy as np, random
from faker import *

# 120 for FHD; 180 for QHD; 240 for UHD
block_size = 240
block_shape = block_size, block_size
screenx, screeny = 16 * block_size, 9 * block_size


def parse(*numbers):
    if len(numbers) == 0:
        return block_size * numbers[0]
    else:
        return [block_size * n for n in numbers]


class Motion(IntEnum):
    idle = 0
    right_to_left = auto()
    left_to_right = auto()
    top_to_button = auto()
    button_to_top = auto()


motions = (Motion.right_to_left,
           Motion.left_to_right,
           Motion.top_to_button,
           Motion.button_to_top)


class RandomFlipper(AbstractLayer):
    instance: "RandomFlipper" = None
    semaphore = 7
    duration_hint = 40

    @cached_property
    def pos_map(self):
        """from rightmost to leftmost"""
        return [round(self.faker.at(block_size, 0, i)) for i in np.linspace(0, 1, self.duration_hint)]

    @property
    def next_pair(self):
        return next(self.pos_it)

    @cached_property
    def pos_it(self):
        return pairwise(self.pos_map)

    @cached_property
    def name_next(self):
        while (name_next := f"{random.choice(self.names)}{self.chromatic}") == self.name_last:
            pass  # retry
        return name_next

    def __init__(self, surface_names: list[str], grid_x, grid_y):
        self.anchor = self.x, self.y = parse(grid_x, grid_y)

        self.names = surface_names
        self.motion = Motion.idle
        self.chromatic = 0
        self.name_last = f"{random.choice(surface_names)}0"

        self.pair = None

    @property
    def dirty(self):
        if self.motion:
            try:
                self.pair = self.next_pair
                return True
            except StopIteration:
                self.__delattr__("pos_it")  # reset iterator
                if not self.name_last.endswith("0"):
                    RandomFlipper.semaphore += 1
                self.name_last = self.name_next  # set last to next
                self.__delattr__("name_next")  # reset next
                self.motion = Motion.idle
                return False
        else:
            if random.randrange(60 * 4) == 0:  # 进入working状态
                self.motion = random.choice(motions)
                if RandomFlipper.semaphore:
                    self.chromatic = random.choice((1, 2))
                    RandomFlipper.semaphore -= 1
                else:
                    self.chromatic = 0

            return False

    def draw(self) -> pg.Rect:
        RandomFlipper.instance = self
        return self.screen.blit(
            self.get_buffer_in(self.name_last, self.name_next, *self.pair, self.motion), self.anchor
        )

    @staticmethod
    @cache
    @surfcache(block_shape)
    def get_buffer_in(name_last, name_next, x_from, x_to, direction):
        self: RandomFlipper = RandomFlipper.instance
        if x_from == x_to:
            return self.get_buffer_at(name_last, name_next, x_to, direction)

        image = np.zeros(shape := (block_size, block_size, 3), np.float_)
        for x in range(x_from, x_to, x_from < x_to or -1):
            image += np.frombuffer(
                self.get_buffer_at(name_last, name_next, x, direction).get_buffer(), np.uint8
            ).reshape(shape)
        return pg.image.frombuffer((image / abs(x_from - x_to)).astype(np.uint8), block_shape, "BGR")

    def get_buffer_at(self, name_last, name_next, shift, direction):
        self.shadow.set_alpha(int(self.faker.at(64, 0, (shift / block_size) ** 2.5)))

        if direction is Motion.right_to_left:  # baseline
            last_anchor = 0, 0
            last_rect = (block_size - shift) // 2, 0, shift, block_size
            next_anchor = shift, 0
            next_rect = 0, 0, block_size - shift, block_size

        elif direction is Motion.left_to_right:
            last_anchor = block_size - shift, 0
            last_rect = (block_size - shift) // 2, 0, shift, block_size
            next_anchor = 0, 0
            next_rect = shift, 0, block_size - shift, block_size

        elif direction is Motion.button_to_top:
            last_anchor = 0, 0
            last_rect = 0, (block_size - shift) // 2, block_size, shift
            next_anchor = 0, shift
            next_rect = 0, 0, block_size, block_size - shift

        elif direction is Motion.top_to_button:
            last_anchor = 0, block_size - shift
            last_rect = 0, (block_size - shift) // 2, block_size, shift
            next_anchor = 0, 0
            next_rect = 0, shift, block_size, block_size - shift

        else:
            raise ValueError(direction)

        self.buffer.blits(((Asset.load(name_last, block_shape), last_anchor, last_rect),
                           (Asset.load(name_next, block_shape), next_anchor, next_rect),
                           (self.shadow, last_anchor, last_rect)), False)

        return self.buffer.copy()

    @cached_property
    def buffer(self):
        return pg.Surface(block_shape, depth=24)

    @cached_property
    def shadow(self):
        return pg.Surface(block_shape, depth=24)

    def __repr__(self):
        return f"<RandomFlipper at ({self.x}, {self.y})>"

    __str__ = __repr__

    def full_cache(self):
        from alive_progress import alive_it
        lth = len(self.pos_map) - 1
        for motion in motions:
            for i, j in alive_it(pairwise(self.pos_map), total=lth, title=motion.name):
                for name_last, name_next in permutations(self.names, 2):
                    for m, n in permutations("012", 2):
                        self.get_buffer_in(name_last + m, name_next + n, i, j, motion)

    @staticmethod
    def do_full_cache(names):
        Faker(*block_shape)
        block = RandomFlipper(names, 0, 0)
        RandomFlipper.instance = block
        block.full_cache()


class AbstractSlide:
    def __init__(self, last_: "AbstractSlide" = None, next_: "AbstractSlide" = None):
        self.last_ = last_
        self.next_ = next_
        self.layers: list[AbstractLayer] = []

    def render(self):
        pg.display.update([layer.draw() for layer in self.layers if layer.dirty])


class PowerPointFaker(Faker):
    def __init__(self, slides: list[AbstractSlide] = None, title=""):
        super().__init__(screenx, screeny, title)
        self.slides = slides or []
        self.buffer = self.screen

    @cached_property
    def slide(self):
        return next(iter(self.slides))

    def mainloop(self):
        while True:
            self.slide.render()
            if self.refresh() == -1:
                return
