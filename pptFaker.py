try:
    from itertools import pairwise
except ImportError:  # to support python3.9
    def pairwise(iterable):
        from itertools import tee
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)
import numpy as np, pygame as pg, random, cv2
from pygame.gfxdraw import filled_circle
from contextlib import suppress
from collections import deque
from enum import IntEnum, auto
from faker import *

# 80 for HD; 120 for FHD; 160 for QHD; 240 for UHD
block_size = 240
block_shape = block_size, block_size
screenx, screeny = 16 * block_size, 9 * block_size


def parse(*numbers) -> int or list[int]:
    if len(numbers) == 1:
        return round(block_size * numbers[0])
    else:
        return [round(block_size * n) for n in numbers]


class Layer(AbstractLayer):
    def __init__(self, x, y, w, h):
        self.anchor = parse(x, y)
        self.size = parse(w, h)
        self.rect = pg.Rect(self.anchor, self.size)


class LeftTopCircles(Layer):
    def __init__(self, x, y, length, gap, speed):
        super().__init__(x, y, length, length)
        self.length = parse(length)
        self.gap = parse(gap)
        self.speed = parse(speed)
        self.buffer = pg.Surface(self.size, depth=24)
        self.colors = [
            (240, 240, 240),
            (230, 230, 230)
        ]
        self.circles: deque[list[int, int]] = deque([[0, 0]])

    def draw_circle(self, r, color_num):
        filled_circle(self.buffer, 0, 0, r, self.colors[color_num])
        # pg.draw.circle(self.buffer, self.colors[color_num], (0, 0), r, 0, 0, 0, 0, 1)

    def get_different_color(self, n):
        while (result := random.randrange(len(self.colors))) == n:
            pass
        return result

    def draw(self):
        self.buffer.blit(self.screen, (0, 0), self.rect)  # draw background
        circles = self.circles

        # update
        for circle in circles:
            circle[0] += self.speed

        # render
        for radius, color in reversed(circles):
            self.draw_circle(radius, color)

        # update
        smallest = circles[0]
        if smallest[0] > self.gap:
            circles.appendleft([smallest[0] - self.gap, self.get_different_color(smallest[1])])
        with suppress(IndexError):
            second_biggest = circles[-2]
            if second_biggest[0] > self.gap + self.length * 2 ** 0.5:
                circles.pop()

        return self.screen.blit(self.buffer, self.anchor)


class ReversedLayer(AbstractLayer):
    def __init__(self, x, y, another):
        self.anchor = parse(x, y)
        self.bound: Layer = another

    @cached_property
    def rect(self):
        return pg.Rect(self.anchor, self.bound.rect.size)

    @property
    def buffer(self):
        return pg.transform.rotate(self.bound.buffer, 180)


class AcrylicImage(Layer):
    def __init__(self, image_name, x, y, w, h, luma=10, alpha=80):
        super().__init__(x, y, w, h)
        self.image = Asset(f"{image_name}_{block_size}", self.size, flags=pg.SRCALPHA)
        self.luma = luma, luma, luma
        self.alpha = pg.Surface(self.size, depth=24)
        self.alpha.fill((255 - luma, 255 - luma, 255 - luma))
        self.alpha.set_alpha(alpha)

    @property
    def shaded(self):
        layer: RandomFlipper
        return [
            (layer.buffer, rect := layer.rect.clip(self.rect), rect.move(-layer.x, -layer.y))
            for layer in Slide.current.layers
            if layer.rect.colliderect(self.rect) and layer is not self and layer.optimized
        ]

    def draw(self):
        rect = self.rect
        self.screen.blits(self.shaded)
        with timer("blurring"):
            x, y = self.anchor
            w, h = self.size
            area = pg.surfarray.pixels3d(self.screen)[x:x + w, y:y + h]
            area[:] = cv2.GaussianBlur(
                cv2.blur(cv2.blur(area, (47, 57)), (57, 47)),
                (0, 0), sigmaX=7, sigmaY=7
            )
            del area
        self.screen.blit(self.alpha, rect)
        self.screen.fill(self.luma, rect, pg.BLEND_ADD)
        self.screen.blit(self.image, rect)
        return rect


class RandomFlipper(AbstractLayer):
    instance: "RandomFlipper" = None
    semaphore = 7
    duration_hint = 40

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
        self.motion = RandomFlipper.Motion.idle
        self.chromatic = 0
        self.name_last = f"{random.choice(surface_names)}0"

        self.pair = None
        self.optimized = NotImplemented

    @property
    def dirty(self):
        if self.motion:
            try:
                self.pair = self.next_pair
                self.optimized = False
                return True
            except StopIteration:
                self.__delattr__("pos_it")  # reset iterator
                if self.name_last.endswith("2"):
                    RandomFlipper.semaphore += 1
                self.name_last = self.name_next  # set last to next
                self.__delattr__("name_next")  # reset next
                self.motion = RandomFlipper.Motion.idle
                self.buffer.blit(Asset.load(self.name_last, block_shape), (0, 0))
                self.optimized = True
                return False
        else:
            if random.randrange(60 * 4) == 0:  # 进入working状态
                self.motion = random.choice(RandomFlipper.motions)
                if RandomFlipper.semaphore:
                    self.chromatic = 2
                    RandomFlipper.semaphore -= 1
                else:
                    self.chromatic = random.choice((0, 0, 0, 1))

            self.optimized = True
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
        if abs(x_from - x_to) <= 1:
            self.render_at(name_last, name_next, x_from, direction)
            return self.buffer.copy()

        image = np.zeros(shape := (block_size, block_size, 3), np.float_)
        for x in range(x_from, x_to, x_from < x_to or -1):
            self.render_at(name_last, name_next, x, direction)
            image += np.frombuffer(
                self.buffer.get_buffer(), np.uint8
            ).reshape(shape)
        return pg.image.frombuffer((image / abs(x_from - x_to)).astype(np.uint8), block_shape, "BGR")

    @timer("render_at")
    def render_at(self, name_last, name_next, shift, direction):
        self.shadow.set_alpha(int(self.faker.at(64, 0, (shift / block_size) ** 2.5)))

        if direction is RandomFlipper.Motion.right_to_left:  # baseline
            last_anchor = 0, 0
            last_rect = (block_size - shift) // 2, 0, shift, block_size
            next_anchor = shift, 0
            next_rect = 0, 0, block_size - shift, block_size

        elif direction is RandomFlipper.Motion.left_to_right:
            last_anchor = block_size - shift, 0
            last_rect = (block_size - shift) // 2, 0, shift, block_size
            next_anchor = 0, 0
            next_rect = shift, 0, block_size - shift, block_size

        elif direction is RandomFlipper.Motion.button_to_top:
            last_anchor = 0, 0
            last_rect = 0, (block_size - shift) // 2, block_size, shift
            next_anchor = 0, shift
            next_rect = 0, 0, block_size, block_size - shift

        elif direction is RandomFlipper.Motion.top_to_button:
            last_anchor = 0, block_size - shift
            last_rect = 0, (block_size - shift) // 2, block_size, shift
            next_anchor = 0, 0
            next_rect = 0, shift, block_size, block_size - shift

        else:
            raise ValueError(direction)

        self.buffer.blits(((Asset.load(name_last, block_shape), last_anchor, last_rect),
                           (Asset.load(name_next, block_shape), next_anchor, next_rect),
                           (self.shadow, last_anchor, last_rect)), False)

    @cached_property
    def buffer(self):
        surface = pg.Surface(block_shape, depth=24)
        surface.blit(self.screen, (0, 0), self.rect)
        return surface

    @cached_property
    def shadow(self):
        return pg.Surface(block_shape, depth=24)

    def __repr__(self):
        return f"<RandomFlipper at ({self.x}, {self.y})>"

    __str__ = __repr__

    def full_cache(self):
        from alive_progress import alive_it
        lth = len(self.pos_map) - 1
        for motion in RandomFlipper.motions:
            for i, j in alive_it(pairwise(self.pos_map), total=lth, title=motion.name):
                for m in "012":
                    for n in "012":
                        for name_last in self.names:
                            for name_next in self.names:
                                if (name_1 := name_last + m) != (name_2 := name_next + n):
                                    with timer("get_buffer_in"):
                                        self.get_buffer_in(name_1, name_2, i, j, motion)

    @staticmethod
    def do_full_cache(names):
        Faker(*block_shape)
        block = RandomFlipper(names, 0, 0)
        RandomFlipper.instance = block
        block.full_cache()

    @cached_property
    def rect(self) -> pg.Rect:
        return pg.Rect(self.anchor, block_shape)


class Slide:
    current: "Slide" = None

    def __init__(self, last_: "Slide" = None, next_: "Slide" = None):
        Slide.current = self
        self.last_ = last_
        self.next_ = next_
        self.layers: list[AbstractLayer] = []

    def render(self):
        pg.display.update([layer.draw() for layer in self.layers if layer.dirty])

    @cached_property
    def faker(self):
        return Faker.instance

    @cached_property
    def screen(self):
        return self.faker.screen


class PowerPointFaker(Faker):
    def __init__(self, slides: list[Slide] = None, title="", flags=0b0):
        super().__init__(screenx, screeny, title, flags)
        self.slides = slides or []
        self.buffer = self.screen

    @cached_property
    def slide(self):
        return next(iter(self.slides))

    def mainloop(self):
        while True:
            with timer("render"):
                self.slide.render()
            if self.refresh() == -1:
                return
