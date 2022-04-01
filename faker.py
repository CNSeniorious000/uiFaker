from functools import cache, lru_cache, wraps, cached_property
import numpy as np, pygame as pg, ctypes, cv2, imageio
from blosc import compress, decompress
from contextlib import contextmanager
from time import perf_counter
from diskcache import Index

ctypes.windll.user32.SetProcessDPIAware(2)
model_fluent = [0, 1 / 8, 1 / 2, 3 / 4, 7 / 8, 15 / 16, 31 / 32, 63 / 64, 127 / 128, 1]
null = lambda x: x
time_map = {}


@contextmanager
def timer(key):
    past, count = time_map.get(key, (0., 0))
    t = perf_counter()
    yield
    time_map[key] = past + perf_counter() - t, count + 1


def show_time_map():
    from rich import print
    for key, val in time_map.items():
        print(f"[bright_yellow]{key:>20}"
              f"[bright_white] : "
              f"[bright_green]{60 * val[0] / val[1]}")


class Asset(pg.Surface):
    def __init__(self, name, size=None, interpolation=None, flags=0b0):
        self.name = name
        filepath = f"assets/{name}.png"
        if size is None:
            surface = pg.image.load(filepath)
            super().__init__(surface.get_size(), pg.HWSURFACE, surface.get_bitsize())
            self.get_buffer().write(surface.get_buffer().raw)
        else:
            image = imageio.imread(filepath)
            image[..., (0, 1, 2)] = image[..., (2, 1, 0)]

            w_from, h_from = image.shape[:2]
            w_to, h_to = size

            if w_to and not h_to:
                h_to = h_from * w_to // w_from
            if h_to and not w_to:
                w_to = w_from * h_to // h_from

            if w_from == w_to and h_from == h_to:
                raw = image
            else:
                w_scale = w_to / w_from
                h_scale = h_to / h_from
                max_scale = max(w_scale, h_scale)
                min_scale = min(w_scale, h_scale)

                if interpolation is None:
                    if max_scale <= 0.5:  # super-sampling
                        interpolation = cv2.INTER_AREA
                    elif max_scale <= 1:  # normal downscale
                        interpolation = cv2.INTER_LINEAR
                    elif w_scale == h_scale:
                        if min_scale >= 2:  # pixel art
                            interpolation = cv2.INTER_NEAREST
                        else:  # photography
                            interpolation = cv2.INTER_LANCZOS4
                    else:  # gradient background
                        interpolation = cv2.INTER_CUBIC

                raw = cv2.resize(image, size, interpolation=interpolation)

            super().__init__(size, pg.HWSURFACE | flags, 8 * image.size // w_from // h_from)
            self.get_buffer().write(raw)

    @classmethod
    @cache
    def load(cls, name):
        return cls(name)

    def __eq__(self, other):
        return isinstance(other, Asset) and self.name == other.name

    def __hash__(self):
        return hash(self.name)


def surfcache(size, pixel_format="BGR"):
    """cache a function that returns a surface"""

    def decorator(func):
        memo = Index(f"cache/{func.__qualname__}@{size}")  # ClassName.method_name

        @wraps(func)
        def wrapped(*args) -> pg.Surface:
            try:
                data = decompress(memo[args])
                return pg.image.frombuffer(data, size, pixel_format)
            except KeyError:
                surface: pg.Surface = func(*args)
                data = surface.get_buffer().raw
                memo[args] = compress(data, 1, 9, cname="lz4hc")
                return surface

        return wrapped

    return decorator


class Curve:
    f = staticmethod(null)

    @staticmethod
    def show(x, y):
        from matplotlib import pyplot as plt
        plt.scatter(x, y, s=10)
        plt.plot(x, y, c="red")
        return plt.show()

    def show_map(self, start, end, n):
        return Curve.show(x := np.linspace(0, 1, n), [self.at(start, end, ratio) for ratio in x])

    def show_f(self, n=32):
        return Curve.show_map(self, 0, 1, n)

    def use_model(self, model, kind="quadratic"):
        from scipy.interpolate import interp1d
        self.f = interp1d(np.linspace(0, 1, len(model)), model, kind)

    def at(self, start, end, ratio):
        if 0 < ratio < 1:
            return start + (end - start) * self.f(ratio)
        else:
            return start if ratio <= 0 else end


class AbstractLayer:
    surface: pg.Surface = NotImplemented
    anchor: tuple[int, int] = NotImplemented

    def draw(self, *args, **kwargs) -> pg.Rect:
        return self.screen.blit(self.surface, self.anchor)

    @cached_property
    def dirty(self):
        return True

    @cached_property
    def faker(self):
        return Faker.instance

    @cached_property
    def screen(self):
        return self.faker.buffer

    __str__ = __repr__ = classmethod(lambda cls: cls.__name__)

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __hash__(self):
        return hash(repr(self))


class Faker(Curve):
    instance: "Faker" = None

    def __init__(self, w, h):
        Faker.instance = self

        self.w = w
        self.h = h
        self.size = w, h

        self.clock = pg.time.Clock()
        self.buffer = pg.Surface((w, h), pg.HWSURFACE, 24)

        self.use_model(model_fluent)

    def debug_inspect(self, surface=None):
        from matplotlib import pyplot as plt
        plt.imshow(pg.surfarray.pixels3d(surface or self.buffer).swapaxes(0, 1))
        plt.show()

    @cached_property
    def screen(self):
        return pg.display.set_mode(self.size, pg.HWACCEL, 24, vsync=True)

    @timer("refresh")
    def refresh(self):
        self.screen.blit(self.buffer, (0, 0))
        pg.display.flip()
        # parse events
        for event in pg.event.get():
            if event.type == pg.QUIT:
                return -1
        pg.display.set_caption(f"FPS: {self.clock.get_fps():.2f}")
        self.clock.tick(60)
