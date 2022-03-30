from functools import cache, lru_cache, wraps, cached_property
import numpy as np, pygame as pg, ctypes, cv2, imageio
from blosc import compress, decompress
from enum import IntEnum, auto
from diskcache import Index

ctypes.windll.user32.SetProcessDPIAware(2)

model_fluent = [0, 1 / 8, 1 / 2, 3 / 4, 7 / 8, 15 / 16, 31 / 32, 63 / 64, 127 / 128, 1]

null = lambda x: x


class Style(IntEnum):
    moving = auto()
    level = auto()


class Surface(pg.Surface):
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
        return isinstance(other, Surface) and self.name == other.name

    def __hash__(self):
        return hash(self.name)


def surfcache(size, threshold=None, pixel_format="BGR"):
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

        return lru_cache(maxsize=threshold)(wrapped)

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
    def faker(self):
        return Faker.instance

    @cached_property
    def screen(self):
        return self.faker.surface

    __str__ = __repr__ = classmethod(lambda cls: cls.__name__)

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __hash__(self):
        return hash(repr(self))


class StaticCover(AbstractLayer):
    anchor = 0, 0

    @cached_property
    def surface(self):
        return Surface("cover", self.faker.size, flags=pg.SRCALPHA)


class SolidShadow(AbstractLayer):
    @cached_property
    def surface(self):
        return pg.Surface(Faker.instance.size, pg.HWSURFACE, 8)

    def draw(self, x, alpha):
        self.surface.set_alpha(alpha)
        return self.screen.blit(self.surface, (x, 0))


class Page(AbstractLayer):
    instances = {}

    def __new__(cls, name):
        try:
            return Page.instances[name]
        except KeyError:
            return object.__new__(cls)

    def __init__(self, name):
        Page.instances[name] = self
        self.name = name

    def __str__(self):
        return f"Page(name={self.name})"

    __repr__ = __str__

    @cached_property
    def surface(self):
        return Surface(self.name, self.faker.size)

    def draw(self, x):
        return self.screen.blit(self.surface, (x, 0))

    def __hash__(self):
        return hash(self.name)


class Dock(AbstractLayer):
    faker: "AppFaker"

    def __init__(self, luma, alpha):
        self.color = luma, luma, luma
        self.alpha = alpha

    @cached_property
    def h(self):
        return self.faker.h_dock

    @cached_property
    def size(self):
        return self.faker.w, self.h

    @cached_property
    def y(self):
        return self.faker.h - self.h

    @cached_property
    def anchor(self):
        return 0, self.y

    @cached_property
    def rect(self):
        return pg.Rect(*self.anchor, *self.size)

    @cached_property
    def surface(self):
        surface = pg.Surface(self.size, pg.HWSURFACE, 8)
        surface.fill(255)
        surface.set_alpha(self.alpha)
        return surface

    def draw(self):
        global t1, t2
        from time import perf_counter as time
        self.screen.fill(self.color, self.rect, pg.BLEND_ADD)
        dirty_rect = self.screen.blit(self.surface, self.anchor)
        dock_area = pg.surfarray.pixels3d(self.screen)[:, self.y:]
        t = time()
        dock_area[:] = cv2.GaussianBlur(
            cv2.blur(cv2.blur(dock_area, (37, 77)), (77, 37)),
            (0, 0), sigmaX=7, sigmaY=17
        )
        t2 += time() - t
        return dirty_rect


class Faker(Curve):
    instance: "Faker" = None

    def __init__(self, w, h):
        Faker.instance = self

        self.w = w
        self.h = h
        self.size = w, h
        self.clock = pg.time.Clock()
        self.surface = pg.Surface((w, h), pg.HWSURFACE, 24)

        self.use_model(model_fluent)

    def debug_inspect(self, surface=None):
        from matplotlib import pyplot as plt
        plt.imshow(pg.surfarray.pixels3d(surface or self.surface).swapaxes(0, 1))
        plt.show()


t1 = 0
t2 = 0


class AppFaker(Faker):
    bar = staticmethod(lambda iterator, message: print(message) or iterator)

    def __init__(self, w=1440, h=2960, h_dock=247, duration=40):
        super().__init__(w, h)
        self.h_dock = h_dock
        self.duration = duration

        self.cover = StaticCover()
        self.shadow = SolidShadow()
        self.Pages = {}
        self.dock = Dock(27, 37)

        self.cached_render_motion_once = surfcache((w, h), w + w + 1)(self.cached_render_motion_once)
        self.cached_render_motion_multi = surfcache((w, h))(self.cached_render_motion_multi)

    def render_at(self, page_last, page_next, i, style, reverse):
        if reverse:
            page_last, page_next = page_next, page_last
            x = int(self.at(0, self.w, i))
        else:
            x = int(self.at(self.w, 0, i))

        # maybe once more
        self.surface.blit(self.cached_render_motion_once(page_last.name, page_next.name, x, style), (0, 0))
        self.render_static_once()

    def render_in(self, page_last, page_next, i, j, style, reverse):
        if reverse:
            page_last, page_next = page_next, page_last
            x = round(self.at(0, self.w, i))
            y = round(self.at(0, self.w, j))
        else:
            x = round(self.at(self.w, 0, i))
            y = round(self.at(self.w, 0, j))

        # maybe once more
        self.surface.blit(self.cached_render_motion_multi(page_last.name, page_next.name, x, y, style), (0, 0))
        self.render_static_once()

    def render_motion_once(self, page_last: Page, page_next: Page, x_next, style):
        assert isinstance(x_next, int), x_next
        x_last = x_next - self.w if style is Style.moving else (x_next - self.w) // 2
        page_last.draw(x_last)
        if style is Style.level:
            self.shadow.draw(x_last, int(self.at(64, 0, (x_next / self.w) ** 2.5)))
        page_next.draw(x_next)

    def cached_render_motion_once(self, name_last, name_next, x_next, style):
        self.render_motion_once(Page(name_last), Page(name_next), x_next, style)
        return self.surface.copy()

    def cached_render_motion_multi(self, name_last, name_next, x, y, style):
        assert x != y
        sample = range(x, y, 1 if x < y else -1)
        w, h = self.w, self.h
        image = np.zeros((h, w, 3), np.float_)
        use_alive()
        for x in self.bar(sample, f"sampling in {sample}"):
            image += np.frombuffer(self.cached_render_motion_once(
                name_last, name_next, x, style
            ).get_buffer(), np.uint8).reshape(h, w, 3)

        self.surface.blit(pg.image.frombuffer((image / len(sample)).astype(np.uint8), (w, h), "BGR"), (0, 0))
        return self.surface.copy()

    def render_static_once(self):
        self.dock.draw()
        self.cover.draw()


class AppPlayer(AppFaker):
    def refresh(self):
        self.screen.blit(self.surface, (0, 0))
        pg.display.flip()
        # parse events
        for event in pg.event.get():
            if event.type == pg.QUIT:
                return -1
        pg.display.set_caption(f"FPS: {self.clock.get_fps():.2f}")
        self.clock.tick(60)

    @cached_property
    def screen(self):
        return pg.display.set_mode(self.size, pg.HWACCEL, 24, vsync=True)

    def animate(self, page_from: Page, page_to: Page, style, reverse):
        if (name_from := page_from.name) == (name_to := page_to.name):
            return print(f"invalid changing from {name_from} to {name_to}")
        use_null()
        it = iter(self.bar(np.linspace(0, 1, self.duration), f"{name_from}->{name_to}"))
        i = next(it)
        for j in it:
            self.render_in(page_from, page_to, i, j, style, reverse)
            if self.refresh() == -1:
                break
            i = j


def use_null():
    AppFaker.bar = staticmethod(lambda iterator, message: print(message) or iterator)


def use_rich():
    from rich.progress import track
    AppFaker.bar = staticmethod(lambda iterator, message: track(iterator, description=message))


def use_alive():
    from alive_progress import alive_it
    AppFaker.bar = staticmethod(lambda iterator, message: alive_it(iterator, title=message))


use_alive()
