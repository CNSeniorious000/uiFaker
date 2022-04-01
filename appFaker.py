from enum import IntEnum, auto
from faker import *


class Style(IntEnum):
    moving = auto()
    level = auto()


class StaticCover(AbstractLayer):
    anchor = 0, 0

    @cached_property
    def surface(self):
        return Asset("cover", self.faker.size, flags=pg.SRCALPHA)


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
        return Asset(self.name, self.faker.size)

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
        self.screen.fill(self.color, self.rect, pg.BLEND_ADD)
        dirty_rect = self.screen.blit(self.surface, self.anchor)
        dock_area = pg.surfarray.pixels3d(self.screen)[:, self.y:]
        with timer("blurring"):
            dock_area[:] = cv2.GaussianBlur(
                cv2.blur(cv2.blur(dock_area, (37, 77)), (77, 37)),
                (0, 0), sigmaX=7, sigmaY=17
            )
        return dirty_rect


class AppFaker(Faker):
    bar = staticmethod(lambda iterator, message: print(message) or iterator)

    def __init__(self, w=1440, h=2960, h_dock=247, duration=40):
        super().__init__(w, h)
        self.h_dock = h_dock
        self.duration = duration

        self.cover = StaticCover()
        self.shadow = SolidShadow()
        self.dock = Dock(27, 37)

        self.cached_render_motion_once = lru_cache(w + w + 1)(surfcache((w, h))(self.cached_render_motion_once))
        self.cached_render_motion_multi = cache(surfcache((w, h))(self.cached_render_motion_multi))

    def render_at(self, page_last, page_next, i, style, reverse):
        if reverse:
            page_last, page_next = page_next, page_last
            x = int(self.at(0, self.w, i))
        else:
            x = int(self.at(self.w, 0, i))

        # maybe once more
        self.buffer.blit(self.cached_render_motion_once(page_last.name, page_next.name, x, style), (0, 0))
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
        self.buffer.blit(self.cached_render_motion_multi(page_last.name, page_next.name, x, y, style), (0, 0))
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
        return self.buffer.copy()

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

        self.buffer.blit(pg.image.frombuffer((image / len(sample)).astype(np.uint8), (w, h), "BGR"), (0, 0))
        return self.buffer.copy()

    def render_static_once(self):
        self.dock.draw()
        self.cover.draw()


class AppPlayer(AppFaker):
    def animate(self, page_from: Page, page_to: Page, style, reverse):
        if (name_from := page_from.name) == (name_to := page_to.name):
            return print(f"invalid changing from {name_from} to {name_to}")
        use_null()
        it = iter(self.bar(np.linspace(0, 1, self.duration), f"{name_from}->{name_to}"))
        i = next(it)
        for j in it:
            with timer("render_in"):
                self.render_in(page_from, page_to, i, j, style, reverse)
            if self.refresh() == -1:
                break
            i = j
        self.buffer.blit(page_to.surface, (0, 0))  # the last frame
        self.render_static_once()
        self.refresh()


def use_null():
    AppFaker.bar = staticmethod(lambda iterator, message: print(message) or iterator)


def use_rich():
    from rich.progress import track
    AppFaker.bar = staticmethod(lambda iterator, message: track(iterator, description=message))


def use_alive():
    from alive_progress import alive_it
    AppFaker.bar = staticmethod(lambda iterator, message: alive_it(iterator, title=message))


use_alive()
