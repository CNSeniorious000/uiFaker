import numpy as np, pygame as pg, ctypes, cv2, enum
from functools import cache

ctypes.windll.user32.SetProcessDPIAware(2)

model_fluent = [0, 1 / 8, 1 / 2, 3 / 4, 7 / 8, 15 / 16, 31 / 32, 63 / 64, 127 / 128, 1]

pg.display.set_mode((1, 1), pg.HIDDEN)


class Curve:
    f = staticmethod(lambda x: x)

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


class Page(pg.Surface):
    @staticmethod
    @cache
    def load_surface(title):
        print(f"loading {title} for the first time ...")
        return pg.image.load(f"assets/{title}.png").convert_alpha()

    def __init__(self, asset_title, depth, index):
        self.title = asset_title
        surface: pg.Surface = self.load_surface(asset_title)
        print(f"{surface.get_flags() = }, {surface.get_bitsize() = }")
        super().__init__(surface.get_size(), surface.get_flags(), surface.get_bitsize())
        self.get_buffer().write(surface.get_buffer().raw)

        self.page_depth = depth
        self.page_index = index

    __repr__ = __str__ = lambda self: f"Page(title={self.title}, depth={self.page_depth}, index={self.page_index})"


class Style(enum.IntEnum):
    moving = 1
    level = 2


class Faker(Curve):
    def __init__(self, w=1440, h=2960, h_dock=247, alpha=32, duration=40, cover=None):
        self.w = w
        self.h = h
        self.h_dock = h_dock
        self.alpha = alpha
        self.duration = duration
        self.cover = Page.load_surface(cover or "cover")

        self.clock = pg.time.Clock()

        self.layer_add = pg.Surface((w, h_dock), pg.SRCALPHA)
        self.layer = pg.Surface((w, h_dock), pg.SRCALPHA)

        self.layer_add.fill((alpha, alpha, alpha))
        self.layer.fill((255, 255, 255, alpha))

        self.surface = pg.Surface((w, h), pg.SRCALPHA)
        self.black = self.surface.convert()
        self.black.fill("#000000")

        self.dock_top = h - h_dock

    def inspect_surface(self, surface=None):
        from matplotlib import pyplot as plt
        plt.imshow(pg.surfarray.pixels3d(surface or self.surface).swapaxes(0, 1))
        plt.show()

    def blur_dock(self):
        img = pg.surfarray.pixels3d(self.surface)
        img[:, self.dock_top:] = cv2.GaussianBlur(
            img[:, self.dock_top:], (155, 155), 55
        )

    def post_processing(self):
        """add dock and cover"""
        self.blur_dock()
        self.surface.blit(self.layer_add, (0, self.dock_top), special_flags=pg.BLEND_RGB_ADD)
        self.surface.blit(self.layer, (0, self.dock_top))
        self.surface.blit(self.cover, (0, 0))

    def render_on(self, _last, _next, i, style: Style, reverse=False):
        if reverse:
            self.render_at(_next, _last, self.at(0, self.w, i), style)
        else:
            self.render_at(_last, _next, self.at(self.w, 0, i), style)

    def render_at(self, _last, _next, x, style):
        anchor_upper = int(x), 0
        anchor_lower = int((x - self.w) / style), 0
        self.surface.blit(_last, anchor_lower)
        if style is Style.level:
            self.black.set_alpha(int(self.at(64, 0, (x / self.w) ** 2.5)))
            self.surface.blit(self.black, anchor_lower)
        self.surface.blit(_next, anchor_upper)
        self.post_processing()


if __name__ == '__main__':
    page_1 = Page.load_surface("assets/首页.png")
    page_2 = Page.load_surface("assets/发布.png")
    page_3 = Page.load_surface("assets/参与.png")
    page_4 = Page.load_surface("assets/量表参考.png")
    page_5 = Page.load_surface("assets/待办.png")
    page_6 = Page.load_surface("assets/我的.png")
