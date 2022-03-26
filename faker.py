from matplotlib import pyplot as plt
import numpy as np
import pygame as pg
from scipy.interpolate import interp1d
import ctypes, cv2
from alive_progress import alive_it

ctypes.windll.user32.SetProcessDPIAware(2)

model = [0, 1/8, 1/2, 3/4, 7/8, 15/16, 31/32, 63/64, 127/128, 1]

f = lambda x: x

def use(model):
    global f
    f = interp1d(np.linspace(0, 1, len(model)), model, "quadratic")

def show(x, y):
    plt.scatter(x, y, s=10)
    plt.plot(x, y, c="red")
    return plt.show()

def show_f(n=32):
    return show_map(0, 1, n)

def show_map(start, end, n):
    x = np.linspace(0, 1, n)
    y = [get_value(start, end, ratio) for ratio in x]
    return show(x, y)

def get_value(start, end, ratio):
    if ratio == 1:
        return end
    else:
        return start + (end-start) * f(ratio)

pg.init()
screenx = 1440
screeny = 2960
height = 247
clock = pg.time.Clock()
dock_cover_add = pg.Surface((screenx, height), pg.SRCALPHA)
dock_cover = pg.Surface((screenx, height), pg.SRCALPHA)
dock_cover_add.fill((32, 32, 32))
dock_cover.fill((255, 255, 255, 32))

class Display:
    def __init__(self):
        self.screen = pg.display.set_mode((screenx//2, screeny//2), vsync=True)
        self.surface = pg.Surface((screenx, screeny), flags=pg.SRCALPHA)

    def __enter__(self):
        return self.surface

    def __exit__(self, exc_type, exc_val, exc_tb):
        pg.transform.smoothscale(
            self.surface, (screenx//2, screeny//2), self.screen
        )
        pg.display.flip()
        pg.display.set_caption(f"FPS: {clock.get_fps():.2f}")
        self.parse_events()
        clock.tick(60)

    @staticmethod
    def parse_events():
        if pg.event.get(pg.QUIT):
            exit()

    def blur_dock(self):
        img = pg.surfarray.pixels3d(self.surface)
        img[:, screeny - height:] = cv2.GaussianBlur(
            img[:, screeny-height:], (155,155), 35
        )

    def add_dock_and_cover(self):
        self.blur_dock()
        self.surface.blit(dock_cover_add, (0, screeny - height), special_flags=pg.BLEND_RGB_ADD)
        self.surface.blit(dock_cover, (0, screeny - height))
        self.surface.blit(cover, (0, 0))


down: pg.Surface
up: pg.Surface
cover = pg.image.load("assets/cover.png")
display = Display()
black = display.surface.copy()


def inspect_surface(surface=display.surface):
    plt.imshow(pg.surfarray.pixels3d(surface).swapaxes(0, 1))
    plt.show()


def do(x):
    with display as screen:
        pos_down = (x // 2 - screenx // 2, 0)
        pos_up = (x, 0)

        screen.blit(down, pos_down)

        black.fill([0, 0, 0, int(get_value(64, 0, (x/screenx)**2.5))])
        screen.blit(black, pos_down)

        screen.blit(up, pos_up)
        display.add_dock_and_cover()


total = 40

count = 0


def render_in():
    global count
    for i in alive_it(range(total)):
        do(get_value(screenx, 0, i / total))
        pg.image.save(display.surface, f"output/{count + i}.png")

    count += total

def render_out():
    global count
    for i in alive_it(range(total)):
        do(get_value(0, screenx, i / total))
        pg.image.save(display.surface, f"output/{count + i}.png")

    count += total

def render_static(page):
    global count
    with display as screen:
        screen.blit(page, (0,0))
        display.add_dock_and_cover()
    it = iter(alive_it(range(total)))
    i = next(it)
    pg.image.save(display.surface, f"output/{count + i}.png")
    b = open(f"output/{count + i}.png", "rb").read()
    for i in it:
        open(f"output/{count + i}.png", "wb").write(b)

    count += total


if __name__ == '__main__':
    use(model)
    page_1 = pg.image.load("assets/首页.png")
    page_2 = pg.image.load("assets/发布.png")
    page_3 = pg.image.load("assets/参与.png")
    page_4 = pg.image.load("assets/量表参考.png")
    page_5 = pg.image.load("assets/待办.png")
    page_6 = pg.image.load("assets/我的.png")

    render_static(page_1)

    down, up = page_1, page_2
    render_in()
    render_static(page_2)
    render_out()

    down, up = page_1, page_3
    render_in()
    render_static(page_3)
    render_out()

    down, up = page_1, page_4
    render_in()
    render_static(page_4)

    down, up = page_4, page_5
    render_in()
    render_static(page_5)

    down, up = page_5, page_6
    render_in()
    render_static(page_6)

    down, up = page_1, page_6
    render_out()
