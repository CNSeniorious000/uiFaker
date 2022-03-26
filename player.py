from faker import Faker, Page, Style, model_fluent, pg, np
from alive_progress import alive_it
from contextlib import suppress
from math import gcd


class Player(Faker):
    def __init__(self, w=1440, h=2960, *args, scale=2, **kwargs):
        print(f"{gcd(w, h) = }")
        self.scale = scale
        self.screen = pg.display.set_mode((w // scale, h // scale), vsync=True)
        super().__init__(w, h, *args, **kwargs)

    def __enter__(self):
        return self.surface

    def __exit__(self, exc_type, exc_val, exc_tb):
        pg.transform.smoothscale(
            self.surface, (self.w // self.scale, self.h // self.scale), self.screen
        )
        pg.display.flip()
        pg.display.set_caption(f"FPS: {self.clock.get_fps():.2f}")
        self.parse_events()
        self.clock.tick_busy_loop(60)

    @staticmethod
    def parse_events():
        if pg.event.get(pg.QUIT):
            exit()


class AniPlayer(Player):
    def __init__(self, *args, start_page, page_map: dict = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_model(model_fluent)
        self.current = start_page
        self.page_map = page_map

    def show_animation(self, _from: Page, _to: Page):
        if _from.page_depth == _to.page_depth == 0:
            if _from.page_index == _to.page_index:
                return print(f"invalid moving from {_from} to {_to}")
            style = Style.moving
            reverse = _from.page_index > _to.page_index
        else:
            style = Style.level
            reverse = _to.page_depth == 0

        self.current = _to

        for i in alive_it(np.linspace(0, 1, self.duration), title=f"{_from}->{_to}"):
            with self as screen:
                self.render_at(_from, _to, i, style, reverse)

    def mainloop(self):
        while True:
            for event in pg.event.get():
                if event.type == pg.KEYDOWN:
                    with suppress(KeyError):
                        print(event.key)
                        self.show_animation(self.current, self.page_map[event.key])
                        break
                elif event.type == pg.QUIT:
                    return
            else:
                self.parse_events()
                self.clock.tick(60)
