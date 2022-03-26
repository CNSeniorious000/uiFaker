from faker import Faker, Page, Style, model_fluent, pg, np, cv2, cache
from alive_progress import alive_it
from contextlib import suppress
from diskcache import Cache
from math import gcd


class Player(Faker):
    def __init__(self, w=1440, h=2960, *args, scale=2, **kwargs):
        print(f"{gcd(w, h) = }")
        self.scaled_size = (w // scale, h // scale)
        self.screen = pg.display.set_mode(self.scaled_size, vsync=True)
        super().__init__(w, h, *args, **kwargs)

    def __enter__(self):
        return self.surface

    def __exit__(self, exc_type, exc_val, exc_tb):
        pg.transform.smoothscale(self.surface, self.scaled_size, self.screen)
        self.ending()

    def ending(self):
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

        self.animate(_from, _to, style, reverse)

    def animate(self, _from, _to, style, reverse):
        for i in alive_it(np.linspace(0, 1, self.duration), title=f"{_from}->{_to}"):
            with self:
                self.render_on(_from, _to, i, style, reverse)

    def mainloop(self):
        while True:
            for event in pg.event.get():
                if event.type == pg.KEYDOWN:
                    with suppress(KeyError):
                        self.show_animation(self.current, self.page_map[event.key])
                        break
                elif event.type == pg.QUIT:
                    return
            else:
                self.parse_events()
                self.clock.tick(60)


class CachedPlayer(AniPlayer):
    current_player: "CachedPlayer" = None
    persistent_cache = Cache("cache")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        CachedPlayer.current_player = self

    render_on = render_at = None

    def animate(self, _from, _to, style, reverse):
        for i in alive_it(np.linspace(0, 1, self.duration), title=f"{_from}->{_to}"):
            self.screen.blit(self.get_surface_at(_from, _to, i, style, reverse), (0, 0))
            self.ending()

    @staticmethod
    @cache
    def get_surface_at(_from, _to, i, style, reverse):
        """get real-size pygame surface"""
        print(f"getting surface @ {_from.title}->{_to.title}")
        player = CachedPlayer.current_player
        size = player.scaled_size
        return pg.image.frombuffer(
            CachedPlayer.get_buffer_at(_to.title, _from.title, player.at(0, player.w, i), style, size),
            size, "BGR"
        ).convert() if reverse else pg.image.frombuffer(
            CachedPlayer.get_buffer_at(_from.title, _to.title, player.at(player.w, 0, i), style, size),
            size, "BGR"
        ).convert()

    @staticmethod
    @persistent_cache.memoize()
    def get_buffer_at(title_from, title_to, x, style, size):
        """get real-size ndarray buffer"""
        print(f"getting buffer @ {title_from}->{title_to}")
        player = CachedPlayer.current_player
        Faker.render_at(player, Page.load_surface(title_from), Page.load_surface(title_to), x, style)
        surf = player.surface
        return cv2.resize(
            np.frombuffer(surf.get_buffer(), np.uint8).reshape(*surf.get_size()[::-1], -1)[..., :3],
            size, interpolation=cv2.INTER_AREA
        )
