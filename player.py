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
    persistent_cache = Cache("cache", cull_limit=0)

    def __init__(self, *args, sampling=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        CachedPlayer.current_player = self
        self.sampling = sampling
        self.animate = self.animate_blurred

    render_on = render_at = None

    def animate(self, _from, _to, style, reverse):
        if (title_from := _from.title) == (title_to := _to.title):
            return print(f"invalid changing from {title_from} to {title_to}")
        for i in alive_it(np.linspace(0, 1, self.duration), title=f"{title_from}->{title_to}", calibrate=60):
            self.screen.blit(self.get_surface_at(title_from, title_to, i, style, reverse), (0, 0))
            self.ending()

    def animate_blurred(self, _from, _to, style, reverse):
        if (title_from := _from.title) == (title_to := _to.title):
            return print(f"invalid changing from {title_from} to {title_to}")
        it = iter(alive_it(np.linspace(0, 1, self.duration), title=f"{title_from}->{title_to}", calibrate=60))
        i = next(it)
        for j in it:
            self.screen.blit(self.get_surface_in(title_from, title_to, i, j, style, reverse), (0, 0))
            i = j
            self.ending()
        # ending
        self.screen.blit(self.get_surface_at(title_from, title_to, 1, style, reverse), (0, 0))
        self.ending()

    @staticmethod
    @cache
    def get_surface_in(title_from, title_to, i, j, style, reverse):
        self = CachedPlayer.current_player
        size = self.scaled_size
        return pg.image.frombuffer(
            self.get_buffer_in(title_to, title_from, self.at(0, self.w, i), self.at(0, self.w, j), style, size)
            if reverse else
            self.get_buffer_in(title_from, title_to, self.at(self.w, 0, i), self.at(self.w, 0, j), style, size),
            size, "BGR"
        )

    @staticmethod
    @cache
    def get_surface_at(title_from, title_to, i, style, reverse):
        """get real-size pygame surface"""
        self = CachedPlayer.current_player
        size = self.scaled_size
        return pg.image.frombuffer(
            self.get_buffer_at.__wrapped__(title_to, title_from, self.at(0, self.w, i), style, size)
            if reverse else
            self.get_buffer_at.__wrapped__(title_from, title_to, self.at(self.w, 0, i), style, size),
            size, "BGR"
        ).convert()

    @staticmethod
    @persistent_cache.memoize()
    def get_buffer_in(title_from, title_to, x, y, style, size):
        self = CachedPlayer.current_player
        sample = np.linspace(x, y, int(abs(y - x) * self.sampling + 1))
        lth = len(sample)
        print(f"{x}->{y} sampling at {lth}")
        image = np.zeros((*self.scaled_size[::-1], 3), np.float_)
        for x in sample:
            image += self.get_buffer_at(title_from, title_to, x, style, size)
        return (image / lth).astype(np.uint8)

    @staticmethod
    @persistent_cache.memoize()
    def get_buffer_at(title_from, title_to, x, style, size):
        """get real-size ndarray buffer"""
        # print(f"getting buffer @ {title_from}->{title_to}")
        self = CachedPlayer.current_player
        Faker.render_at(self, Page.load_surface(title_from), Page.load_surface(title_to), x, style)
        surf = self.surface
        return cv2.resize(
            np.frombuffer(surf.get_buffer(), np.uint8).reshape(*surf.get_size()[::-1], -1)[..., :3],
            size, interpolation=cv2.INTER_AREA
        )

    def full_cache(self):
        from itertools import permutations
        for _from, _to in permutations(set(self.page_map.values()), 2):
            self.show_animation(_from, _to)
