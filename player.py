from faker import Faker, pg
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
        self.clock.tick(60)

    @staticmethod
    def parse_events():
        if pg.event.get(pg.QUIT):
            exit()
