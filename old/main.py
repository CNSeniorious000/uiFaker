from player import CachedPlayer, Page, pg
import sys

page_0 = Page("首页", 0, 0)
page_1 = Page("发布", 1, 0)
page_2 = Page("参与", 1, 0)
page_3 = Page("量表参考", 1, 0)
page_4 = Page("待办", 0, 1)
page_5 = Page("我的", 0, 2)

try:
    duration = int(sys.argv[-1])
except (ValueError, TypeError):
    duration = 40
    # BUG:
    # setting duration at higher number may cause x < 0 sometimes
    # because the interpolation may be out of bound of [0,1]

player = CachedPlayer(
    start_page=page_0,
    duration=duration,
    scale=2,
    page_map={
        # home
        pg.K_0: page_0,
        pg.K_h: page_0,
        pg.K_KP0: page_0,
        pg.K_SPACE: page_0,
        pg.K_RETURN: page_0,
        pg.K_KP_ENTER: page_0,
        # post
        pg.K_1: page_1,
        pg.K_p: page_1,
        pg.K_KP1: page_1,
        # join
        pg.K_2: page_2,
        pg.K_j: page_2,
        pg.K_KP2: page_2,
        # reference
        pg.K_3: page_3,
        pg.K_r: page_3,
        pg.K_KP3: page_3,
        # to-do
        pg.K_4: page_4,
        pg.K_t: page_4,
        pg.K_KP4: page_4,
        # mine
        pg.K_5: page_5,
        pg.K_m: page_5,
        pg.K_KP5: page_5,
    }
)
player.full_cache()
player.mainloop()
