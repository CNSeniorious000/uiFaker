from appFaker import AppPlayer, Page, Style, show_time_map


def test_faking_app():
    page_home = Page("首页")
    page_post = Page("发布")
    page_join = Page("参与")
    page_todo = Page("待办")
    page_mine = Page("我的")
    page_reference = Page("量表参考")

    player = AppPlayer(720, 1480, 124)

    player.animate(page_home, page_post, Style.level, False)
    player.animate(page_post, page_join, Style.level, True)
    player.animate(page_join, page_reference, Style.level, True)
    player.animate(page_reference, page_home, Style.level, False)
    player.animate(page_home, page_todo, Style.level, False)
    player.animate(page_todo, page_mine, Style.level, True)
    print(f"{player.cached_render_motion_once.cache_info() = }")
    print(f"{player.cached_render_motion_multi.cache_info() = }")

    show_time_map()


if __name__ == '__main__':
    test_faking_app()
