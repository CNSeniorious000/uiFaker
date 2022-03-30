def test_new_mechanism():
    from faker import AppPlayer, Page, Style
    page_home = Page("首页")
    page_post = Page("发布")
    page_join = Page("参与")
    page_todo = Page("待办")
    page_mine = Page("我的")
    page_reference = Page("量表参考")

    player = AppPlayer(720, 1480, 124)

    player.animate(page_home, page_post, Style.level, False)
    # player.debug_inspect()
    print(f"{player.cached_render_motion_once.cache_info() = }")
    print(f"{player.cached_render_motion_multi.cache_info() = }")
    player.animate(page_home, page_post, Style.level, False)
    print(f"{player.cached_render_motion_once.cache_info() = }")
    print(f"{player.cached_render_motion_multi.cache_info() = }")
    player.animate(page_post, page_home, Style.level, True)
    # player.debug_inspect()
    print(f"{player.cached_render_motion_once.cache_info() = }")
    print(f"{player.cached_render_motion_multi.cache_info() = }")


if __name__ == '__main__':
    test_new_mechanism()
    import faker
    print(faker.t1 / 2)
    print(faker.t2 / 2)
