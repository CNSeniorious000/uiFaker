def test_changing():
    from faker import Page, model_fluent, Style
    from alive_progress import alive_it
    from player import Player

    player = Player(scale=4, duration=20)
    player.use_model(model_fluent)

    page_1 = Page.load_surface("首页")
    page_2 = Page.load_surface("发布")
    page_3 = Page.load_surface("参与")
    page_4 = Page.load_surface("量表参考")
    page_5 = Page.load_surface("待办")
    page_6 = Page.load_surface("我的")

    for i in alive_it(range(player.duration)):
        with player as screen:
            player.render_on(page_1, page_2, i / player.duration, Style.level)

    for i in alive_it(range(player.duration)):
        with player as screen:
            player.render_on(page_2, page_5, i / player.duration, Style.level, True)
