from pptFaker import *


class HomeSlide(Slide):
    def __init__(self):
        super().__init__()
        names = list("例会&培训")
        RandomFlipper.do_full_cache(names)
        positions = [(i, j) for i in range(12) for j in range(5, 9)] + \
                    [(i, 4) for i in range(4)]
        self.layers.extend([RandomFlipper(names, *pos) for pos in positions])
        self.layers.append(AcrylicImage("uiFaker", 1.5, 6.75, 4, 0.5))
        self.layers.append(circles := LeftTopCircles(0, 0, 4, 0.7, 0.015))
        self.layers.append(ReversedLayer(12, 5, circles))


class ppt(PowerPointFaker):
    def __init__(self):
        # super().__init__([HomeSlide()], "2022年4月2日例会培训主视觉demo", pg.NOFRAME)
        super().__init__([HomeSlide()], "2022年4月2日例会培训主视觉demo")
        # self.centering()
        self.screen.blit(Asset("bgd", parse(16, 9)), (0, 0))
        pg.display.flip()


if __name__ == '__main__':
    ppt().mainloop()
    show_time_map()
