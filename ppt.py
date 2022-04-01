from pptFaker import *


class HomeSlide(AbstractSlide):
    def __init__(self):
        super().__init__()
        names = list("例会&培训")

        RandomFlipper.do_full_cache(names)

        self.layers.extend([
            RandomFlipper(names, i, j)
            for i in range(16) for j in range(9)
        ])


if __name__ == '__main__':
    slide = HomeSlide()
    ppt = PowerPointFaker([slide], "2022年4月2日例会培训主视觉demo")
    ppt.mainloop()
