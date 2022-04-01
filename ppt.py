from pptFaker import *


class HomeSlide(AbstractSlide):
    def __init__(self):
        super().__init__()
        self.layers.append(RandomFlipper(list("视觉设计部"), 0, 0))


if __name__ == '__main__':
    PowerPointFaker([HomeSlide()]).mainloop()
