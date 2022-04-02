pptFaker是一个基于网格系统来排版的ppt框架

TO-DOs:

- 让`RandomFlipper`也可以自由变换大小
- 可能在检测到图片是黑白的时候可以内存的使用量到原来的1/3

BUGs:

- 任何`Slide`对象都应该有`bgd`，否则切换问题没法解决
- `Slide.render`应该接受一个可选的`surface`参数，否则切换时会有闪烁