import ctypes

library = ctypes.cdll.LoadLibrary("./libCLionOpencv.dll")
library.show_image_c()