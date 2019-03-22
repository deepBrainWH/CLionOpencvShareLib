import ctypes
dir = "C:\\Users\\wangheng\\Documents\\software_cup\\train_card_images\\8.jpg"
dir = dir.encode("utf8")
lib = ctypes.cdll.LoadLibrary("./libCLionOpencv.dll")
lib.resize_image(dir, 408, 306)
