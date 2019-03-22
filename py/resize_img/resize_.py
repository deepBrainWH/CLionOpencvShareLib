import ctypes
dir = "C:\\Users\\wangheng\\Documents\\software_cup\\train_card_images\\"
file_path = dir + "8.jpg"
file_path = file_path.encode("utf8")
outpu_path = (dir + "resize-8.jpg").encode("utf8")
lib = ctypes.cdll.LoadLibrary("../../lib_winX64/libCLionOpencv.dll")
lib.resize_image(file_path, 408, 306, outpu_path)
