import cv2
import os
import pandas as pd
pd.set_option('display.max_columns', None)

class split_number:
    def __init__(self):
        self.prefix_dir = "C:\\Users\\wangheng\\Documents\\software_cup\\images\\"
        self.write_img_file_prefix_path = "C:\\Users\\wangheng\\Documents\\software_cup\\number\\"
        self.file_names = os.listdir(self.prefix_dir)
        self.file_path = []
        self.__all_file_path()

    def __all_file_path(self):
        for f in self.file_names:
            self.file_path.append(os.path.join(self.prefix_dir, f))

    def _split_img(self, img_index):
        imread = cv2.imread(self.file_path[img_index], cv2.IMREAD_ANYCOLOR)
        tmp_path = self.file_path[img_index].split(".")[0][-7:-3]
        for i in range(4):
            cv2.imwrite(self.write_img_file_prefix_path+tmp_path[i]+"-"+str(img_index)+str(i)+".jpg", imread[3:44, ((i*30)+1):((i+1)*30-1)])
            cv2.rectangle(imread, (i * 30, 2), ((i + 1) * 30 - 2, 44), (200, 0, 0), 1)
        cv2.imshow("image", imread)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print(self.file_names[img_index], "===>", imread.shape, tmp_path[0][-7:-3])

    def write_to_csv_file(self):
        listdir = os.listdir(self.write_img_file_prefix_path)
        frame = []
        for fn in listdir:
            frame.append({"x": os.path.join(self.write_img_file_prefix_path, fn), "y":fn[0]})
        frame = pd.DataFrame(frame, columns=("x", "y"))
        frame.to_csv("C:\\Users\\wangheng\\Documents\\software_cup\\dataframe.csv")

    def one_hot_encoder(self):
        df = pd.read_csv("C:\\Users\\wangheng\\Documents\\software_cup\\dataframe.csv", index_col=0)
        dummies = pd.get_dummies(df["y"], "y")
        concat = pd.concat([df, dummies], 1)
        concat.to_csv("C:\\Users\\wangheng\\Documents\\software_cup\\dataframe1.csv")


if __name__ == "__main__":
    number = split_number()
    # for i in range(len(number.file_names)):
    #     number._split_img(i)
    # number.write_to_csv_file()
    number.one_hot_encoder()