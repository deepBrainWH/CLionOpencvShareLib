//
// Created by wangheng on 2019/3/21.
//
#include <iostream>
#include <opencv/cv.h>
#include <opencv2/opencv.hpp>
#include <string>

using namespace cv;
using namespace std;

class ImageOpeartion {
public:
    void show_image() {
        Mat image = imread(R"(C:\Users\wangheng\Pictures\IMG_20190315_163220.png)", IMREAD_COLOR);
        namedWindow("image", WINDOW_FREERATIO);
        resizeWindow("image", 300, 400);
        imshow("image", image);
        waitKey(0);
        VideoCapture capture(0);
        destroyAllWindows();
    }

    void resize_image(char *str, int width, int height, char *outpuPath) {
        string path(str);
        string output_path(outpuPath);
        Mat image = imread(path, IMREAD_ANYCOLOR);
        Size size(width, height);
        resize(image, image, size);
        imwrite(output_path, image);
    }
};

extern "C" {
ImageOpeartion obj;
void show_image_c() {
    obj.show_image();
}
void resize_image(char *str, int wigth, int height, char *outputPath) {
    obj.resize_image(str, wigth, height, outputPath);
}
void add_test(int a, int b) {
    cout << a << "\t" << b;
}
}


