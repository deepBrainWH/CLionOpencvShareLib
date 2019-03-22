#include <iostream>
#include <opencv/cv.h>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

int main() {
    Mat image = imread(R"(C:\Users\wangheng\Pictures\IMG_20190315_163220.png)", IMREAD_COLOR);
    namedWindow("image", WINDOW_FREERATIO);
    resizeWindow("image", 300, 400);
    imshow("image", image);
    waitKey(0);
    VideoCapture capture(0);
    destroyAllWindows();
    return 0;
}

void RSAAlgorithm(){

}