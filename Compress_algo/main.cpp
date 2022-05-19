#include <iostream>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/ximgproc/ridgefilter.hpp>
#include <string>

int main() {
    std::string path = "/home/ppadas/Documents/Find_filament/";
    std::string file_names[] = {"M1.png", "M2.png", "M3.png", "M4.png", "M5.png", "M6.png", "M7.png", "M8.png", "Example.png"};
    for (auto name : file_names) {
        //cv::Mat image = cv::imread(path + "Samples/" + name, cv::IMREAD_GRAYSCALE);
        cv::Mat image = cv::imread(path + "Samples/" + name);
        for (int i = 0; i < image.size[0]; ++i) {
            for (int j = 0; j < image.size[1]; ++j) {
                int value = (image.at<cv::Vec3b>(i, j)[2] * 2 - image.at<cv::Vec3b>(i, j)[0] - image.at<cv::Vec3b>(i, j)[1]) / 2;
                //image.at<cv::Vec3b>(i, j)[2] = value > 0 ? value : 0; 
                //image.at<cv::Vec3b>(i, j)[2] = image.at<cv::Vec3b>(i, j)[2]; 
                image.at<cv::Vec3b>(i, j)[0] = 0;
                image.at<cv::Vec3b>(i, j)[1] = 0;
            }
        }
        double ratio = 0.3;
        int x_dim = image.size[0] * ratio;
        int y_dim = image.size[1] * ratio;
        cv::Mat image_compressed;
        cv::resize(image, image_compressed, cv::Size(y_dim, x_dim));
        cv::imwrite(path + "Compressed/" + name, image_compressed);
    }
    return 0;
}