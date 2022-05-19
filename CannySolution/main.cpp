#include <iostream>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/ximgproc/ridgefilter.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include <string>
#include <vector>
#include <algorithm>
#include <utility>

cv::Mat canny(cv::Mat image) {
    cv::Mat edges;
    //cv::Canny(image, edges, 20, 110, 3); //для моей фильтрации
    cv::Canny(image, edges, 20, 140, 3); //для только красный
    return edges;
}

cv::Mat ridge_filter(cv::Mat image) {
    cv::Mat ridges;
    cv::Ptr<cv::ximgproc::RidgeDetectionFilter> ridge_filter = cv::ximgproc::RidgeDetectionFilter::create();
    (*ridge_filter).getRidgeFilteredImage(image, ridges);
    return ridges;
}

cv::Mat overlay(cv::Mat image, cv::Mat template_im) {
    for (int i = 0; i < image.size[0]; ++i) {
        for (int j = 0; j < image.size[1]; ++j) {
            uchar& pixel = template_im.at<uchar>(i, j);
            if (pixel == 255) {
                image.at<cv::Vec3b>(i, j)[0] = 0;
                image.at<cv::Vec3b>(i, j)[1] = 255;
                image.at<cv::Vec3b>(i, j)[2] = 0;
            }
        }
    }
    return image;
}

bool isEdge(uchar point) {
    return (point == 255);
}

bool hasDiffrent(cv::Mat image, cv::Mat edges, int x, int y) {
    std::vector<int> offset_x({0, 1, 2, 2, 2, 2, 2, 1}); //обработка рамки 2
    std::vector<int> offset_y({2, 2, 2, 1, 0, -1, -2, -2}); //обработка рамки 2
    for (int i = 0; i < offset_x.size(); ++i) {
        if (std::abs(image.at<uchar>(x + offset_x[i], y + offset_y[i]) - image.at<uchar>(x - offset_x[i], y - offset_y[i])) > 10) {
            //std::cout << (int)image.at<uchar>(x + offset_x[i], y + offset_y[i]) << " " << (int)image.at<uchar>(x - offset_x[i], y - offset_y[i]) << "\n";
            return true;
        }
    }
    return false;
}

void deleteErrors(cv::Mat image_source, cv::Mat edges_source) { //на вход -- одноканальные ребра
    int window = 7;
    cv::Mat image;
    cv::Scalar value(0, 0, 0);
    cv::copyMakeBorder(image_source, image, window, window, window, window, cv::BORDER_CONSTANT, value );
    cv::Mat edges;
    cv::copyMakeBorder(edges_source, edges, window, window, window, window, cv::BORDER_CONSTANT, value );

    std::vector<int> x_offset({0, 1, 1, 1, 0, -1, -1, -1}); //обрабатываем 8 направлений
    std::vector<int> y_offset({1, 1, 0, -1, -1, -1, 0, 1});
    std::vector<bool> find_edges(x_offset.size(), false);
    std::vector<std::pair<int, int>> to_exclude;
    for (int i = window; i < edges.size[0] - window; ++i) {
        for (int j = window; j < edges.size[1] - window; ++j) {
            if (!hasDiffrent(image, edges, i, j)) {
                edges.at<uchar>(i, j) = 0;
                edges_source.at<uchar>(i - window, j - window) = 0;
            }
        }
    }

    for (int i = window; i < edges.size[0] - window; ++i) {
        for (int j = window; j < edges.size[1] - window; ++j) {
            if (isEdge(edges.at<uchar>(i, j))) {
                for (int pos = 0; pos < find_edges.size(); ++pos) {
                    find_edges[pos] = false;
                }
                for (int offset = 1; offset <= window; ++offset) {
                    for (int pos = 0; pos < x_offset.size(); ++pos) {
                        if (isEdge(edges.at<uchar>(i + x_offset[pos] * offset, j + y_offset[pos] * offset))) {
                            find_edges[pos] = true;
                        }
                    }
                }
                int neighbours = 0;
                for (int pos = 0; pos < find_edges.size(); ++pos) {
                    if (find_edges[pos]) {
                        ++neighbours;
                    }
                }
                if (neighbours < 4) {
                    to_exclude.push_back(std::make_pair(i - window, j - window));
                }
            }
        }
    }
    for (auto i : to_exclude) {
        edges_source.at<uchar>(i.first, i.second) = 0;
    }
}

struct Component {
    Component(int left, int right, int up, int down, int x, int y) : left(left), right(right), up(up), down(down), x(x), y(y) {}
    int left = 0;
    int right = 0;
    int up = 0;
    int down = 0;
    int x = 0;
    int y = 0;
};

void insertInComponents(int x, int y, std::vector<Component>& components) {
    int threshold = 15;
    for (auto& current : components) {
        if (std::min(std::abs(current.left - x), std::abs(current.right - x)) + 
            std::min(std::abs(current.up - y), std::abs(current.down - y)) < threshold) {
                current.left = std::min(current.left, x);
                current.right = std::max(current.right, x);
                current.up = std::min(current.up, y);
                current.down = std::max(current.down, y);
                return;
            }
    }
    components.push_back(Component(x, x, y, y, x, y));
}

void borders(cv::Mat image_source) { //на вход -- одноканальные ребра
    int window = 7;
    cv::Mat image;
    cv::Scalar value(0, 0, 0);
    cv::copyMakeBorder(image_source, image, window, window, window, window, cv::BORDER_CONSTANT, value);
    std::vector<Component> components;
    for (int i = window; i < image.size[0] - window; ++i) {
        for (int j = window; j < image.size[1] - window; ++j) {
            if (isEdge(image.at<uchar>(i, j))) {
                insertInComponents(i, j, components);
                image_source.at<uchar>(i - window, j - window) = 0;
            }
        }
    }
    for (auto borders : components) {
        if ((borders.right - borders.left) * (borders.down - borders.up) < 60) {
            continue;
        }
        for (int i = borders.left; i < borders.right; ++i) {
            image_source.at<uchar>(i - window, borders.down - window) = 255;
            image_source.at<uchar>(i - window, borders.up - window) = 255;
        }

        for (int i = borders.up; i < borders.down; ++i) {
            image_source.at<uchar>(borders.left - window, i - window) = 255;
            image_source.at<uchar>(borders.right - window, i - window) = 255;
        }
    }
}

int main() {
    std::string path = "/home/ppadas/Documents/Find_filament/Compressed/";
    std::string path_initial = "/home/ppadas/Documents/Find_filament/Samples/";
    std::string file_names[] = {"M1.png", "M2.png", "M3.png", "M4.png", "M5.png", "M6.png", "M7.png", "M8.png"};
    for (auto name : file_names) {
        cv::Mat initial_image = cv::imread(path_initial + name);
        //cv::imwrite("./Result/" + name.substr(0, name.size() - 4) + std::string("_0_input.png"), initial_image);
        cv::Mat image = cv::imread(path + name, cv::IMREAD_GRAYSCALE);
        cv::Mat input = cv::imread(path + name);
        cv::Mat dst;
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(4, 4));
        cv::dilate(image, dst, kernel);
        //cv::imwrite("./Result/" + name.substr(0, name.size() - 4) + std::string("_2_wb.png") , image);
        //cv::imwrite("./Result/" + name.substr(0, name.size() - 4) + std::string("_3_delate.png") , dst);
        cv::Mat edges = canny(dst);
        cv::Mat output_canny = input.clone();
        overlay(output_canny, edges);

        deleteErrors(image, edges);
        cv::Mat output_delete_errors = input.clone();
        overlay(output_delete_errors, edges);

        borders(edges);
        cv::Mat output = input.clone();
        overlay(output, edges);
        //cv::imwrite("./Result/" + name.substr(0, name.size() - 4) + std::string("_4_canny.png"), output_canny);
        //cv::imwrite("./Result/" + name.substr(0, name.size() - 4) + std::string("_1_compressed_and_red.png") , input);
        //cv::imwrite("./Result/" + name.substr(0, name.size() - 4) + std::string("_5_errors.png"), output_delete_errors);
        //cv::imwrite("./Result/" + name.substr(0, name.size() - 4) + std::string("_6_borders.png"), output);
        cv::imwrite("./Result/" + name.substr(0, name.size() - 4) + std::string("_Canny.png"), output);
    }
    return 0;
}
