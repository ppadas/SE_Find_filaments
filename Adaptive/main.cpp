#include <iostream>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/ximgproc/ridgefilter.hpp>
#include <string>
#include <cmath>
#include <vector>
#include <stack>

int neighborsCount (cv::Mat image, int i, int j) {
    std::vector<int> offset_x({0, 1, 2, 2, 2, 2, 2, 1, 0, 1, 1, 1}); //обработка окна 2
    std::vector<int> offset_y({2, 2, 2, 1, 0, -1, -2, -2, 1, 0, -1, -1}); //обработка окна 2
    int my_neighbours = 0;
    int pixel = image.at<cv::Vec3b>(i, j)[2];
    for (int side = 0; side < offset_x.size(); ++side) {
        int neighbor1 = image.at<cv::Vec3b>(i + offset_x[side], j + offset_y[side])[2];
        int neighbor2 = image.at<cv::Vec3b>(i - offset_x[side], j - offset_y[side])[2];
        if (neighbor1 == 255) {
            ++my_neighbours;
        }
        if (neighbor2 == 255) {
            ++my_neighbours;
        }
    }
    return my_neighbours;
}

bool isEdge(uchar point) {
    return (point == 255);
}

bool hasDiffrent(cv::Mat image, cv::Mat edges, int x, int y) {
    std::vector<int> offset_x({0, 1, 2, 2, 2, 2, 2, 1}); //обработка рамки 2
    std::vector<int> offset_y({2, 2, 2, 1, 0, -1, -2, -2}); //обработка рамки 2
    for (int i = 0; i < offset_x.size(); ++i) {
        if (std::abs(image.at<uchar>(x + offset_x[i], y + offset_y[i]) - image.at<uchar>(x - offset_x[i], y - offset_y[i])) > 10) {
            return true;
        }
    }
    return false;
}

struct Component {
    Component(int left, int right, int up, int down, int x = 0, int y = 0, int n = 0) 
        : left(left), right(right), up(up), down(down), x(x), y(y), max_environment(n) {}
    int left = 0;
    int right = 0;
    int up = 0;
    int down = 0;
    int x = 0;
    int y = 0;
    int max_environment = 0;
};

void insertInComponents(int x, int y, std::vector<Component>& components, int neighbours) {
    int threshold = 7;
    int contribution = neighbours == 8 ? 1 : 0;
    for (auto& current : components) {
        int height_dist = std::min(std::abs(current.up - y), std::abs(current.down - y));
        int width_dist = std::min(std::abs(current.left - x), std::abs(current.right - x));
        if (height_dist * height_dist + width_dist * width_dist < threshold * threshold) {
                current.left = std::min(current.left, x);
                current.right = std::max(current.right, x);
                current.up = std::min(current.up, y);
                current.down = std::max(current.down, y);
                current.max_environment += contribution;
                return;
            }
    }
    components.push_back(Component(x, x, y, y, x, y, contribution));
}

int neighborsCountOneCanal(cv::Mat image, int x, int y) {
    std::vector<int> x_offset({0, 1, 1, 1});
    std::vector<int> y_offset({1, 1, 0, -1});
    int answer = 0;
    for (int i = 0; i < x_offset.size(); ++i) {
        if (isEdge(image.at<uchar>(x + x_offset[i], y + y_offset[i]))) {
            ++answer;
        }
        if (isEdge(image.at<uchar>(x - x_offset[i], y - y_offset[i]))) {
            ++answer;
        }
    }
    return answer;
}

void draw(cv::Mat image_source, std::vector<Component>& components, int window) {
    for (auto borders : components) {
        
        if ((borders.right - borders.left) * (borders.down - borders.up) < 20) {
            continue;
        }
        if ((borders.right - borders.left) <= 3 || (borders.down - borders.up) <= 3) {
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

void deleteErrorsAndDraw(cv::Mat image_source) { //формируем компоненты связности и находим самых отрисовываем представителей
    int window = 7;
    cv::Mat image;
    cv::Scalar value(0, 0, 0);
    cv::copyMakeBorder(image_source, image, window, window, window, window, cv::BORDER_CONSTANT, value);
    std::vector<Component> components;
    for (int i = window; i < image.size[0] - window; ++i) {
        for (int j = window; j < image.size[1] - window; ++j) {
            if (isEdge(image.at<uchar>(i, j))) {
                insertInComponents(i, j, components, neighborsCountOneCanal(image, i, j));
                image_source.at<uchar>(i - window, j - window) = 0;
            }
        }
    }

    draw(image_source, components, window);
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

std::tuple<bool, int, int, int, int> breadth_first_and_clear(cv::Mat edges_source, int x, int y,  std::vector<std::vector<bool>>& visit) {
    int threshold = 30;
    int window = 20; //согласовано с функцией(переделать)
    std::stack<std::pair<int, int>> state;
    int max_right = x;
    int min_left = x;
    int min_up = y;
    int max_down = y;
    state.push(std::make_pair(x, y));
    visit[x][y] = true;
    std::pair<int, int> current = {0, 0};
    std::vector<std::pair<int, int>> connected;
    connected.push_back(std::make_pair(x, y));
    int component_size = 1;
    while(!state.empty()) {
        std::pair<int, int> current = state.top();
        state.pop();
        max_right = std::max(max_right, current.first);
        min_left = std::min(min_left, current.first);
        min_up = std::min(min_up, current.second);
        max_down = std::max(max_down, current.second);
        for (int x_offset = -window; x_offset < window; ++x_offset) {
            for (int y_offset = -window; y_offset < window; ++y_offset) {
                    int new_x = current.first + x_offset;
                    int new_y = current.second + y_offset;
                    if (isEdge(edges_source.at<uchar>(new_x, new_y))) {
                        if (!visit[new_x][new_y]) {
                            ++component_size;
                            visit[new_x][new_y] = true;
                            connected.push_back(std::make_pair(new_x, new_y));
                            state.push(std::make_pair(new_x, new_y));
                        }
                    }
            }
        }
    }
    bool flag = true;
    if (max_right - min_left > 80 || max_down - min_up > 60 || component_size < 20) {
        flag = false;
    }
    return {flag, max_right, min_left, min_up, max_down};
}

//смотрим на всех соседей и пытаемся понять, кто самый правый и самый левый
void deleteErrors(cv::Mat edges_source) { //на вход -- одноканальные ребра
    int window = 20;
    cv::Mat edges;
    cv::Scalar value(0, 0, 0);
    cv::copyMakeBorder(edges_source, edges, window, window, window, window, cv::BORDER_CONSTANT, value);
    std::vector<Component> components;
    std::vector<std::vector<bool>> visit(edges.size[0], std::vector<bool>(edges.size[1], false));
    for (int i = 0; i < edges.size[0]; ++i) {
        for (int j = 0; j < edges.size[1]; ++j) {
            if (isEdge(edges.at<uchar>(i, j))) {
                edges_source.at<uchar>(i - window, j - window) = 0;
                if (!visit[i][j]) {
                    visit[i][j] = true;
                    bool flag = true;
                    int max_right = 0;
                    int min_left = 0;
                    int min_up = 0;
                    int max_down = 0;
                    std::tie(flag, max_right, min_left, min_up, max_down) = breadth_first_and_clear(edges, i, j, visit);
                    if (flag) {
                        components.push_back(Component(min_left, max_right, min_up, max_down));
                    }
                }
            }
        }
    }
    draw(edges_source, components, window);
}


int main() {
    //std::string path = "/home/ppadas/Documents/Find_filament/Compressed/";
    std::string path = "/home/ppadas/Documents/Find_filament/Adaptive/";
    //std::string file_names[] = {"M1.png", "M2.png", "M3.png", "M4.png", "M5.png", "M6.png", "M7.png", "M8.png"};
    std::string file_names[] = {"Example.png"};
    //std::string file_names[] = {"M3.png"};
    for (auto name : file_names) {
        cv::Mat image = cv::imread(path + name, cv::IMREAD_GRAYSCALE);
        cv::Mat input_color = cv::imread(path + name);
        cv::imwrite("./Result/" + name.substr(0, name.size() - 4) + std::string(".png") , input_color);
        cv::Mat dst;
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::dilate(image, dst, kernel);
        cv::Mat reverse; 
        cv::bitwise_not(dst, reverse);

        cv::Mat adaptive;
        cv::adaptiveThreshold(reverse, adaptive, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 21, 12);
        cv::Mat kernel2 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));

        //cv::imwrite("./Result/" + name.substr(0, name.size() - 4) + std::string("_2_adaptive.png"), adaptive);
        deleteErrors(adaptive);
        //cv::imwrite("./Result/" + name.substr(0, name.size() - 4) + std::string("_3_errors_adaptive.png"), adaptive);
        cv::Mat update_im = overlay(input_color, adaptive);
        cv::imwrite("./Result/" + name.substr(0, name.size() - 4) + std::string("_AdaptiveThreshold.png"), update_im);
    }
    return 0;
}