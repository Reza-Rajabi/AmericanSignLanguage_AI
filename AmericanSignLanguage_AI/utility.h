//
//  utility.h
//  AmericanSignLanguage_AI
//
//  Created by Reza Rajabi on 2020-03-08.
//  Copyright © 2020 RR. All rights reserved.
//

#ifndef utility_h
#define utility_h

#include <opencv2/opencv.hpp>
#include <fstream>
#include <sstream>
#include <future>

enum EXIT_CODE { ERR_ARG = 1, ERR_OPEN };

const int NUM_LABLE = 25;
const int NUM_FEATURE = 784;
const std::string windowName = "m x n random samples";
const double DISPLAY_SCALE = 2.0;

void openStream(const char* file, std::ifstream& in) {
    in.open(file, std::ifstream::in);
    if (in.fail()) {
        std::cout << "Couldn't open data file." << std::endl;
        exit(ERR_OPEN);
    }
}

int countRows(std::ifstream& in) {
    int counter = 0;
    std::string dispose;
    while(getline(in, dispose)) ++counter;
    in.close();
    return counter;
}

void loadData(const char* file, int& rows, cv::Mat& Y, cv::Mat& X) {
    std::ifstream in;
    // count rows and initialize matrix
    openStream(file, in);
    std::future<int> _rows(std::async(countRows, std::ref(in)));
    rows = _rows.get() - 1; /// ignore the first row (titles)
    Y = cv::Mat::zeros(rows, 1, CV_8U);
    X = cv::Mat::zeros(rows, NUM_FEATURE, CV_8U);
    
    // read rows and put data in matrix
    openStream(file, in);
    std::string line, single;
    getline(in, line);  /// DISPOSE the first line (titles)
    for(int r = 0; r < rows; r++) {
        getline(in, line);
        std::stringstream ss(line);
        for(int c = 0; c < NUM_FEATURE+1; c++) {
            getline(ss, single, ',');
            if (c == 0) Y.at<uchar>(r, 0) = std::atoi(single.c_str());
            else X.at<uchar>(r,c) = std::atoi(single.c_str());
        }
    }
    in.close();
}

cv::Mat getImageFromModelRow(const cv::Mat& row) {
    int rows = sqrt(NUM_FEATURE);
    cv::Mat image = row.reshape(1, rows); /// 1 chanel
    return image;
}

void displayImage(cv::Mat& image, double scale) {
    // DISPLAY with scale times of sample sizs
    cv::namedWindow(windowName);
    cv::resize(image, image, cv::Size(), scale, scale);
    // while space-bar key (ASCII 32 ) has not pressed show the image
    while(cv::waitKey(33) != 32) {
        cv::imshow(windowName, image);
    }
}

void display_nxm_random_samples_image(const cv::Mat& model, int n, int m) {
    std::vector<int> randomRowsOfModel;
    int sampleDim = sqrt(NUM_FEATURE);
    cv::RNG randomGenerator;
    for (int i = 0; i < n*m; i++) {
        randomRowsOfModel.push_back(randomGenerator.uniform(0, model.rows));
    }

    int imageRows = sampleDim * n + n - 1; /// n - 1 pixel padding between images
    int imageCols = sampleDim * m + m - 1; /// m - 1 pixel padding between images
    cv::Mat image = cv::Mat::zeros(imageRows, imageCols, CV_8U);
    int indexInRandomVect;
    for (int r = 0; r < imageRows; r++) {
        for (int c = 0; c < imageCols; c++) {
            if (r % (sampleDim+1) == 0 && r != 0) image.at<uchar>(r,c) = 0;
            else if (c % (sampleDim+1) == 0 && c != 0) image.at<uchar>(r,c) = 0;
            else {
                int padding_row = (r/sampleDim) % sampleDim * 1;
                int padding_col = (c/sampleDim) % sampleDim * 1;
                int noDimRow = r/sampleDim - padding_row;
                int noDimCol = c/sampleDim - padding_col;
                indexInRandomVect = noDimRow * m + noDimCol;
                std::cout << "rand ind: " << indexInRandomVect << std::endl;
                cv::Mat sample = getImageFromModelRow(model.row(randomRowsOfModel[indexInRandomVect]));
                int rowInSample = (r - padding_row) / m;
                int colInSample = (c - padding_col) / n;
                image.at<uchar>(r,c) = sample.at<uchar>(rowInSample, colInSample);
            }
        }
    }
    
    // DISPLAY with scale times of sample sizs
    displayImage(image, DISPLAY_SCALE);
}


#endif /* utility_h */
