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

const int NUM_FEATURE = 784; /// each row of data includes lables and the gray scale value of the 28x28 = 784 pixels
const std::string windowName_mxn = "m x n random samples";
const std::string windowName_one = "sample";
const double DISPLAY_SCALE = 2.0;
const double THRESHOLD = 0.5;
const double BETA = 1.0;

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
    return --counter; /// dispose the title
}

void loadData(const char* file, int& rows, cv::Mat& Y, cv::Mat& X) {
    std::ifstream in;
    // count rows and initialize matrix
    openStream(file, in);
    std::future<int> _rows(std::async(countRows, std::ref(in)));
    rows = _rows.get();
    std::cout << "file: " << std::setw(20) << std::left;
    std::cout << file << "\t rows: " << rows << std::endl;
    Y = cv::Mat::zeros(rows, 1, CV_64F);
    X = cv::Mat::zeros(rows, NUM_FEATURE, CV_64F);
    
    // read rows and put data in matrix
    openStream(file, in);
    std::string line, single;
    getline(in, line);  /// DISPOSE the first line (titles)
    for(int r = 0; r < rows; r++) {
        getline(in, line);
        std::stringstream ss(line);
        for(int c = 0; c < NUM_FEATURE+1; c++) {
            getline(ss, single, ',');
            if (c == 0) Y.at<double>(r, 0) = std::atof(single.c_str());
            else X.at<double>(r,c-1) = std::atof(single.c_str());
        }
    }
    in.close();
}

void filter(const cv::Mat& X, const cv::Mat& Y, std::set<int> lables, cv::Mat& X_filtered, cv::Mat& Y_filtered) {
    for (int r = 0; r < Y.rows; r++) {
        double lable = Y.at<double>(r,0);
        if (lables.find(lable) != lables.end()) {
            Y_filtered.at<double>(r,0) = lable;
            X_filtered.row(r) = X.row(r).clone();
        }
    }
}

void reduceFeatures(const cv::Mat& X, int features, cv::Mat& X_reduced) {
    int reduced_rows = sqrt(features);
    int original_rows = sqrt(NUM_FEATURE);
    cv::Mat temp = cv::Mat::zeros(reduced_rows,reduced_rows, CV_64F);
    cv::Mat original = cv::Mat::zeros(original_rows,original_rows, CV_64F);
    for (int r = 0; r < X.rows; r++) {
        (X.row(r)).reshape(0,original_rows).copyTo(original);
        cv::resize(original, temp, temp.size(), cv::INTER_AREA);
        X_reduced.row(r) = temp.reshape(0,1).clone();
    }
}

cv::Mat getImageFromModelRow(const cv::Mat& row) {
    int rows = sqrt(row.cols);
    cv::Mat image = row.reshape(0, rows);
    return image;
}

void displayImage(cv::Mat& image, double scale, std::string windowName) {
    cv::resize(image, image, cv::Size(), scale, scale);
    // while space-bar key (ASCII 32 ) has not pressed show the image
    cv::namedWindow(windowName);
    while(cv::waitKey(33) != 32) {
        cv::imshow(windowName, image);
    }
}

void display_nxm_random_samples_image(const cv::Mat& model, int nHeight, int mWidth ) {
    std::vector<int> randomRowsOfModel;
    int sampleDim = sqrt(NUM_FEATURE);
    cv::RNG randomGenerator;
    for (int i = 0; i < (nHeight * mWidth); i++) {
        randomRowsOfModel.push_back(randomGenerator.uniform(0, model.rows));
    }

    int imageRows = sampleDim * nHeight + nHeight - 1; /// n - 1 pixel padding between images
    int imageCols = sampleDim * mWidth + mWidth - 1; /// m - 1 pixel padding between images
    cv::Mat image = cv::Mat::zeros(imageRows, imageCols, CV_8U);
    int indexInRandomVect = -1;
    cv::Mat sample;
    for (int r = 0; r < imageRows; r++) {
        for (int c = 0; c < imageCols; c++) {
            /// if it is padding, make it black
            if (r % (sampleDim+1) == 0 && r != 0) image.at<uchar>(r,c) = 0;
            else if (c % (sampleDim+1) == 0 && c != 0) image.at<uchar>(r,c) = 0;
            /// otherwise colorize based on samples
            else {
                /// calculate the num of padding pixels so far
                int padding_rows = (r+1)/(sampleDim+1);
                int padding_cols = (c+1)/(sampleDim+1);
                /// calculate the index of sample from 0 to (nHeight * mWidth)
                int new_index = ((r - padding_rows)/sampleDim) * nHeight + (c - padding_cols)/sampleDim;
                /// if index changed, find the sample image; otherwise use the sample that already have
                if (new_index != indexInRandomVect) {
                    indexInRandomVect = new_index;
                    sample = getImageFromModelRow(model.row(randomRowsOfModel[indexInRandomVect]));
                }
                /// colorize the pixel at (r, c) similar to the pixel at (rowInSample, colInSample) of sample
                int rowInSample = r - indexInRandomVect / nHeight * sampleDim - padding_rows;
                int colInSample = c - (indexInRandomVect % mWidth) * sampleDim - padding_cols;
                image.at<uchar>(r,c) = (uchar) sample.at<double>(rowInSample, colInSample);
            }
        }
    }
    
    // scale and DISPLAY image
    displayImage(image, DISPLAY_SCALE, windowName_mxn);
}

// evaluates the accuracy of the predicts
/// Test: a matrix of m x NUM_LABLE (look at `cost.h`) representing m number of rows, each row is a test lable (answer),
/// and NUM_LABLE columns consist of all 0 but only one 1 which the index of 1 represents the lable number.
/// Predict: a matrix of m x NUM_LABLE representing m rows of predicted values in NUM_LABLE columns having a
/// value between 0 and 1 which then the index of maximum value in a row represents the lable number for that row
/// PRF: an array of three elements consist of Precision, Recall and F_beta parameters, in order, based on confusion matrix
bool evalFun(const cv::Mat& Predict, const cv::Mat& Test, double beta, double threshold, double* PRF) {
    /*
     * TP -> has a max value in the Predict row at the same index that Test has 1 and
     *       that max value is grater than threshold
     * FP -> has a max value in the Predict row at the diff index that Test has 1 and
     *       that max value is grater than threshold
     * TN -> has a max value in the Predict which is less than threshold while Test
     *       doesn't have any 1 value for that row
     * FN -> has a max value in the Predict which is less than threshold while Test
     *       has a 1 value at some index for that row
     */
    if (threshold < 0 || threshold > 1) {
        std::cout << "Threshold must be in [0,1]" << std::endl;
        return false;
    }
    if (Predict.rows != Test.rows || Predict.cols != Test.cols) {
        std::cout << "Number of tests and predicts doesn't match" << std::endl;
        return false;
    }
    // defining parameters of confusion matrix
    double TP = 0.0, TN = 0.0, FP = 0.0, FN = 0.0;
    
    int indexOf_one_inTestRow;
    int indexOf_max_inPredRow;
    double maxOfPredRow;
    
    for (int r = 0; r < Test.rows; r++) {
        indexOf_one_inTestRow = -1;
        indexOf_max_inPredRow = -1;
        maxOfPredRow = 0;
        for (int c = 0; c < Test.cols; c++) {
            if (Test.at<double>(r,c) == 1.0) indexOf_one_inTestRow = c;
            if (Predict.at<double>(r,c) > maxOfPredRow) {
                maxOfPredRow = Predict.at<double>(r,c);
                indexOf_max_inPredRow = c;
            }
        }
        // one row has been evaluated. setup confusion matrix parameters for that row
        if (maxOfPredRow > threshold && indexOf_max_inPredRow == indexOf_one_inTestRow)
            TP++; /// true predict indicating the class sample belongs to
        else if (maxOfPredRow > threshold && indexOf_max_inPredRow != indexOf_one_inTestRow) FP++; /// false predict indicating some class that sample mistakely belongs to
        else if (maxOfPredRow < threshold && indexOf_one_inTestRow == -1)
            TN++; /// true predict indicating that sample doesn,t belong to any group
        else if (maxOfPredRow < threshold && indexOf_one_inTestRow > -1)
            FN++; /// false predict indicating that sample doesn,t belong to any group while it does
        else {
            std::cout << "something is wrong at row " << r << std::endl;
            return false;
        }
        
    }
    
    std::cout << "TP: " << TP << " FP: " << FP << " TN: " << TN << " FN: " << FN << std::endl;
    
    if (TP + FN == 0) {
        std::cout << "There were no positive cases in the input data!" << std::endl;
    }
    else if (TP + FP == 0) {
        std::cout << "None of the cases has been predicted positive!" << std::endl;
    }
    else {
        PRF[0] = (double) TP/(TP + FP); /// Precision
        PRF[1] = (double) TP/(TP + FN); /// Recall
        
        double b = pow(beta,2);
        PRF[2] = (1 + b) * PRF[0] * PRF[1] / (b * PRF[0] + PRF[1]); /// F_beta
    }
    
    return true;
}


#endif /* utility_h */
