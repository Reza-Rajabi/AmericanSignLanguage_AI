//
//  Milestone2.cpp
//  AmericanSignLanguage_AI
//
//  Created by Reza Rajabi on 2020-03-08.
//  Copyright © 2020 RR. All rights reserved.
//

#include <opencv2/opencv.hpp>
#include <thread>

#include "utility.h"
#include "cost.h"



int _main_(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "Data files names are not provided." << std::endl;
        exit(ERR_ARG);
    }
    
    
    int train_rows, test_rows;
    cv::Mat train_Y, test_Y; /// LABLES
    cv::Mat train_X, test_X; /// FEATURES
    
    
    std::thread setupTrain(loadData,argv[1], std::ref(train_rows), std::ref(train_Y),std::ref(train_X));
    std::thread setupTest(loadData,argv[2], std::ref(test_rows), std::ref(test_Y),std::ref(test_X));

    
    setupTrain.join();
    setupTest.join();
    
    
    // TESTING evalFun
    /// first: we should get 100% TP if we test test_Y against itself
    std::cout << "\nTest lables against lables:" << std::endl;
    /// changing test_Y to a m x NUM_LABLE matrix copying the logic we have inside cost function in cost.h at line 105
    int m = test_Y.rows;
    cv::Mat Y_ = cv::Mat::zeros(m, NUM_LABLE, CV_64F);
    int columnForOne;
    for (int i = 0; i < m; i++) {
        columnForOne = test_Y.at<double>(i,0);
        /// we don't have lable 9=J in dataset, so we need to minus one from index 9 to match lables 0 to 24 in 24 cols
        /// and we need to consider to add one later on, when we want to translate a binary row in Y_ to a lable in Y
        if (columnForOne >= 9) --columnForOne;
        Y_.at<double>(i, columnForOne) = 1;  /// other columns of the row i  is zero
    }
    double PRF[3] {0.0};
    bool works = evalFun(Y_, Y_, 1.0, 0.5, PRF);
    if (works)
        std::cout << "P: " << PRF[0] << " R: " << PRF[1] << " F1: " << PRF[2] << std::endl;
    
    /// second: we probably shouldn't get an acceptable results for a random predict
    /// there are no Negative value in the test data so TN will be zero, FN barely happens unless for a big thershold (ex. 0.8)
    std::cout << "\nTest random predicts against lables:" << std::endl;
    cv::RNG randomGenerator;
    cv::Mat Rand_Predict = cv::Mat::zeros(m, NUM_LABLE, CV_64F);
    for (int r = 0; r < m; r++) {
        for (int c = 0; c < NUM_LABLE; c++) {
            Rand_Predict.at<double>(r,c) = randomGenerator.uniform(0.0, 1.0);
        }
    }
    /// we make a bigger thershold to have chance for FN
    works = evalFun(Rand_Predict, Y_, 1.0, 0.8, PRF);
    if (works)
        std::cout << "P: " << PRF[0] << " R: " << PRF[1] << " F1: " << PRF[2] << std::endl;
    
    std::cout << std::endl;
    return 0;
}
