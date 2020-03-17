//
//  tester.cpp
//  AmericanSignLanguage_AI
//
//  Created by Reza Rajabi on 2020-03-17.
//  Copyright © 2020 RR. All rights reserved.
//

#include <opencv2/opencv.hpp>
#include <thread>

#include "utility.h"
#include "train.h"


#define CASE 1
/// -CASE 1 : loads data, displays a sample and tests train function
/// -CASE 2 : test sigmoid and sigmoidPrime and log
/// -CASE 3 : test random initializer
/// -CASE 4 : test roll and unroll functions
/// -CASE 5 : test hconcat function of opencv


int main(int argc, char* argv[]) {
    // NOTE: -CASE 1 : loads data, displays a sample and tests train function
    /// THIS IS A GENERAL TEST OF: loading data, calculating cost, and training
    if (CASE == 1) {
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
        
        
        display_nxm_random_samples_image(test_X, 10, 20);
        
        
        // Testing train function
        cv::Mat Theta_unroll[NUM_LAYER-1];
        for(int i = 0; i < NUM_LAYER-1; i++) {
            Theta_unroll[i] = initializeLayerParameters(S[i], S[i+1]);
        }
        cv::Mat Theta;
        rollTheta(Theta_unroll, Theta);
        cv::Mat J_history;
        train(train_X, train_Y, Theta, J_history);
        /// outputs J_history on a csv file. J_history should decrement consistantly to about zero
    }
    
    
    // TESTING evalFun
    /// tested in milestone2
    
    
    // NOTE: -CASE 2 : test sigmoid and sigmoidPrime and log
    if (CASE == 2) {
        std::vector<double> data {-1, -0.5, 0, 0.5, 1};
        cv::Mat ex(data, CV_64F);
        std::cout <<
        sigmoidPrime(ex) << std::endl; // values between 0 to 0.25 (at zero)
        std::cout << log(ex) << std::endl; // values nan, nan, -inf, -0.6, 0 (at 1)
    }
    
    
    // NOTE: -CASE 3 : test random initializer
    if (CASE == 3) {
        for (int i = 0; i < NUM_LAYER-1; i++) {
            cv::Mat init = initializeLayerParameters(S[i], S[i+1]);
            std::cout << init.size() << std::endl;
            std::cout << "a value:" << init.at<double>(20,20) << std::endl;
            std::cout << std::endl;
        }
    }
    
    
    // NOTE: -CASE 4 : test roll and unroll functions
    /// this test has a conditional embeded test
    if (CASE == 4) {
        cv::Mat unrolled[4];
        double d0[] {1,2,3,4,5,6,7,8,9};
        unrolled[0] = cv::Mat(3,3,CV_64F,d0);
        std::cout << unrolled[0] << std::endl;
        double d1[] {10,12,13,14};
        unrolled[1] = cv::Mat(2,2,CV_64F,d1);
        std::cout << unrolled[1] << std::endl;
        double d2[] {11,15,16,17,18,19};
        unrolled[2] = cv::Mat(3,2,CV_64F,d2);
        std::cout << unrolled[2] << std::endl;
        double d3[] {111,115,146,147,168,159};
        unrolled[3] = cv::Mat(2,3,CV_64F,d3);
        std::cout << unrolled[3] << std::endl;
        
        cv::Mat rolled;
        rollTheta(unrolled, rolled);
        std::cout << rolled << std::endl;
        
        // for this part: temporary needs to change S[] = {1,2,2,2,3}
        // then, from 25 rows makes S[l+1] x (S[l]+1) --> (2x2)+(2x3)+(2x3)+(3x3) matrice
        bool makeSureChanged_S = false;
        if (makeSureChanged_S) {
            unrollTheta(unrolled, rolled);
            for(int i = 0; i < 4; i++) {
            std::cout << unrolled[i] << std::endl;
            }
        }
    }
    
    
    // NOTE: -CASE 5 : test hconcat function of opencv
    if (CASE == 5) {
        double d[] {111,115,146,147,168,159};
        cv::Mat right = cv::Mat(2,3,CV_64F,d);
        cv::Mat ones = cv::Mat::ones(right.rows, 1,CV_64F);
        cv::hconcat(ones, right, right);
        std::cout << right << std::endl;
    }
    
    
    std::cout << std::endl;
    return 0;
}
