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
#include "predict.h"


/// counter on terminal
#define TERMINAL
/// load what it has learned so far
#define LOAD_PERCEPTION
/// keep training
//#define TRAIN

/// test scenarios
#define CASE1 1
#define CASE2 3
// Each test case may have its acceptence criteria commented
/// -CASE 0 : off two simultanous scenarios
/// -CASE 1 : tests loading data
/// -CASE 2 : tests displays function                                         ---> needs CASE 1
/// -CASE 3 : tests train function, MOST GENERAL TEST SCENARIO                ---> needs CASE 1
/// -CASE 4 : tests backpropagation algorithm in cost function                ---> needs CASE 1
/// -CASE 5 : tests sigmoid and sigmoidPrime and log
/// -CASE 6 : tests random initializer
/// -CASE 7 : tests roll and unroll functions with an embeded condition
/// -CASE 8 : tests hconcat function of opencv
/// -CASE 9 : test lable function                                             ---> needs CASE 1



int main(int argc, char* argv[]) {
    int train_rows, test_rows;
    cv::Mat train_Y, test_Y; /// LABLES
    cv::Mat train_X, test_X; /// FEATURES
    
    // MARK: -CASE 1 : test loading data
    if (CASE1 == 1 || CASE2 == 1) {
        if (argc < 3) {
            std::cout << "Data files names are not provided." << std::endl;
            exit(ERR_ARG);
        }
        std::cout << "Loading training and test data:" << std::endl;
        std::thread setupTrain(loadData,argv[1], std::ref(train_rows), std::ref(train_Y),std::ref(train_X));
        std::thread setupTest(loadData,argv[2], std::ref(test_rows), std::ref(test_Y),std::ref(test_X));
        
        setupTrain.join();
        setupTest.join();
        std::cout << std::endl;
    }
        
        
    // MARK: -CASE 2 : tests displays function
    if (CASE1 == 2 || CASE2 == 2) {
//        display_nxm_random_samples_image(test_X, 10, 20);
        cv::Mat image = getImageFromModelRow(test_X.row(15));
        displayImage(image, 2.0, windowName_one);
    }
        
        
    // MARK: -CASE 3 : tests train function
    if (CASE1 == 3 || CASE2 == 3) {
        cv::Mat Theta;
    #ifdef LOAD_PERCEPTION
        std::ifstream ifs;
        std::cout << "Loading learned parameters:" << std::endl;
        openStream("learnedParams-256-iter700.csv", ifs);
        int theta_rows = countRows(ifs);
        Theta = cv::Mat(theta_rows, 1, CV_64F);
        openStream("learnedParams-256-iter700.csv", ifs);
        if (ifs.is_open()) {
            std::string num;
            for (int r = 0; r < theta_rows; r++) {
                std::getline(ifs, num);
                Theta.at<double>(r,0) = std::atof(num.c_str());
            }
        }
        std::cout << theta_rows << " learned parameters loaded." << std::endl;
        std::cout << std::endl;
    #else
        cv::Mat Theta_roll[NUM_LAYER-1];
        for(int i = 0; i < NUM_LAYER-1; i++) {
            Theta_roll[i] = initializeLayerParameters(S[i], S[i+1]);
        }
        unrollTheta(Theta_roll, Theta);
    #endif
    #ifdef TRAIN
        cv::Mat J_history;
        cv::Mat X;
        Activation ac0;
        ac0.f = swish;
        ac0.fPrime = swishPrime;
        normalize(train_X);
        //train(ac0, train_X, train_Y, Theta, J_history);
        train(AC, train_X, train_Y, Theta, J_history);
        /// outputs J_history on a csv file. J_history should decrement consistantly to about zero
    #endif
        normalize(test_X);
        cv::Mat Predict;
        predict(AC, test_X, Theta, Predict);
        //std::cout << Theta << std::endl;
        
        std::cout << "Test predicts against lables:" << std::endl;
        /// changing test_Y to a m x NUM_LABLE matrix
        cv::Mat Y_;
        makeBinaryLables(test_Y, Y_);
        
        double PRF[3] {0.0};
        bool works = evalFun(Predict, Y_, BETA, THRESHOLD, PRF);
        if (works)
            std::cout << "P: " << PRF[0] << " R: " << PRF[1] << " F1: " << PRF[2] << std::endl << std::endl;

        cv::Mat Lable;
        lablePredict(Predict, THRESHOLD, Lable);
        cv::hconcat(test_Y, Lable, Lable);
        std::cout << "lables, predicts" << std::endl;
        std::cout << Lable << std::endl;
    }
        
        
    // MARK: -CASE 4 : test backpropagation algorithm in cost function
    if (CASE1 == 4 || CASE2 == 4) {
        double eps = 1e-4, J_plus, J_minus;
        const int ROWS = 100;
        cv::Mat temp, T_[NUM_LAYER-1], T, T_prime;
        cv::Mat X = train_X.rowRange(0, ROWS);
        cv::Mat Y = train_Y.rowRange(0,ROWS);
        for(int i = 0; i < NUM_LAYER-1; i++) {
            T_[i] = initializeLayerParameters(S[i], S[i+1]);
        }
        unrollTheta(T_, T);
        cv::Mat gradApproximate = cv::Mat::zeros(T.rows, 1, CV_64F);
        cv::Mat d = cv::Mat::zeros(T.rows, 1, CV_64F);
        costFunction(AC, T, X, Y, LAMBDA, J_plus, T_prime);
        for (int i = 0; i < T.rows; i++) {
        #ifdef TERMINAL
            std::cout << '\r' << "Testing Backpropagation: " << i+1 << std::flush;
        #endif

            d.at<double>(i,0) = eps;
            costFunction(AC, (T + d), X, Y, LAMBDA, J_plus, temp);
            costFunction(AC, (T - d), X, Y, LAMBDA, J_minus, temp);
            gradApproximate.at<double>(i,0) = (J_plus - J_minus)/(2.0*eps);
            d.at<double>(i,0) = 0;
        }
        std::cout << std::endl;
        double diff = cv::norm(gradApproximate - T_prime)/cv::norm(gradApproximate + T_prime);
        std::cout << diff << std::endl; /// should get very small number
        cv::hconcat(gradApproximate, T_prime, gradApproximate);
        std::cout << gradApproximate << std::endl; /// should get 2 almost similar columns of result
    }
    
    
    // TESTING evalFun
    /// tested in milestone2
    
    
    // MARK: -CASE 5 : test sigmoid and sigmoidPrime and log
    if (CASE1 == 5 || CASE2 == 5) {
        std::vector<double> data {-1, -0.5, 0, 0.5, 1, 2};
        cv::Mat ex(data, CV_64F);
        std::cout << sigmoid(ex) << std::endl; // 0.5 at zero
        std::cout << sigmoidPrime(ex) << std::endl; // values between 0 to 0.25 (0.25 at zero)
        std::cout << log(ex) << std::endl; // values nan, nan, -inf, -0.7, 0, 0.7 (0 at 1)
        std::cout << log(0.5) << std::endl; // log --> ln  and log10 is base 10
    }
    
    
    // MARK: -CASE 6 : test random initializer
    if (CASE1 == 6 || CASE2 == 6) {
        for (int i = 0; i < NUM_LAYER-1; i++) {
            cv::Mat init = initializeLayerParameters(S[i], S[i+1]);
            std::cout << init.size() << std::endl;
            std::cout << "a value:" << init.at<double>(20,20) << std::endl;
            std::cout << std::endl;
        }
    }
    
    
    // MARK: -CASE 7 : test roll and unroll functions
    /// this test has a conditional embeded test that needs to setup manually
    if (CASE1 == 7 || CASE2 == 7) {
        cv::Mat rolled[4];
        double d0[] {1,2,3,4,5,6,7,8,9};
        rolled[0] = cv::Mat(3,3,CV_64F,d0);
        std::cout << rolled[0] << std::endl;
        double d1[] {10,12,13,14};
        rolled[1] = cv::Mat(2,2,CV_64F,d1);
        std::cout << rolled[1] << std::endl;
        double d2[] {11,15,16,17,18,19};
        rolled[2] = cv::Mat(3,2,CV_64F,d2);
        std::cout << rolled[2] << std::endl;
        double d3[] {111,115,146,147,168,159};
        rolled[3] = cv::Mat(2,3,CV_64F,d3);
        std::cout << rolled[3] << std::endl;
        
        cv::Mat unrolled;
        unrollTheta(rolled, unrolled);
        std::cout << unrolled << std::endl;
        
        // for this part: temporary needs to change S[] = {1,2,2,2,3}
        // then, from 25 rows makes S[l+1] x (S[l]+1) --> (2x2)+(2x3)+(2x3)+(3x3) matrice
        bool makeSureChanged_S = false;
        if (makeSureChanged_S) {
            rollTheta(rolled, unrolled);
            for(int i = 0; i < 4; i++) {
                std::cout << rolled[i] << std::endl;
            }
        }
    }
    
    
    // MARK: -CASE 8 : test hconcat function of opencv
    if (CASE1 == 8 || CASE2 == 8) {
        double d[] {111,115,146,147,168,159};
        cv::Mat right = cv::Mat(2,3,CV_64F,d);
        cv::Mat ones = cv::Mat::ones(right.rows, 1,CV_64F);
        cv::hconcat(ones, right, right);
        std::cout << right << std::endl;
    }
    
    
    // MARK: -CASE 9 : test lable function
    if (CASE1 == 9 || CASE2 == 9) {
        /// should get identical result if we lable the test_Y
        cv::Mat Y_;
        makeBinaryLables(test_Y, Y_);
        //cv::hconcat(test_Y, Y_, Y_);
        //std::cout << Y_ << std::endl;
        cv::Mat Lable;
        lablePredict(Y_, 0.0, Lable);
        int countErr = 0;
        for (int r = 0; r < Lable.rows; r++) {
            double l = Lable.at<double>(r,0);
            double y = test_Y.at<double>(r,0);
            if (l != y) {
                std::cout << "at " << r << ": "<< y << ", " << l << std::endl;
                countErr++;
            }
        }
        std::cout << "errors: " << countErr << std::endl;
    }
    
    
    std::cout << std::endl;
    return 0;
}
