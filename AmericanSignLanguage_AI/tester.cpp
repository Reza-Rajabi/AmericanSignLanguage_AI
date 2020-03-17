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
#include "test.h"


int main(int argc, char* argv[]) {
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
    
    
//    display_nxm_random_samples_image(test_X, 10, 20);
    
    
    // TESTING evalFun
    /// tested in milestone2
    
    
    // Testing train function
//    cv::Mat Theta_unroll[NUM_LAYER-1];
//    for(int i = 0; i < NUM_LAYER-1; i++) {
//        Theta_unroll[i] = initializeLayerParameters(S[i], S[i+1]);
//    }
//    cv::Mat Theta;
//    rollTheta(Theta_unroll, Theta);
//    cv::Mat J_history;
//    train(train_X, train_Y, Theta, J_history);
    /// outputs J_history on a csv file. J_history should decrement consistantly to about zero
    
    
    
    std::cout << std::endl;
    return 0;
}
