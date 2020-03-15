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
#include "train.h"


/// each row of data includes lables and the gray scale value of the 28x28 = 784 pixels


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
    
    
    display_nxm_random_samples_image(test_X, 10, 20);
    
    
    return 0;
}

