//
//  predict.h
//  AmericanSignLanguage_AI
//
//  Created by Reza Rajabi on 2020-03-19.
//  Copyright © 2020 RR. All rights reserved.
//

#ifndef predict_h
#define predict_h

void predict(cv::Mat& X, cv::Mat& params, cv::Mat& Predict) {
    cv::Mat Theta[NUM_LAYER-1];
    rollTheta(Theta, params);
    
    int m = X.rows;
    cv::Mat ones = cv::Mat::ones(m, 1, CV_64F);
    cv::Mat h[NUM_LAYER];
    for(int i = 0; i < NUM_LAYER; i++) {
        h[i] = (i == 0) ? X : sigmoid(h[i-1] * Theta[i-1].t());
        if(i != NUM_LAYER-1) cv::hconcat(ones, h[i], h[i]);  /// don't add ones to last h[i]
    }
    
    Predict = h[NUM_LAYER-1];
}

void lablePredict(cv::Mat& Predict, double Threshold, cv::Mat& Lable) {
    int m = Predict.rows;
    Lable = cv::Mat::zeros(m, 1, CV_64F);
    int maxIndex = 0;
    double val, max;
    
    for (int r = 0; r < m; r++) {
        max = Predict.at<double>(r,0);
        maxIndex = 0;
        for (int c = 0; c < NUM_LABLE; c++) {
            val = Predict.at<double>(r,c);
            if (val > max) {
                maxIndex = c;   /// maxIndex is the lable for row r
                max = val;
            }
        }
        /// we don have lable 9=J in predicts, so we need to plus one from index 9 to map 0 to 24 into 25 predicts
        /// we have considered that, when we wanted to make binary rows from lables in cost function
        if (max < Threshold) maxIndex = -1; /// labled as NOT BELONG
        else if (maxIndex > 8) ++maxIndex;
        Lable.at<double>(r,0) = maxIndex;
    }
}

#endif /* predict_h */
