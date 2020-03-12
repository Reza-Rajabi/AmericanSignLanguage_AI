//
//  cost.h
//  AmericanSignLanguage_AI
//
//  Created by Reza Rajabi on 2020-03-11.
//  Copyright © 2020 RR. All rights reserved.
//

#ifndef cost_h
#define cost_h

#include <opencv2/opencv.hpp>
                                                        // layers length (L layers)
const int IN_SIZE = 784;        /// same as NUM_FEATURE                 n
const int HIDEN1_SIZE = 28;                                 //s1
const int HIDEN2_SIZE = 37;                                 //s2
const int HIDEN3_SIZE = 28;                                 //s3
const int OUT_SIZE = 25;        /// same as NUM_LABLE                  //  k



cv::Mat sigmoid(const cv::Mat& Z) {
    cv::Mat G(Z);  /// copy
    for(int r = 0; r < Z.rows; r++) {
        for(int c = 0; c < Z.cols; c++) {
            double z = Z.at<uchar>(r,c);
            G.at<uchar>(r,c) = 1.0 / (1.0 + exp(-z));
        }
    }
    return G;
}

cv::Mat sigmoidGradient(const cv::Mat& Z) {
    cv::Mat G = cv::Mat::zeros(Z.size(), Z.type());
    G = sigmoid(Z).mul(1 - sigmoid(Z));
    return G;
}

cv::Mat log(const cv::Mat& M) {
    cv::Mat logM;
    cv::log(M, logM);
    return logM;
}

void initialParameters(cv::Mat& params) {}

void minimizeCost(const cv::Mat& params,    /// initial parameters in a a rolled up vector
                  const cv::Mat& X,         /// featurs
                  const cv::Mat& Y,         /// lables
                  long double lambda,       /// regulaization parameter
                  long double& J,           /// cost to return
                  cv::Mat& gradient) {      /// gradient to return (partial derivative of cost func.)
    // NOTE: - extracting Theta from vertival vector
    /// the dimention of each Thetha is:  the next layer size x its layer size
    cv::Range range0(0, HIDEN1_SIZE * (IN_SIZE+1));
    cv::Mat Theta0 = params(range0, cv::Range::all());
    Theta0.resize(HIDEN1_SIZE, IN_SIZE+1);              /// Theta0 --> s1 x (n+1)
    
    cv::Range range1(range0.end, range0.end + HIDEN2_SIZE * (HIDEN1_SIZE+1));
    cv::Mat Theta1 = params(range1, cv::Range::all());
    Theta1.resize(HIDEN2_SIZE, HIDEN1_SIZE+1);          /// Theta1 --> s2 x (s1+1)
    
    cv::Range range2(range1.end, range1.end + HIDEN3_SIZE * (HIDEN2_SIZE+1));
    cv::Mat Theta2 = params(range2, cv::Range::all());
    Theta2.resize(HIDEN3_SIZE, HIDEN2_SIZE+1);          /// Theta2 --> s3 x (s2+1)
    
    cv::Range range3(range2.end, range2.end + OUT_SIZE * (HIDEN3_SIZE+1));
    cv::Mat Theta3 = params(range3, cv::Range::all());
    Theta3.resize(OUT_SIZE, HIDEN3_SIZE+1);             /// Theta3 --> k x (s3+1)
    
    
    // NOTE: - feeding forward and calculating activation parameters
    int m = X.rows;
    cv::Mat ones = cv::Mat::ones(m, 1, CV_8U);
    
    cv::Mat a0(X);                              /// X  --> m x n
    cv::hconcat(ones, a0, a0);                  /// a0 --> m x (n+1)
    
    cv::Mat a1 = sigmoid(a0 * Theta0.t());      ///( m x (n+1) ) * ( (n+1) x s1 )
    cv::hconcat(ones, a1, a1);                  /// a1 --> m x (s1+1)
    
    cv::Mat a2 = sigmoid(a1 * Theta1.t());      ///( m x (s1+1) ) * ( (s1+1) x s2 )
    cv::hconcat(ones, a2, a2);                  /// a2 --> m x (s2+1)
    
    cv::Mat a3 = sigmoid(a2 * Theta2.t());      ///( m x (s2+1) ) * ( (s2+1) x s3 )
    cv::hconcat(ones, a3, a3);                  /// a3 --> m x (s3+1)
    
    /// this is is hypothesis
    cv::Mat a4 = sigmoid(a3 * Theta3.t());      ///( m x (s3+1) ) * ( (s3+1) x k)  --> m x k


    // NOTE: - changing each lable of lables to a vector of (0s and 1)
    cv::Mat Y_ = cv::Mat::zeros(m, NUM_LABLE, CV_8U);
    for (int i = 0; i < m; i++) {
        int columnForOne = Y.at<uchar>(i,1);
        Y_.at<uchar>(i, columnForOne) = 1;
    }
    
    
    // NOTE: - calculating cost J
    /// Y_.t() -->  k x m           a4  --> m x k     ----> sum over a k x k matrice
    /// openCV's Mat has scalar elements of potentialy up to 4 color chanel
    /// but here we have use Mat as matrice, so we only want the first chanel
    cv::Scalar j = -1/m * cv::sum( Y_.t() * log(a4) + (1 - Y_.t()) * log(1 - a4) );
    J = j[0];       /// we only want the first chanel of the Mat matrice
    
    
    // NOTE: - backpropagation to calcute gradients
    cv::Mat Theta0_ = Theta0.colRange(2, Theta0.cols);
    cv::Mat Theta1_ = Theta1.colRange(2, Theta1.cols);
    cv::Mat Theta2_ = Theta2.colRange(2, Theta2.cols);
    cv::Mat Theta3_ = Theta3.colRange(2, Theta3.cols);
    
    cv::Mat delta4 = a4 - Y_;
    cv::Mat delta3 = (delta4 * Theta3_).mul(sigmoidGradient(a2 * Theta2.t()));
    cv::Mat delta2 = (delta3 * Theta2_).mul(sigmoidGradient(a1 * Theta1.t()));
    cv::Mat delta1 = (delta2 * Theta1_).mul(sigmoidGradient(a0 * Theta0.t()));
    
    cv::Mat DELTA4 = delta4.t() * a3;
    cv::Mat DELTA3 = delta3.t() * a2;
    cv::Mat DELTA2 = delta2.t() * a1;
    cv::Mat DELTA1 = delta1.t() * a0;

    cv::Mat Theta0_g = 1/m * cv::Mat(DELTA1);
    cv::Mat Theta1_g = 1/m * cv::Mat(DELTA2);
    cv::Mat Theta2_g = 1/m * cv::Mat(DELTA3);
    cv::Mat Theta3_g = 1/m * cv::Mat(DELTA4);

    
    // NOTE: - regulization of the cost to prevent overfitting

    // NOTE: - unroll gradients

        
}


#endif /* cost_h */
