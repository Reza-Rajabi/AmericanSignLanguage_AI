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
                                                        // each layer length (L layers)
const int IN_SIZE = 784;        /// same as NUM_FEATURE                 n
const int HIDEN1_SIZE = 28;     ///                                     s1
const int HIDEN2_SIZE = 28;     ///                                     s2
const int HIDEN3_SIZE = 28;     ///                                     s3
const int OUT_SIZE = 24;        /// same as NUM_LABLE                   k

const int NUM_LABLE = 24; /// labeles in (0-25) mapping letter A-Z, but no lable for 9=J or 25=Z because of gesture motions.

const int NUM_LAYER = 5;
const double lambda = 2.0;      /// the regulization factor value to prevent overfitting

const int S[NUM_LAYER] = { IN_SIZE, HIDEN1_SIZE, HIDEN2_SIZE, HIDEN3_SIZE, OUT_SIZE }; ///layers size



cv::Mat sigmoid(const cv::Mat& Z) {
    cv::Mat G;
    cv::exp(Z, G);
    return 1.0 / (1.0 + G);
}

cv::Mat sigmoidPrime(const cv::Mat& Z) {
    cv::Mat G = sigmoid(Z);
    return G.mul(1 - G);
}

cv::Mat log(const cv::Mat& M) {
    cv::Mat logM;
    cv::log(M, logM);
    return logM;
}

cv::Mat initializeLayerParameters(int inConnections, int outConnections) {
    // NOTE: - randomly initializes parameters for a layer with inConnections number of
    ///        incomming connections and outConnections number of outcomming connections
    ///        it returns a matrice of size (outConnections, 1 + inConnections)
    double epsilon = sqrt(6)/(outConnections+inConnections+1);
    cv::Mat weights = cv::Mat::zeros(outConnections, 1 + inConnections, CV_64F);
    cv::RNG randomGenerator;
    for(int r = 0; r < outConnections; r++) {
        for (int c = 0; c < 1+inConnections; c++) {
            weights.at<double>(r,c) = randomGenerator.uniform((double)0.0, (double)1.0) * 2 * epsilon - epsilon;
        }
    }
    
    return weights;
}

void unrollTheta(cv::Mat unrolled[], const cv::Mat& rolled) {
    /// the dimention of each Thetha is:  the next layer size x its layer size
     /// Theta0 --> s1 x (n+1)
     /// Theta1 --> s2 x (s1+1)
     /// Theta2 --> s3 x (s2+1)
     /// Theta3 --> k  x (s3+1)
     int rangeStart = 0, rangeEnd = 0;
     for (int l = 0; l < NUM_LAYER-1; l++) {
         rangeEnd += S[l+1] * (S[l]+1);
         cv::Range range(rangeStart, rangeEnd);
         cv::Mat temp = rolled(range, cv::Range::all());
         unrolled[l] = cv::Mat(S[l+1], S[l]+1, CV_64F, temp.data);
         rangeStart = rangeEnd;
     }
}

void rollTheta(cv::Mat unrolled[], cv::Mat& rolled) {
    rolled = cv::Mat::zeros(1, 1, CV_64F);   /// got one extra zero here
    for (int i = 0; i < NUM_LAYER-1; i++) {
        cv::Mat temp = cv::Mat(unrolled[i].rows * unrolled[i].cols , 1, CV_64F, unrolled[i].data);
        cv::vconcat(rolled, temp, rolled);
    }
    rolled = rolled.rowRange(1, rolled.rows); /// removed the extra zero here
}

void costFunction(const cv::Mat& params,    /// initial parameters in a rolled up vector
                  const cv::Mat& X,         /// featurs
                  const cv::Mat& Y,         /// lables
                  double lambda,            /// regulaization parameter
                  double& J,                /// cost to return
                  cv::Mat& gradient) {      /// gradient to return (partial derivative of cost func.)
    // NOTE: - extracting Theta from vertival vector
    cv::Mat Theta[NUM_LAYER-1];
    unrollTheta(Theta, params);
    
    
    // NOTE: - feeding forward and calculating activation parameters
    /// X  --> m x n                                                a0 --> m x (n+1)
    /// a0 * Theta0' --> ( m x (n+1) )  * ( (n+1) x s1 )            a1 --> m x (s1+1)
    /// a1 * Theta1' --> ( m x (s1+1) ) * ( (s1+1) x s2 )           a2 --> m x (s2+1)
    /// a2 * Theta2' --> ( m x (s2+1) ) * ( (s2+1) x s3 )           a3 --> m x (s3+1)
    /// a3 * Theta3' --> ( m x (s3+1) ) * ( (s3+1) x k)             a4 --> m x k
    /// a4 is the hypothesis
    int m = X.rows;
    cv::Mat ones = cv::Mat::ones(m, 1, CV_64F);
    cv::Mat a[NUM_LAYER];
    for(int i = 0; i < NUM_LAYER; i++) {
        a[i] = (i == 0) ? X : sigmoid(a[i-1] * Theta[i-1].t());
        if(i != NUM_LAYER-1) cv::hconcat(ones, a[i], a[i]);  /// don't add ones to last a[i]
    }
    

    // NOTE: - changing each lable of lables to a vector of (0s and a 1)
    cv::Mat Y_ = cv::Mat::zeros(m, OUT_SIZE, CV_64F);
    for (int i = 0; i < m; i++) {
        int columnForOne = Y.at<double>(i,0);
        /// we don't have lable 9=J in dataset, so we need to minus one from index 9 to match lables 0 to 24 in 24 cols
        /// and we need to consider to add one later on, when we want to translate a binary row in Y_ to a lable in Y
        if (columnForOne >= 9) --columnForOne;
        Y_.at<double>(i, columnForOne) = 1;  /// other columns of the row i  is zero
    }
    
    
    // NOTE: - calculating cost J
    /// Y_.t() -->  k x m       a4  --> m x k     ----> sum over a k x k matrice which has 0...0..1..0 in one dimention
    /// openCV's Mat has scalar elements of potentialy up to 4 color chanel
    /// but here we have used Mat as matrice of numbers, and we only want the first chanel
    cv::Scalar j = -1/m * cv::sum( Y_.t() * log(a[4]) + (1 - Y_.t()) * log(1 - a[4]) );
    J = j[0];       /// we only want the first chanel of the Mat matrice
    
    
    // NOTE: - backpropagation to calcute gradients
    cv::Mat Theta_[NUM_LAYER-1];
    cv::Mat delta[NUM_LAYER-1];
    cv::Mat Theta_g[NUM_LAYER-1];
    /// removing the first column (ons) from each of Theta
    for (int i = 0; i < NUM_LAYER - 1; i++) {
        Theta_[i] = Theta[i].colRange(1, Theta[i].cols);
    }
    /// finding each error using `backpropagation`
    /// delta4 --> delta[3] --> m x k                                            Thetha_g3 --> (k x m)  * (m x s3+1)
    /// delta3 --> delta[2] --> (m x k)  x ( k x s3) . (m x s3)                  Thetha_g2 --> (s3 x m) * (m x s2+1)
    /// delta2 --> delta[1] --> (m x s3) x (s3 x s2) . (m x s2)                  Thetha_g1 --> (s2 x m) * (m x s1+1)
    /// delta1 --> delta[0] --> (m x s2) x (s2 x s1) . (m x s1)                  Thetha_g0 --> (s1 x m) * (m x n+1)
    /// delta0 -->  ------  -->                  ---> we ignore this
    for (int i = NUM_LAYER-2; i >= 0; i--) {
        if (i == NUM_LAYER-2)
            delta[i] = (a[i+1] - Y_);
        else
            delta[i] = ( delta[i+1] * Theta_[i+1] ).mul( sigmoidPrime(a[i] * Theta[i].t()) );
    }
    /// feeding forward again to calculate Theta gredient
    for (int i = 0; i < NUM_LAYER-1; i++) {
        Theta_g[i] = 1/m * delta[i].t() * a[i];
    }

    
    // NOTE: - regulization of the cost and Theta_g to prevent overfitting
    for (int i = 0; i < NUM_LAYER-1; i++) {
        j += cv::sum( Theta_[i].mul(Theta_[i]) );
        Theta_g[i] += lambda/m * Theta[i];
    }
    J = lambda/(2 * m) * j[0];

    
    // NOTE: - roll gradients
    rollTheta(Theta_g, gradient);
     
    
    // NOTE: Done. J and gradient has been calculated
}


#endif /* cost_h */
