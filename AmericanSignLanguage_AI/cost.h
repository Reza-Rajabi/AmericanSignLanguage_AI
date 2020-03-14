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
const int HIDEN1_SIZE = 28;     ///                           s1
const int HIDEN2_SIZE = 37;     ///                           s2
const int HIDEN3_SIZE = 28;     ///                           s3
const int OUT_SIZE = 25;        /// same as NUM_LABLE                       k

const int NUM_LAYER = 5;
const double epsilon = 0.15;    /// a smal double to initialize layers' weights randomly


cv::Mat sigmoid(const cv::Mat& Z) {
    cv::Mat G(Z);  /// copy
    double z;
    for(int r = 0; r < Z.rows; r++) {
        for (int c = 0; c < Z.cols; c++) {
            z = G.at<double>(r,c);
            G.at<double>(r,c) = 1.0 / (1.0 + exp(-z));
        }
    }
    
    return G;
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
    cv::Mat weights = cv::Mat::zeros(outConnections, 1 + inConnections, CV_64F);
    cv::RNG randomGenerator;
    for(int r = 0; r < outConnections; r++) {
        for (int c = 0; c < 1+inConnections; c++) {
            weights.at<double>(r,c) = randomGenerator.uniform(outConnections, 1+inConnections) * 2 * epsilon - epsilon;
        }
    }
    
    return weights;
}

void costFunction(const cv::Mat& params,    /// initial parameters in a rolled up vector
                  const cv::Mat& X,         /// featurs
                  const cv::Mat& Y,         /// lables
                  long double lambda,       /// regulaization parameter
                  long double& J,           /// cost to return
                  cv::Mat& gradient) {      /// gradient to return (partial derivative of cost func.)
    // NOTE: - extracting Theta from vertival vector
    /// the dimention of each Thetha is:  the next layer size x its layer size
    /// Theta0 --> s1 x (n+1)
    /// Theta1 --> s2 x (s1+1)
    /// Theta2 --> s3 x (s2+1)
    /// Theta3 --> k x (s3+1)
    int S[] = { IN_SIZE, HIDEN1_SIZE, HIDEN2_SIZE, HIDEN3_SIZE, OUT_SIZE };
    cv::Mat Theta[NUM_LAYER-1];
    int rangeStart = 0, rangeEnd = 0;
    for (int l = 0; l < NUM_LAYER-1; l++) {
        rangeEnd += S[l+1] * (S[l]+1);
        cv::Range range(rangeStart, rangeEnd);
        Theta[l] = params(range, cv::Range::all());
        Theta[l].resize(S[l+1], S[l]+1);
        rangeStart = rangeEnd;
    }
    
    
    // NOTE: - feeding forward and calculating activation parameters
    /// X  --> m x n                                                                a0 --> m x (n+1)
    /// a0 * Theta0' --> ( m x (n+1) ) * ( (n+1) x s1 )             a1 --> m x (s1+1)
    /// a1 * Theta1' --> ( m x (s1+1) ) * ( (s1+1) x s2 )          a2 --> m x (s2+1)
    /// a2 * Theta2' --> ( m x (s2+1) ) * ( (s2+1) x s3 )          a3 --> m x (s3+1)
    /// a3 * Theta3' --> ( m x (s3+1) ) * ( (s3+1) x k)             a4 --> m x k
    /// a4 is the hypothesis
    int m = X.rows;
    cv::Mat ones = cv::Mat::ones(m, 1, CV_64F);
    cv::Mat a[NUM_LAYER];
    for(int i = 0; i < NUM_LAYER-1; i++) {
        a[i] = (i == 0) ? X : sigmoid(a[i-1] * Theta[i-1].t());
        cv::hconcat(ones, a[i], a[i]);
    }
    

    // NOTE: - changing each lable of lables to a vector of (0s and a 1)
    cv::Mat Y_ = cv::Mat::zeros(m, NUM_LABLE, CV_64F);
    for (int i = 0; i < m; i++) {
        int columnForOne = Y.at<double>(i,1);
        Y_.at<double>(i, columnForOne) = 1;  /// other columns of the row i  is zero
    }
    
    
    // NOTE: - calculating cost J
    /// Y_.t() -->  k x m           a4  --> m x k     ----> sum over a k x k matrice which is actuly 0...0..1..0 in one dimention
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
    /// delta4 --> delta[3] --> m x k                                                        Thetha_g3 --> (k x m) * (m x s3+1)
    /// delta3 --> delta[2] --> (m x k) x ( k x s3) . (m x s3)                     Thetha_g2 --> (s3 x m) * (m x s2+1)
    /// delta2 --> delta[1] --> (m x s3) x (s3 x s2) . (m x s2)                  Thetha_g1 --> (s2 x m) * (m x s1+1)
    /// delta1 --> delta[0] --> (m x s2) x (s2 x s1) . (m x s1)                  Thetha_g0 --> (s1 x m) * (m x n+1)
    /// delta0 --> ---------- -->                  ---> we ignore this
    for (int i = NUM_LAYER-2; i >= 0; i--) {
        if (i == NUM_LAYER-2)
            delta[i] = (a[i+1] - Y_);
        else
            delta[i] = ( delta[i+1] * Theta_[i+1] ).mul( sigmoidPrime(a[i] * Theta[i].t()) );
    }
    /// feeding forward again
    for (int i = 0; i < NUM_LAYER-1; i++) {
        Theta_g[i] = 1/m * delta[i].t() * a[i];
    }

    
    // NOTE: - regulization of the cost to prevent overfitting
    for (int i = 0; i < NUM_LAYER-1; i++) {
        j += cv::sum( Theta_[i].mul(Theta_[i]) );
        Theta_g[i] += lambda/m * Theta[i];
    }
    J = lambda/(2 * m) * j[0];

    
    // NOTE: - unroll gradients
    gradient = cv::Mat::zeros(1, 1, CV_64F);         /// got one extra zero here
    for (int i = 0; i < NUM_LAYER-1; i++) {
        Theta_g[i].resize(Theta_g[i].rows + Theta_g[i].cols , 1);
        cv::vconcat(Theta_g[i], gradient, gradient);
    }
    gradient = gradient.rowRange(1, gradient.rows); /// removing the extra zero here
     
    
    // NOTE: Done. J and gradient has been set up
}


#endif /* cost_h */
