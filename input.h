#include <cstdio>
#include <iostream>
#include <cstdlib>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

double ***inputMatrix(int channel, int rowsize, int colsize){

    cv::Mat img = cv::imread("9.jpg",IMREAD_GRAYSCALE);

    double ***result;
    result = (double***)malloc(sizeof(double**)*channel);

    for(int cha = 0 ; cha < channel ; cha++){
        *(result + cha) = (double**)malloc(sizeof(double*)*rowsize);
        for(int row = 0 ; row < rowsize ; row++){
            *(*(result + cha) + row) = (double*)malloc(sizeof(double)*colsize);
            }
        }


    for(int cha = 0; cha < channel; cha++){
        for(int row = 0; row < rowsize; row++){
            for(int col = 0; col < colsize; col++){
                result[cha][row][col] = img.at<uchar>(row,col);
            }
        }
    }
    return result;
}
