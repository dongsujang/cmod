#include <cstdio>
#include <iostream>
#include <cstdlib>

using namespace std;

double ****filter4d(int quantity, int channel, int rowsize, int colsize){

    double ****result;
    
    result = (double****)malloc(sizeof(double***)*quantity);
    for(int qty = 0; qty < quantity ; qty++){
        *(result + qty) = (double***)malloc(sizeof(double**)*channel);
        for(int cha = 0; cha < channel ; cha++){
            *(*(result + qty) + cha) = (double**)malloc(sizeof(double*)*rowsize);
            for(int row = 0 ; row < rowsize ; row++){
                *(*(*(result + qty) + cha) + row) = (double*)malloc(sizeof(double)*colsize);
                for(int col = 0 ; col < colsize ; col++){
                    result[qty][cha][row][col] = 1; // *(*(*(*(result+qty)+cha)+row)+col)
                }
            }
        }
    }

    return result;
}
double **filter2d(int rowsize, int colsize){
    
    double **result;

    result = (double**)malloc(sizeof(double*)*rowsize);
    for(int row = 0; row<rowsize;row++){
        *(result+row) = (double*)malloc(colsize*sizeof(double));
    }
    return result;
}
