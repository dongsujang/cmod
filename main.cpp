#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include "conv.h"
#include "filter.h"
#include "input.h"
#include "memoryFree.h"
#include "maxpool.h"

int main()
{
    
    double ***input;
    double ****conv1filter;
    double ***conv1Result;
    double ***pool1Result;
    double ****conv2filter;
    double ***conv2Result;
    double ***pool2Result;
    double fully1input[800];
    double **fully1filter;
    double fully1output[500];
    double **fully2filter;
    double fully2output[10];

    int k,m,n,i,j,x,y,l;

    FILE *fp1;
    FILE *fp2;
    FILE *fp3;
    FILE *fp4;

    fp1 = fopen("/home/socmgr/workspcae/week6/1/Untitled Folder/realreference/conv1_weight.txt","r"); ///home/socmgr/work/Untitled Folder/
    fp2 = fopen("/home/socmgr/workspcae/week6/1/Untitled Folder/realreference/conv2_weight.txt","r");
    fp3 = fopen("/home/socmgr/workspcae/week6/1/Untitled Folder/realreference/fully1_weight.txt","r");
    fp4 = fopen("/home/socmgr/workspcae/week6/1/Untitled Folder/realreference/fully2_weight.txt","r");

//layer1 
    input = inputMatrix(1, 28, 28);
//layer1 end

//layer2 conv1

    conv1filter = filter4d(20, 1, 5, 5);
    
    for( i = 0 ; i<20 ; i++){
		for( j = 0 ; j<1 ; j++){
			for( k = 0 ; k<5 ; k++){
				for( l=0 ; l<5 ; l++){
					fscanf(fp1, "%lf", &conv1filter[i][j][k][l]);
				}
			}
		}
	}
    
    conv1Result = convolution(20, 1, 24, 24, 5, 1, input, conv1filter);//out_cha,channel,o_size_rows,o_size_cols,filter_size,stride,***input,****filter

//layer2 end

//layer3
    pool1Result = pooling(20,12,12,2,2,conv1Result);//(channel, out_rowsize, out_colsize , filter_size, stride, ***input)
//layer3 end

//layer4

    conv2filter = filter4d(50, 20, 5, 5);
    
    for( i = 0 ; i<50 ; i++){
		for( j = 0 ; j<20 ; j++){
			for( k = 0 ; k<5 ; k++){
				for( l=0 ; l<5 ; l++){
					fscanf(fp2, "%lf", &conv2filter[i][j][k][l]);
				}
			}
		}
	}
   
    conv2Result = convolution(50, 20, 8, 8, 5, 1, pool1Result, conv2filter);//out_cha,channel,o_size_rows,o_size_cols,filter_size,stride,***input,****filter

//layer 4 end

//layer 5

    pool2Result = pooling(50,4,4,2,2,conv2Result);//(channel, out_rowsize, out_colsize , filter_size, stride, ***input)

//layer 5 end

//layer 6
    // 3차원을 1열로 800개 세우기
    n = 0;
    for(i=0;i<50;i++){
        for(j=0;j<4;j++){
            for(k=0;k<4;k++){
                fully1input[n] = pool2Result[i][j][k];
                n++;
            }
        }
    }
    // 필터 동적할당
    fully1filter = filter2d(500, 800);
    // 필터값 읽어오기
    for( i=0 ; i<500 ; i++){
		for( j=0 ; j<800 ; j++){
			fscanf(fp3, "%lf", &fully1filter[i][j]);
		}
	}
    
    // fully connected 연산
    for(i=0;i<500;i++){
        for(j=0;j<800;j++){
        fully1output[j] = fully1output[j] + fully1input[j]*fully1filter[i][j];
        }
    }

//layer 6 end

//layer 7 relu activation

    for(i=0;i<500;i++){
        if(fully1output[i]<=0){
            fully1output[i] = 0;
        }
        else{
            fully1output[i] = fully1output[i];
        }
    }
    
//layer 7 end


//layer 8

    fully2filter = filter2d(10, 500);

    for( i=0 ; i<10 ; i++){
		for( j=0 ; j<500 ; j++){
			fscanf(fp4, "%lf", &fully2filter[i][j]);
            //printf("%lf ", fully2filter[i][j]);
		}
	}


    for(i=0;i<10;i++){
        for(j=0;j<500;j++){
        fully2output[j] = fully2output[j] + fully1output[j]*fully2filter[i][j];
        }
    }

    for(i=0;i<10;i++){
        printf("[%d] : %lf \n", i,fully2output[i]);
    }

//layer 8 end
    memoryFree3d(1, 28, input);//int channel, int rowsize, double ***input
    memoryFree4d(20, 1, 5, conv1filter);//int quantity, int channel, int rowsize, double ****input
    memoryFree3d(20, 24, conv1Result);//int channel, int rowsize, double ***input
    memoryFree3d(20, 12, pool1Result);//int channel, int rowsize, double ***input
    memoryFree4d(50, 20, 5, conv2filter);//int quantity, int channel, int rowsize, double ****input
    memoryFree3d(50, 8, conv2Result);//int channel, int rowsize, double ***input
    memoryFree3d(50, 4, pool2Result);//int channel, int rowsize, double ***input
    memoryFree2d(500, fully1filter);
    memoryFree2d(10, fully2filter);

    return 0;
}
