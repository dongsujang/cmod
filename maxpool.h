#include <stdio.h>
#include <iostream>
#include <stdlib.h>

double ***pooling(int channel, int out_rowsize, int out_colsize ,int filter_size, int stride, double ***input)
{
	int i,j,m,n,k,x,y;
	int max;

    double ***pool_output;
    pool_output = (double***)malloc(sizeof(double**)*channel);

    for(int cha = 0 ; cha < channel ; cha++){
        *(pool_output + cha) = (double**)malloc(sizeof(double*)*out_rowsize);
        for(int row = 0 ; row < out_rowsize ; row++){
            *(*(pool_output + cha) + row) = (double*)malloc(sizeof(double)*out_colsize);
            }
        }

	for(k=0;k<channel;k++){
		for(m=0;m<out_rowsize;m++){
			for(n=0;n<out_colsize;n++){
				max = INT_MIN;
				for(i=0;i<filter_size;i++){
					for(j=0;j<filter_size;j++){
						if(max < input[k][i+m*stride][j+n*stride]){
							max = input[k][i+m*stride][j+n*stride];
						}
					}
				}
				pool_output[k][m][n] = max;
			}
		}
	}
    return pool_output;
}