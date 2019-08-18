#include <stdio.h>
#include <iostream>
#include <stdlib.h>

double ***convolution(int out_cha, int channel, int o_size_rows, int o_size_cols, int filter_size, int stride, double ***input, double ****filter)
{
   /*
    int out_cha = 2;
    int cha = 3;
    int o_size_rows = 3;
    int o_size_cols = 3;
    int filter_size = 3;
    int stride = 1;
*/    
    double ***output;
    output = (double***)malloc(sizeof(double**)*out_cha);
    for(int cha = 0 ; cha < out_cha ; cha++){
        *(output + cha) = (double**)malloc(sizeof(double*)*o_size_rows);
        for(int row = 0 ; row < o_size_rows ; row++){
            *(*(output + cha) + row) = (double*)malloc(sizeof(double)*o_size_cols);
        }
    }

    for(int x=0;x<out_cha;x++){
        for(int y=0;y<channel;y++){
           for(int m=0;m<o_size_rows;m++){
		        for(int n=0;n<o_size_cols;n++){
	          		for(int i=0;i<filter_size;i++){
	    	        	for(int j=0;j<filter_size;j++){
		    		        output[x][m][n] = output[x][m][n] + input[y][i+m*stride][j+n*stride]*filter[x][y][i][j];				
        			    }
	            	}
	           	}
        	}
        }
     }
    return output;
}