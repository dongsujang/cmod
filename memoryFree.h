#include <stdio.h>
#include <iostream>
#include <stdlib.h>

void memoryFree4d(int quantity, int channel, int rowsize, double ****input){
    for(int qty = 0; qty < quantity ; qty++){
        for(int cha = 0;cha<channel;cha++){
		    for(int row=0;row<rowsize;row++){
			    free(*(*(*(input+qty)+cha)+row));
		    }
		    free(*(*(input+qty)+cha));
	    }
	    free(*(input+qty));
    }
    free(input);

}

void memoryFree3d(int channel, int rowsize, double ***input){
    for(int cha = 0;cha<channel;cha++){
		for(int row=0;row<rowsize;row++){
			free(*(*(input+cha)+row));
		}
		free(*(input+cha));
	}
	free(input);
}

void memoryFree2d(int rowsize, double **input){
    for(int row = 0; row < rowsize ; row++){
        free(*(input+row));
    }
    free(input);
}