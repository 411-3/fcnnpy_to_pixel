#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "jpeglib.h"
#include "SLIC.h"
#include "opencv_lbp.h"
#include "cnpy.h"
#include "densecrf.h"
#include <limits>
#include <iostream>
#include <fstream>
#include <sstream>
#include "time.h"
#include "math.h"

using namespace cv;
using namespace std;

#define CLASS 21	//label number
#define IterativeNumber 10

// 21 color for labels
int colors[21] = {0, 128, 32768, 32896, 8388608, 8388736,8421376, 8421504, 64, 192, 32832, 32960,8388672, 8388800, 8421440, 8421568,8192, 16512, 49152, 49280, 8404992};
unsigned int getColor( const unsigned char * c );
int   putColor( unsigned char * c, unsigned int cc );
unsigned char * colorize( int * map, int W, int H, unsigned char * r );
void  writePPM( const char* filename, int W, int H, unsigned char* data );
void  elbp(Mat& src, Mat &dst, int radius, int neighbors);
float calcDistance(int first, int second, vector<vector<float> > & lbp_superpixel, vector<vector<float> > & labxy_superpixel, int h, int w);

int main(int argc, char* argv[])
{
	clock_t start, finish;
	printf("main params = %d.\n", argc);
	const char * img_name = argv[1];
	const char * img_fcn  = argv[2];
	
	//-----------------------------------
	// 01: read jpg to buffer[char]
	//-----------------------------------
	start = clock();
	Mat raw_image = imread(img_name , 1);
	if(raw_image.empty())
		printf("imread failed!\n");
	
	int h = raw_image.rows;
	int w = raw_image.cols;
	int channels = 3;
	int i(0), j(0);
	int x(0), y(0);	//pixel:x,y

	unsigned char * image_buffer = new unsigned char[w * h * channels]; 		// BGRA  
	
	for(i = 0; i < h; i++)
	{	
		for(j = 0; j < w; j++)
		{	
			*(image_buffer + i * w * channels + j*channels+2 ) = raw_image.at<Vec3b>(i,j).val[0];		//B
			*(image_buffer + i * w * channels + j*channels+1 ) = raw_image.at<Vec3b>(i,j).val[1];		//G
			*(image_buffer + i * w * channels + j*channels+0 ) = raw_image.at<Vec3b>(i,j).val[2];		//R
		}
	}	
	finish = clock();
	printf("01 read jpeg to buffer[char]: %lf.\n", (double)(finish-start)/CLOCKS_PER_SEC);
		
	//----------------------------------
	// 02 : Unary from file.npy test
	//----------------------------------
	start = clock();
	cnpy::NpyArray arr = cnpy::npy_load(img_fcn);	
   	float * raw_unary  = new float[w * h * CLASS];
	float temp_score(0);
	if(arr.shape[1] != h || arr.shape[2] != w || arr.shape[0] != CLASS)
		printf("\nimage not match npy.\n");	
	float min_score(0), max_score(0);

    // Put into unary[0-21][1-21][2-21]...
    for(i=0; i<w*h; i++)
	{	
		min_score = INT_MAX;
		max_score = INT_MIN;
		for(j=0; j<CLASS; j++)
        {	
			temp_score = ((const float *)(arr.data))[i + j*w*h];
			raw_unary[i * CLASS + j] = temp_score;
			if( temp_score < min_score )    min_score = temp_score ;
			if( temp_score > max_score )    max_score = temp_score ;
		}
		for(j=0; j<CLASS; j++)
			raw_unary[i * CLASS + j] = -log((raw_unary[i * CLASS + j] - min_score) / (max_score - min_score));
	}
	arr.destruct();		//释放掉读取的标注文件内存	
	printf("raw_unary is ok ! \n");
	finish = clock();
	printf("02 read npy to unary: %lf.\n", (double)(finish-start)/CLOCKS_PER_SEC);


	//-------------------------------------
	// 03 : CRF model
	//-------------------------------------
#if 1
	printf("Start CRF modeling...\n");
	start = clock();
	short * map = new short[w * h];
	int   * map1 = new int[w * h];

	//DenseCRF2D crf(w, h, CLASS);
	DenseCRF2D crf(w, h, CLASS);

	// 把像素点概率映射到块上
	crf.setUnaryEnergy( raw_unary );
	crf.addPairwiseGaussian( 3, 3, 10);
	crf.addPairwiseBilateral( 60, 60, 20, 20, 20, image_buffer, 10 );
	printf("feature into crf ok ! \n");
	finish = clock();
	printf("03 energy: %lf.\n", (double)(finish-start)/CLOCKS_PER_SEC);

	//迭代求每个块的最大概率	

	start = clock();
	crf.map(IterativeNumber, map);

	for(i=0; i< w*h; i++)
		map1[i] = map[i];

	colorize( map1, w, h , image_buffer);
	printf("map to colorize ok !\n");

	writePPM( argv[3], w, h, image_buffer);
	finish = clock();
	printf("03 map: %lf.\n", (double)(finish-start)/CLOCKS_PER_SEC);
	printf("Segment image save done !\nCRF finish successful !\n");
#endif
	
    //----------------------------------
    // Clean up
    //----------------------------------
	delete[] image_buffer;
	delete[] raw_unary;
	delete[] map;	
	delete[] map1;	

	return 0;
}


//----------------------
// 若干辅助函数
//----------------------

// 将RGB值存为一个整型变量值
unsigned int getColor( const unsigned char * c )
{
    return c[0] + 256*c[1] + 256*256*c[2];
}

// 从整型变量中得到RGB值
int putColor( unsigned char * c, unsigned int cc )
{
    c[0] = cc&0xff; c[1] = (cc>>8)&0xff; c[2] = (cc>>16)&0xff;
	return 0;
}

// Produce a color_image from Map Labels
unsigned char * colorize(int * map, int W, int H , unsigned char * r)
{
	printf("entet into colorize ! \n");
	printf("w=%d ,h=%d \n", W, H);
	printf("new ok!\n");
    for( int k=0; k<W*H; k++ )
	{
        int c = colors[ map[k] ];
		putColor( r + 3*k, c);
    }
    return r;
}

// 将数组存为PPM图
void writePPM ( const char* filename, int W, int H, unsigned char* data )
{
    FILE* fp = fopen ( filename, "wb" );
    if ( !fp )
    {
        printf ( "Failed to open file '%s'!\n", filename );
    }
    fprintf ( fp, "P6\n%d %d\n%d\n", W, H, 255 );
    fwrite  ( data, 1, W*H*3, fp );
    fclose  ( fp );
}

// 计算两个21-dimen特征的距离
float calcDistance(int first, int second, vector<vector<float> > & lbp_superpixel, vector<vector<float> > & labxy_superpixel, int h, int w)
{
	float a[CLASS], b[CLASS];
	int i(0), histcount(0);

	printf("[");
    for(vector<int>::size_type ix = 0, i=0; ix < lbp_superpixel[first].size(), i<16; ix++,i++)  //label number
    {
        a[i] = lbp_superpixel[first][ix];
        histcount +=  a[i];
        printf("%3d", int(lbp_superpixel[first][ix]) );
    }
    for(i=0; i<16; i++)
        a[i] /= histcount;  // 16个LBP特征归一化 -> a[]
    histcount = 0;

    printf("\t");
    for(vector<int>::size_type ix = 0, i=16; ix < labxy_superpixel[first].size(), i<21; ix++,i++) //label number
    {
        a[i] = labxy_superpixel[first][ix];
        printf("%5d", int(labxy_superpixel[first][ix]) );
    }
    printf("]\n");
    for(i=16; i<19; i++)
        a[i] /= 255;    // LAB归一化
    a[19] /= h;         // XY归一化
    a[20] /= w;

	printf("\n[");
    for(vector<int>::size_type ix = 0, i=0; ix < lbp_superpixel[second].size(), i<16; ix++,i++)  //label number
    {
        b[i] = lbp_superpixel[second][ix];
        histcount +=  b[i];
        printf("%3d", int(lbp_superpixel[second][ix]) );
    }
    for(i=0; i<16; i++)
        b[i] /= histcount;
    histcount = 0;

    printf("\t");
    for(vector<int>::size_type ix = 0, i=16; ix < labxy_superpixel[second].size(), i<21; ix++,i++)   //label number
    {
        b[i] = labxy_superpixel[second][ix];
        printf("%5d", int(labxy_superpixel[second][ix]) );
    }
    printf("]\n");
    for(i=16; i<19; i++)    b[i] /= 255;
    b[19] /= h;
    b[20] /= w;

    printf("\nAfter normalized.\n");
    for(i=0; i<21; i++)     printf("%f,", a[i]);
    printf("\n");
    for(i=0; i<21; i++)     printf("%f,", b[i]);
    printf("\n");

	float sum1(0), sum2(0), sum3(0);
    for(i=0; i<16; i++)
        sum1 += pow( (a[i]-b[i]), 2);
    for(i=16; i<19; i++)
        sum2 += pow( (a[i]-b[i]), 2);
    for(i=19; i<21; i++)
        sum3 += pow( (a[i]-b[i]), 2);
    return (0.7 * sqrt(sum1) + 0.2 * sqrt(sum2) + 0.1 * sqrt(sum3));
}
