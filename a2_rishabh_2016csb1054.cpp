/*
	Name Rishabh Singh 2016csb1054

	sample run

		g++ a2_rishabh_2016csb1054.cpp -o a `pkg-config --cflags --libs opencv`
		./a

	All the function are working as given in documents. I have made some function in which even colored images can also be processed.
	I have done this code individually and only seen the basis like how to initialise Mat objects and other stuff.
	For rotation i have refered to a link for taking hint so that the image properly fit in the window. (http://iiif.io/api/annex/notes/rotation/)

	If there is any bug in code please mail me at 2016csb1054@iitrpr.ac.in

*/



#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdint.h>
#include <climits>
#include <math.h>

// Value of pi in maths
#define PI 3.14159265

//	Values with which in piecewise transformation the grascale values will change.
#define P1 240
#define P2 128
#define P3 10

//	value of the block size in adaptive histogram equlisation.
#define ahc 20


using namespace cv;
using namespace std;

float RMSE(const Mat& I1, const Mat& I2){
	float sum = 0;
	for(int i=0;i<I1.rows;i++){
		for(int j=0;j<I2.cols;j++){
			sum += (I1.at<uchar>(i,j)-I2.at<uchar>(i,j))*(I1.at<uchar>(i,j)-I2.at<uchar>(i,j));
		}
	}
	float err = sqrt(sum/(I1.rows*I1.cols));
	return err;

}

float RMSEc(const Mat& I1, const Mat& I2){
	float sum = 0;
	for(int k=0;k<1;k++){
		for(int i=0;i<I1.rows;i++){
			for(int j=0;j<I2.cols;j++){
				sum += (I1.at<Vec3b>(i,j)[k]-I2.at<Vec3b>(i,j)[k])*(I1.at<Vec3b>(i,j)[k]-I2.at<Vec3b>(i,j)[k]);
			}
		}
	}
	float err = sqrt(sum/(I1.rows*I1.cols));
	return err;

}
//	Negative function to perform the negative operation on image.
void negative(Mat a){
	Mat x = a.clone();
	for(int i=0;i<3;i++){
		for(int r=0;r<a.rows;r++){
			for(int c=0;c<a.cols; c++){
				x.at<Vec3b>(r,c)[i] = 255 - a.at<Vec3b>(r,c)[i];
			}
		}
	}
	namedWindow("Output",WINDOW_AUTOSIZE);
	namedWindow("Input",WINDOW_AUTOSIZE);
	imshow("Output",x);
	imshow("Input", a);
	moveWindow("Input", 200, 400);
	moveWindow("Output", 200+ a.cols + 10, 400);
}

//	Resizing an image using nearestNieghbour principle.

void nearestNeighbour(Mat& a){
	int rows, cols;
	cout << "Initial rows "<<a.rows << " Initial columns " << a.cols<<endl;
	cout << "Enter the final Rows and Final Columns : ";
	cin>>rows>>cols;
	Mat x(rows,cols,CV_8UC3,Scalar(0,0,0));
	for(int i=0;i<3;i++){
		for(int r=0;r<rows;r++){
			for(int c=0;c<cols; c++){ 	
				x.at<Vec3b>(r,c)[i] = a.at<Vec3b>((r)*a.rows/rows,(c)*a.cols/cols )[i];
			}
		}
	}
	Mat ix;
	resize(a,ix,Size(x.cols,x.rows),0,0,0);
	float err = RMSEc(ix,x);		
	cout <<"RMSE Error in the image is equal to "<<err<<endl;
	namedWindow("Output",WINDOW_AUTOSIZE);
	namedWindow("Input",WINDOW_AUTOSIZE);
	imshow("Output",x);
	imshow("Input", a);
	moveWindow("Input", 200, 400);
	moveWindow("Output", 200+ a.cols + 10, 400);
	
}

//	Function to rotate an image with the degree user enter in console.

//		http://iiif.io/api/annex/notes/rotation/ (hint taken from this site to enlarge the image window).
void rotation(Mat a){
	int deg;
	cout << "Enter the degree by which you want to rotate an image: ";
	cin>>deg;
	if (deg < 0){
		deg = 360 + deg;
	}
	float width, height;
	if ((deg <90 && deg>=0) || (deg < 270 && deg >=180)){
		height = a.cols*abs(cos(deg*PI/180)) + a.rows*abs(sin(deg*PI/180));
		width = a.rows*abs(sin(deg*PI/180)) + a.cols*abs(cos(deg*PI/180));
	}else if(deg <180 && deg>=90 || (deg < 360 && deg >=270)){
		height = a.rows*abs(cos(deg*PI/180)) + a.cols*abs(sin(deg*PI/180));
		width = a.cols*abs(sin(deg*PI/180)) + a.rows*abs(cos(deg*PI/180));
	}
	Mat x(height,width,CV_8UC3,Scalar(0,0,0));
	float m[3][3] = {0};
	m[0][0] = cos(deg*PI/180);
	m[0][1] = sin(deg*PI/180);
	m[1][0] = -sin(deg*PI/180);
	m[1][1] = cos(deg*PI/180);
	m[2][2] = 1;
	Mat aClone = x.clone();
	for(int i=0;i<3;i++){
		for(int r=(aClone.rows-a.rows)/2;r<a.rows+int((aClone.rows-a.rows)/2);r++){
			for(int c=int((aClone.cols-a.cols)/2);c<a.cols+int((aClone.cols-a.cols)/2); c++){
				aClone.at<Vec3b>(r,c)[i] = a.at<Vec3b>(r-int((aClone.rows-a.rows)/2),c - int((aClone.cols-a.cols)/2))[i];			
			}
		}
	}
	for(int i=0;i<3;i++){
		for(int r=0;r<x.rows;r++){
			for(int c=0;c<x.cols; c++){
				float x1 = m[0][0]*(r-x.rows/2) + m[0][1]*(c-x.cols/2);
				float y1 = m[1][0]*(r-x.rows/2) + m[1][1]*(c-x.cols/2);
				if (x1+x.rows/2<0 || x1+x.rows/2>=aClone.rows || y1+x.cols/2<0 || y1+x.cols/2>aClone.cols){
					x.at<Vec3b>(r,c)[i] = 0; 
				}else {
					x.at<Vec3b>(r,c)[i] = aClone.at<Vec3b>(x1+x.rows/2,y1+x.cols/2)[i];
				}			
			}
		}
	}
	namedWindow("Output",WINDOW_AUTOSIZE);
	namedWindow("Input",WINDOW_AUTOSIZE);
	imshow("Output",x);
	imshow("Input", a);
	moveWindow("Input", 200, 400);
	moveWindow("Output", 200+ a.cols + 10, 400);
}


//	Function to perform shearing operation on image.

void shear(Mat a){
	float xshear,yshear;
	cout << "Enter the shear fraction (0-1) in horizontal and vertical direction: ";
	cin>>xshear>>yshear;
	if (xshear<0 ||xshear>1 || yshear<0 || yshear>1){
		cout <<"Please Enter valid number !!!"<<endl;
		return;
	}
	Mat x(a.rows + a.cols*yshear,a.cols + a.rows*xshear,CV_8UC3,Scalar(0,0,0));
	float m[3][3] = {0};
	m[0][0] = 1/(1-xshear*yshear);
	m[0][1] = -yshear/(1-xshear*yshear);
	m[1][0] = -xshear/(1-xshear*yshear);
	m[1][1] = 1/(1-xshear*yshear);
	m[2][2] = 1;

	for(int i=0;i<3;i++){
		for(int r=0;r<x.rows;r++){
			for(int c=0;c<x.cols; c++){
				float x1 = m[0][0]*(r) + m[0][1]*(c);
				float y1 = m[1][0]*(r) + m[1][1]*(c);
				if (x1<0 || x1>=a.rows || y1<0 || y1>a.cols){
					x.at<Vec3b>(r,c)[i] = 0;
				}else {
					x.at<Vec3b>(r,c)[i] = a.at<Vec3b>(x1,y1)[i];
				}			
			}
		}
	}
	namedWindow("Output",WINDOW_AUTOSIZE);
	namedWindow("Input",WINDOW_AUTOSIZE);
	imshow("Output",x);
	imshow("Input", a);
	moveWindow("Input", 200, 400);
	moveWindow("Output", 200+ a.cols + 10, 400);
}

//	Function to perform translation on images in horizontal and vertical direction.

void translation(Mat a){
	int xshift,yshift;
	cout <<"Enter the shifting in horizontal and vertical direction to be performed on image: ";
	cin>>yshift>>xshift;
	Mat x(a.rows,a.cols,CV_8UC3,Scalar(0,0,0));

	for(int i=0;i<3;i++){
		for(int r=0;r<a.rows;r++){
			for(int c=0;c<a.cols; c++){
				int x1 = r-xshift;
				int y1 = c-yshift ;
				
				if (x1<0 || x1>=a.rows || y1<0 || y1>a.cols){
					x.at<Vec3b>(r,c)[i] = 0;
				}else {
					x.at<Vec3b>(r,c)[i] = a.at<Vec3b>(x1,y1)[i];
				}
			}
		}
	}
	namedWindow("Output",WINDOW_AUTOSIZE);
	namedWindow("Input",WINDOW_AUTOSIZE);
	imshow("Output",x);
	imshow("Input", a);
	moveWindow("Input", 200, 400);
	moveWindow("Output", 200+ a.cols + 10, 400);
}

//	Function to perform sacling in horizontal and vertical direction.

void scaling(Mat a){
	float xscale,yscale;
	cout << "Enter the scaling factor in horizontal and vertical direction (greater than 0): ";
	cin>>xscale>>yscale;
	Mat x(int(a.rows*yscale),int(a.cols*xscale),CV_8UC3,Scalar(0,0,0));
	for(int i=0;i<3;i++){
		for(int r=0;r<x.rows;r++){
			for(int c=0;c<x.cols; c++){
				float x1 = r/yscale;
				float y1 = c/xscale;
				
				if (x1<0 || x1>=a.rows || y1<0 || y1>a.cols){
					x.at<Vec3b>(r,c)[i] = 0;
				}else {
					x.at<Vec3b>(r,c)[i] = a.at<Vec3b>(x1,y1)[i];
				}
			}
		}
	}
	namedWindow("Output",WINDOW_AUTOSIZE);
	namedWindow("Input",WINDOW_AUTOSIZE);
	imshow("Output",x);
	imshow("Input", a);
	moveWindow("Input", 200, 400);
	moveWindow("Output", 200+ a.cols + 10, 400);
}

// Function which perform log Transformation on the input image.

void logTranformation(Mat a){
	float con;
	cout << "Enter the constant value to perform log Transformation: ";
	cin >> con;
	Mat x(int(a.rows),int(a.cols),CV_8UC3,Scalar(0,0,0));
	for(int i=0;i<3;i++){
		for(int r=0;r<a.rows;r++){
			for(int c=0;c<a.cols; c++){
				x.at<Vec3b>(r,c)[i] = con*log(a.at<Vec3b>(r,c)[i]+1);
			}
		}
	}
	namedWindow("Output",WINDOW_AUTOSIZE);
	namedWindow("Input",WINDOW_AUTOSIZE);
	imshow("Output",x);
	imshow("Input", a);
	moveWindow("Input", 200, 400);
	moveWindow("Output", 200+ a.cols + 10, 400);
}

//	Function which perform the Gamma Transformation on the input Image.

void gammaTranformation(Mat a){
	float gamma,con;
	cout << "Enter the constant value and ganma to perform Gamma Transformation: ";
	cin >> con>>gamma;
	Mat x=a.clone();
	for(int i=0;i<3;i++){
		for(int r=0;r<a.rows;r++){
			for(int c=0;c<a.cols; c++){
				x.at<Vec3b>(r,c)[i] = con*pow(a.at<Vec3b>(r,c)[i],gamma);
			}
		}
	}
	namedWindow("Output",WINDOW_AUTOSIZE);
	namedWindow("Input",WINDOW_AUTOSIZE);
	imshow("Output",x);
	imshow("Input", a);
	moveWindow("Input", 200, 400);
	moveWindow("Output", 200+ a.cols + 10, 400);
}

//	Function to perform the PieceWise Transormation on the input image.

void piecewiseTransformation(Mat a){
	float m,n;
	cout <"Enter two values for performing piecewise Transformation: ";
	cin >> m>>n;
	float max, min;
	Mat x=a.clone();
	if (m>n){
		max=m;
		min=n;
	}else{
		max=n;
		min=m;
	}
	for(int i=0;i<3;i++){
		for(int r=0;r<a.rows;r++){
			for(int c=0;c<a.cols; c++){
				if (a.at<Vec3b>(r,c)[i] > max){
					x.at<Vec3b>(r,c)[i] = P1;
				}
				else if(a.at<Vec3b>(r,c)[i] <min){
					x.at<Vec3b>(r,c)[i] = P3;
				}else{
					x.at<Vec3b>(r,c)[i] = P2;
				}
			}
		}
	}
	namedWindow("Output",WINDOW_AUTOSIZE);
	namedWindow("Input",WINDOW_AUTOSIZE);
	imshow("Output",x);
	imshow("Input", a);
	moveWindow("Input", 200, 400);
	moveWindow("Output", 200+ a.cols + 10, 400);
}

//	Function to perform the bitplane slicing on the input image.

void bitplaneSclicing(){
	cout <<"Enter the image name to perform bit-plane-sclicing: ";
	string name;cin>>name;
	Mat a = imread(name,CV_LOAD_IMAGE_GRAYSCALE);
	cout <<"Enter the plane to perform bitplane Sclicing: ";
	int m;
	cin >> m;
	if (m>=9 || m<1) {
		cout << "Please enter no b/w 1 and 8"<<endl;
		return;
	}
	Mat x=a.clone();
	for(int r=0;r<a.rows;r++){
		for(int c=0;c<a.cols; c++){
			//cout << "value poixesl "<< (a.at<Vec3b>(r,c)[i] & (1<<m)) << endl;
			if ( (a.at<uint8_t>(r,c) & (1<<m-1) )> 0) x.at<uint8_t>(r,c) = 255;
			else x.at<uint8_t>(r,c) = 0;
		}
	}
	
	namedWindow("Output",WINDOW_AUTOSIZE);
	namedWindow("Input",WINDOW_AUTOSIZE);
	imshow("Output",x);
	imshow("Input", a);
	moveWindow("Input", 200, 400);
	moveWindow("Output", 200+ a.cols + 10, 400);
}
//	Function to perform Bilinear Interpolation on image to resize it to new dimension.


void bilinearInterpolation(Mat a){
	int rows, cols;
	cout << "Initial rows "<<a.rows << " Initial columns " << a.cols<<endl;
	cout << "Enter the final Rows and Final Columns : ";
	cin>>rows>>cols;
	Mat x(rows,cols,CV_8UC3,Scalar(0,0,0));
	int count = 0;
	for(int i=0;i<3;i++){
		for(int r=0;r<rows;r++){
			for(int c=0;c<cols; c++){
				//cout << ++count << " row: "<<r << " cols " <<c<< endl;
				float x1 = a.cols*c/cols + a.cols/(2*cols);
				float y1 = a.rows*r/rows + a.rows/(2*rows);
				float n1,n2,n3,n4;
				if (((int)x1 + 0.5 - x1) > 0){
					n2 = (int)x1 + 0.5;
					n1 = (int)x1 - 0.5;
				}else{
					n1 = (int)x1 + 0.5;
					n2 = (int)x1 + 1.5;
				}
				if (((int)y1 + 0.5 - y1) > 0){
					n4 = (int)y1 + 0.5;
					n3 = (int)y1 - 0.5;
				}else{
					n3 = (int)y1 + 0.5;
					n4 = (int)y1 + 1.5;
				}
				float alpha = x1-n1;
				float beta = y1-n3;
				if (n1<0.5) {n1=0.5;}
				if (n3<0.5) n3=0.5;
				if (n2>a.cols-0.5) n2=a.cols-0.5;
				if (n4>a.rows-0.5) n4=a.rows-0.5;
				//cout << "n1 : "<<n1 << " n3 " << n3<<endl;
				x.at<Vec3b>(r,c)[i] = alpha*beta*a.at<Vec3b>(int(n4), int(n2))[i] + alpha*(1-beta)*a.at<Vec3b>(int(n3), int(n2))[i] + beta*(1-alpha)*a.at<Vec3b>(int(n4), int(n1))[i] + (1-alpha)*(1-beta)*a.at<Vec3b>(int(n3), int(n1))[i];
			}
		}
	}
	Mat ix;
	resize(a,ix,Size(x.cols,x.rows),0,0,0);
	float err = RMSEc(ix,x);		
	cout <<"RMSE Error in the image is equal to "<<err<<endl;
	namedWindow("Output",WINDOW_AUTOSIZE);
	namedWindow("Input",WINDOW_AUTOSIZE);
	imshow("Output",x);
	imshow("Input", a);
	moveWindow("Input", 200, 400);
	moveWindow("Output", 200+ a.cols + 10, 400);
}

void histogrmMatching(){
	cout <<"Enter the name of image on which histogram Matching is to be performed: ";
	string name1,name2;
	cin>>name1;
	Mat a = imread(name1, CV_LOAD_IMAGE_GRAYSCALE);
	cout <<"Enter the name of reference image: ";
	cin >> name2;
	Mat b = imread(name2, CV_LOAD_IMAGE_GRAYSCALE);
	Mat x=a.clone();
	float arr1[256] = {0};
	float arr2[256] = {0};
	for(int r=0;r<a.rows;r++){
		for(int c=0;c<a.cols; c++){
			arr1[a.at<uint8_t>(r,c)] = arr1[a.at<uint8_t>(r,c)]+1;
		}
	}
	for(int r=0;r<b.rows;r++){
		for(int c=0;c<b.cols; c++){
			arr2[b.at<uint8_t>(r,c)] = arr2[b.at<uint8_t>(r,c)]+1;
		}
	}
	for(int i=0;i<256;i++){
		arr1[i] = arr1[i]/(a.rows*a.cols);
		arr2[i] = arr2[i]/(b.rows*b.cols);
	}
	float cumarr1[256] = {0};
	float cumarr2[256] = {0};
	cumarr1[0] = arr1[0];
	cumarr2[0] = arr2[0];
	for(int i=1;i<256;i++){
		cumarr1[i] = arr1[i]+ cumarr1[i-1];
		cumarr2[i] = arr2[i]+ cumarr2[i-1];
	}
	for(int i=0;i<256;i++){
		cumarr1[i] = cumarr1[i]*255;
		cumarr2[i] = cumarr2[i]*255;
	}
	for(int i=0;i<256;i++){
		cumarr1[i] = round(cumarr1[i]);
		cumarr2[i] = round(cumarr2[i]);
	}
	for(int i=0;i<256;i++){
		int min = 256;
		int index=-1;
		for(int j=0;j<256;j++){
			if (abs(cumarr1[i] - cumarr2[j]) < min){
				min = abs(cumarr1[i] - cumarr2[j]);
				index = j;
			}
			if (abs(cumarr1[i] - cumarr2[j]) == 0 ){
				index = j;
				break;
			}
		}
		cumarr1[i] = index;
	}
	for(int r=0;r<a.rows;r++){
		for(int c=0;c<a.cols; c++){
			x.at<uint8_t>(r,c) = cumarr1[a.at<uint8_t>(r,c)];
		}
	}
	namedWindow("Output",WINDOW_AUTOSIZE);
	namedWindow("Input",WINDOW_AUTOSIZE);
	imshow("Output",x);
	imshow("Input", a);
	moveWindow("Input", 200, 400);
	moveWindow("Output", 200+ a.cols + 10, 400);

}

void histogrmEqulisation(){
	cout<<"Enter the name of image of which histogram equalization is to be done: ";
	string name;
	cin >> name;
	Mat a = imread(name, CV_LOAD_IMAGE_GRAYSCALE);
	Mat x=a.clone();
	float arr[256] = {0};
	for(int r=0;r<a.rows;r++){
		for(int c=0;c<a.cols; c++){
			arr[a.at<uint8_t>(r,c)] = arr[a.at<uint8_t>(r,c)]+1;
		}
	}
	for(int i=0;i<256;i++){
		arr[i] = arr[i]/(a.rows*a.cols);
	}
	float cumarr[256] = {0};
	cumarr[0] = arr[0];
	for(int i=1;i<256;i++){
		cumarr[i] = arr[i]+ cumarr[i-1];
	}
	for(int i=0;i<256;i++){
		cumarr[i] = cumarr[i]*255;
	}
	for(int i=0;i<256;i++){
		cumarr[i] = round(cumarr[i]);
	}
	for(int r=0;r<a.rows;r++){
		for(int c=0;c<a.cols; c++){
			x.at<uint8_t>(r,c) = cumarr[a.at<uint8_t>(r,c)];
		}

	}
	Mat ix;
	equalizeHist( a,ix );
	float err = RMSEc(ix,x);		
	cout <<"RMSE Error in the image is equal to "<<err<<endl;
	namedWindow("Output",WINDOW_AUTOSIZE);
	namedWindow("Input",WINDOW_AUTOSIZE);
	imshow("Output",x);
	imshow("Input", a);
	moveWindow("Input", 200, 400);
	moveWindow("Output", 200+ a.cols + 10, 400);

}


void adaptiveHistogram(){
	cout<<"Enter the name of image to perform adaptive Histogram Equlisation: ";
	string name; cin>>name;
	Mat a = imread(name, CV_LOAD_IMAGE_GRAYSCALE);
	Mat x=a.clone();
	for(int r=0;r<x.rows;r++){
		for(int c=0;c<x.cols; c++){
			float arr[256] = {0};
			for(int i=r-ahc;i<r+ahc+1;i++){
				for(int j=c-ahc;j<c+ahc+1;j++){
					arr[a.at<uint8_t>(i,j)] = arr[a.at<uint8_t>(i,j)]+1;
				}
			}
			for(int m=0;m<256;m++){
				arr[m] = arr[m]/((2*ahc+1)*(2*ahc+1));
			}
			float cumarr[256] = {0};
			cumarr[0] = arr[0];
			for(int m=1;m<256;m++){
				cumarr[m] = arr[m]+ cumarr[m-1];
			}
			for(int m=0;m<256;m++){
				cumarr[m] = cumarr[m]*255;
			}
			for(int m=0;m<256;m++){
				cumarr[m] = round(cumarr[m]);
			}
			x.at<uint8_t>(r,c) = cumarr[a.at<uint8_t>(r,c)];
		}

	}
	namedWindow("Output",WINDOW_AUTOSIZE);
	namedWindow("Input",WINDOW_AUTOSIZE);
	imshow("Output",x);
	imshow("Input", a);
	moveWindow("Input", 200, 400);
	moveWindow("Output", 200+ a.cols + 10, 400);

}

void tiepoint(){
	cout << "Enter the four coordiantes (x,y) of initial image"<<endl;
	float x[8];
	for(int i=0;i<8;i++){
		cin >> x[i];
	}
	cout << "Enter the four coordiantes (x,y) of transformed image"<<endl;
	Mat x1(8,1,CV_32F,Scalar(0));
	for(int i=0;i<8;i++){
		cin >> x1.at<float>(i,0);
	}
	string inimg,defimg;
	cout << "Enter name of input image : ";
	cin >> inimg;
	cout <<"Enter name of deformed image: ";
	cin >> defimg;
	Mat inmat = imread(inimg,CV_LOAD_IMAGE_GRAYSCALE);
	Mat defmat = imread(defimg,CV_LOAD_IMAGE_GRAYSCALE);
	Mat outMat(inmat.rows,inmat.cols,CV_8UC1,Scalar(0)) ;
	Mat mat(8,8,CV_32F,Scalar(0)) ;
	Mat consmat(8,1,CV_32F,Scalar(0)) ;
	mat.at<float>(0,0) = x[0];
	mat.at<float>(0,1) = x[1];
	mat.at<float>(0,2) = x[0]*x[1];
	mat.at<float>(0,3) = 1;
	mat.at<float>(1,4) = x[0];
	mat.at<float>(1,5) = x[1];
	mat.at<float>(1,6) = x[0]*x[1];
	mat.at<float>(1,7) = 1;

	mat.at<float>(2,0) = x[2];
	mat.at<float>(2,1) = x[3];
	mat.at<float>(2,2) = x[2]*x[3];
	mat.at<float>(2,3) = 1;
	mat.at<float>(3,4) = x[2];
	mat.at<float>(3,5) = x[3];
	mat.at<float>(3,6) = x[2]*x[3];
	mat.at<float>(3,7) = 1;

	mat.at<float>(4,0) = x[4];
	mat.at<float>(4,1) = x[5];
	mat.at<float>(4,2) = x[4]*x[5];
	mat.at<float>(4,3) = 1;
	mat.at<float>(5,4) = x[4];
	mat.at<float>(5,5) = x[5];
	mat.at<float>(5,6) = x[4]*x[5];
	mat.at<float>(5,7) = 1;

	mat.at<float>(6,0) = x[6];
	mat.at<float>(6,1) = x[7];
	mat.at<float>(6,2) = x[6]*x[7];
	mat.at<float>(6,3) = 1;
	mat.at<float>(7,4) = x[6];
	mat.at<float>(7,5) = x[7];
	mat.at<float>(7,6) = x[6]*x[7];
	mat.at<float>(7,7) = 1;
	cout << mat<<endl;
	consmat = mat.inv()*(x1);
	cout << consmat<<endl;
	float carr[8] = {0};
	for(int m=0;m<8;m++){
		carr[m] = consmat.at<float>(m,0);
	}
	for(int r=0;r<inmat.rows;r++){
		for(int c=0;c<inmat.cols; c++){
				outMat.at<uint8_t>(r,c) = defmat.at<uint8_t>(carr[0]*r + carr[1]*c + carr[2]*r*c + carr[3],carr[4]*r + carr[5]*c + carr[6]*r*c + carr[7]);
		}
	}
	namedWindow("Output",WINDOW_AUTOSIZE);
	namedWindow("Input",WINDOW_AUTOSIZE);
	imshow("Output",outMat);
	imshow("Input", defmat);
	moveWindow("Input", 200, 400);
	moveWindow("Output", 200+ defmat.cols + 10, 400);
}

int main( int argc, char** argv ) {
	cout<<"Enter the operation to execute\n"<<endl;
	cout<<"Enter 1 to Resize an Image"<<endl;
	cout<<"Enter 2 for Translation"<<endl;
	cout<<"Enter 3 for Rotating an image"<<endl;
	cout<<"Enter 4 for Scaling an image"<<endl;
	cout<<"Enter 5 for Shearing an image"<<endl;
	cout<<"Enter 6 for performing image negative"<<endl;
	cout<<"Enter 7 for log transformation"<<endl;
	cout<<"Enter 8 for gamma transformation"<<endl;
	cout<<"Enter 9 for bit plane slicing"<<endl;
	cout<<"Enter 10 to construct an image with given tie points"<<endl;
	cout<<"Enter 11 for Histogram equalization"<<endl;
	cout<<"Enter 12 for adaptive Histogram equalization"<<endl;
	cout<<"Enter 13 for Histrogram matching"<<endl;
	cout<<"Enter 14 for piecewise Transformation"<<endl;
	cout<<"Enter 0to exit"<<endl;
	int n;
	cin>>n;
	if (n==1) {
		cout<<"Enter image name: ";
		string name;
		cin>>name;
		Mat a = imread(name);
		int flag=1;
		cout<<"Enter 1 for nearest neighbour transformation and 2 for Bilinear interpolation: ";
		cin>>flag;
		if (flag==1)  {
			nearestNeighbour(a);
		}
		else  {
			bilinearInterpolation(a);
		}
	}
	else if (n==2) {
		cout<<"Enter image name: ";
		string name;
		cin>>name;
		Mat a = imread(name);
		translation(a);
	}
	else if (n==3) {
		cout<<"Enter image name: ";
		string name;
		cin>>name;
		Mat a = imread(name);
		rotation(a);
	}
	else if (n==4) {
		cout<<"Enter image name: ";
		string name;
		cin>>name;
		Mat a = imread(name);
		scaling(a);
	}
	else if (n==5) {
		cout<<"Enter image name: ";
		string name;
		cin>>name;
		Mat a = imread(name);
		shear(a);
	}
	else if (n==6) {
		cout<<"Enter image name: ";
		string name;
		cin>>name;
		Mat a = imread(name);
		negative(a);
	}
	else if (n==7) {
		cout<<"Enter image name: ";
		string name;
		cin>>name;
		Mat a = imread(name);
		logTranformation(a);
	}
	else if (n==8) {
		cout<<"Enter image name: ";
		string name;
		cin>>name;
		Mat a = imread(name);
		gammaTranformation(a);
	}
	else if (n==9) {
	  	bitplaneSclicing();
	}
	else if (n==10) {
	 	tiepoint();
	}
	else if (n==11) {
		histogrmEqulisation();
	}
	else if (n==12) {cout<<"Enter the location of the image"<<endl;
		adaptiveHistogram();
	}
	else if (n==13) {cout<<"Enter the location of the image"<<endl;
		histogrmMatching();
	  
	}
	else if (n==14) {
		cout<<"Enter image name: ";
		string name;
		cin>>name;
		Mat a = imread(name);
		piecewiseTransformation(a);
	}
	else if (n==0) {
	  cout<<"Exiting!! "<<endl;
	}
	else  {
	  cout<<"Please enter a valid number\n"<<endl;
	}
	waitKey();
  return 0;
}
 
