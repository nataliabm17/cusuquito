#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utility.hpp>
using namespace cv;
#include <dirent.h>   //Used in the program to read files in the directory
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <tiffio.h>
#include <stdexcept>
#include <vector>
#include <string>
using namespace std;

Mat HistogramMap(Mat image);
Mat claheEqualization(Mat input);
Mat LogFilter(Mat input), Exponential(Mat input);
Mat medianBlurFilter(Mat input);
Mat dilationOperation(Mat input), erosionOperation(Mat input);
Mat bilatFilter(Mat input);
Mat adaptiveThreshold(Mat input);
vector<Point2f> DetectedObjects(Mat input);
Mat findingContours(Mat input);
int addfiles(string dir, vector<string> &files);
Mat sobelFilter(Mat input);


main(int argc, char** argv){
  Mat originalHistogram, EqHistogram; //Histograms of readImage and after equalization
  Mat claheEquaIm;         //Original and equalized images
  Mat expoIm, bilatFilterIm, LogFilterIm, dilateIm, erodeIm, medianBlurIm; //mat objects for applied filters
  Mat adapThresholdIm;             //thresholding techniques applied
  Mat PatternIm, adapContoursIm; //mat objects for identification in images
  Mat sobelFiltIm;
  //Reading images' directory and its content
  string dir;
  if(argc > 1){
    dir = argv[1];
  }
  vector<Mat> originalIms;                  //Mat vector to store the images
  vector<Mat> segIms;                     //Mat vector to store segmented images
  vector<string> files = vector<string>(); //Vector where the files' names are kept
  addfiles(dir,files);

  //Obtaining Mat array with the original images
  for(int i = 0; i<files.size();i++){
    //show images' paths, read them all and get them into a Mat vector
    //push_back adds new elements at the end of the vector
    originalIms.push_back(imread(files[i], IMREAD_COLOR));
  }

  for(int i = 0; i< originalIms.size();i++){
    //find the histogram for each image
    //HistogramMap(originalIms[i]).copyTo(EqHistogram);

    //Applying CLAHE equalization to each image
    claheEqualization(originalIms[i]).copyTo(claheEquaIm);

    //Histogram obtained after CLAHE equalization
    //HistogramMap(claheEquaIm).copyTo(EqHistogram);

    //sobelFilter(claheEquaIm).copyTo(sobelFiltIm);

    adaptiveThreshold(claheEquaIm).copyTo(adapThresholdIm);
    //Applying median blur after adaptive thresholding to remove black and white small noise
    //medianBlurFilter(adapThresholdIm).copyTo(medianBlurIm);
    //Opening operation to remove additional noise
    //erosionOperation(medianBlurIm).copyTo(erodeIm);
    //dilationOperation(erodeIm).copyTo(dilateIm);


    }

  return 0;
}

//Function for directory reading and image retreiving
int addfiles(string dir, vector<string> &files){   //receives the directory and a vector the dir's files
  DIR *directory;           //directory stream
  struct dirent *dirptr;    //structure which includes file serial number and filename string of entry
  if((directory = opendir(dir.c_str())) == NULL){
    cout << "Error opening directory " << endl;
    return errno;          //error that ends the program
  }
  while((dirptr = readdir(directory)) != NULL){
    if(!strcmp(dirptr->d_name, ".") || !strcmp(dirptr->d_name,"..")){
        //do nothing
    }else{
      files.push_back(string(dir + dirptr->d_name));  //saves full path directory as a string
      sort(files.begin(),files.end());       //sorts files by filename
    }
  }
  closedir(directory);
  return 0;
}

//image processing functions to make the cells in the image readable
Mat claheEqualization(Mat input){
  if(input.empty()){
    cout << "Image: not found" << endl;
  }
  Mat output, labImage, equalizedIm;
  cv::cvtColor(input, labImage, CV_BGR2Lab);
  vector<Mat> sections(3);                        //Division of the Mat object in three sections
  split(labImage,sections);
  Ptr<CLAHE> clahe = createCLAHE();
  clahe->setClipLimit(16);                    //Division of the input image asigning them to the sections matrix            //orig: 24
  clahe->apply(sections[0], output);              //Application of the CLAHE method, selecting Lightness component
  output.copyTo(sections[0]);
  merge(sections,labImage);
  cv::cvtColor(labImage, equalizedIm, CV_Lab2BGR);
  cv::cvtColor(equalizedIm, equalizedIm, CV_BGR2GRAY);
  namedWindow("CLAHEqualized Image", WINDOW_AUTOSIZE);
  imshow("CLAHEqualized Image",equalizedIm);
  waitKey();
  destroyWindow("CLAHEqualized Image");
  return equalizedIm;
}

Mat HistogramMap(Mat image){
  int hist_w = 512;
  int hist_h = 400;
  int histogram[256];
  int max = histogram[0];
  for(int i = 0; i < 256; i++){
    histogram[i] = 0;     //initialization of histogram array with zeros
  }
  for(int row = 0; row < image.rows; row++){
    for(int column = 0; column < image.cols; column++){
      histogram[(int)image.at<uchar>(row,column)]++;  //calculation of number of pixels for each intensity value
    }
  }
  //drawing the histogram map
  int bin_w = cvRound((double) hist_w/256);
  Mat histImage(hist_h, hist_w, CV_8UC1, Scalar(255,255,255));
  for(int i=0;i<256;i++){
    if(max<histogram[i]){
      max=histogram[i];
    }
  }
  //normalizing the histogram
  for(int i=0;i<256;i++){
    histogram[i]=((double)histogram[i]/max)*histImage.rows;
  }
  for(int i=0;i<256;i++){
    line(histImage, Point(bin_w*(i), hist_h), Point(bin_w*(i), hist_h - histogram[i]), Scalar(0,0,0),1,8,0);
  }
  cv::convertScaleAbs(histImage,histImage);
  cv::normalize(histImage,histImage,0,255,cv::NORM_MINMAX);

  namedWindow("Histogram", CV_WINDOW_AUTOSIZE);
  imshow("Histogram", histImage);
  waitKey();
  destroyWindow("Histogram");    //erasing image to use several times
  return histImage;
}

Mat sobelFilter(Mat input){
  Mat output;
  int scale = 1;
  int delta = 0;
  int ddepth = CV_16S;
  Mat grad_x, grad_y, abs_grad_x, abs_grad_y;
  Sobel(input, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
  Sobel(input, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
  convertScaleAbs(grad_x, abs_grad_x);
  convertScaleAbs(grad_y, abs_grad_y);
  addWeighted(abs_grad_x, 0.5,abs_grad_y, 0.5, 0,output);
  namedWindow("sobel Filter", WINDOW_AUTOSIZE);
  imshow("sobel Filter", output);
  waitKey();
  destroyWindow("sobel Filter");
  return output;
}

Point2f colorDetection(Mat input){
  Mat output;
  int LowH, HighH, LowS, HighS, LowV, HighV;
  Mat imgHSV, imgThreshold;
  int iLastX = -1;
  int iLastY = -1;
  Mat imgLines = Mat::zeros(input.size(), CV_8UC3 );
  Point2f finalPos;

  cvtColor(input,imgHSV,COLOR_BGR2HSV);
  inRange(imgHSV,Scalar(LowH,LowS,LowV),Scalar(HighH,HighS,HighV),imgThreshold);

  //morphological opening (removes small objects from the foreground)
  erode(imgThreshold, imgThreshold, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
  dilate( imgThreshold, imgThreshold, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );

  //morphological closing (removes small holes from the foreground)
  dilate( imgThreshold, imgThreshold, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
  erode(imgThreshold, imgThreshold, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );

  //Calculate the moments of the thresholded image
  Moments oMoments = moments(imgThreshold);

  double dM01 = oMoments.m01;
  double dM10 = oMoments.m10;
  double dArea = oMoments.m00;

    // if the area <= 10000, I consider that there are no object in the image and it's because of the noise, the area is not zero

    //************HACER PRUEBAS CON LA BANDERA CON LA DISTANCIA DE 13 METROS*************
  if (dArea > 10000)
  {
    //calculate the position of the ball
    finalPos.x = dM10 / dArea;
    finalPos.y = dM01 / dArea;
  }

  imshow("Thresholded Image", imgThresholded); //show the thresholded image
  waitKey(30);
  return finalPos;
}

Mat adaptiveThreshold(Mat input){
  Mat output;
  adaptiveThreshold(input, output, 100, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 41, 15);
  namedWindow("Adaptive threshold applied", WINDOW_AUTOSIZE);
  imshow("Adaptive threshold applied", output);
  waitKey();
  return output;
}
Mat medianBlurFilter(Mat input){
  Mat output;
  medianBlur(input,output,9);
  namedWindow("median Blur", WINDOW_AUTOSIZE);
  imshow("median Blur", output);
  waitKey();
  destroyWindow("median Blur");
  return output;
}
Mat erosionOperation(Mat input){  //add //a little bit more useful //darkens
  Mat output;
  float erosion_size=1.78;
  Mat element = getStructuringElement(cv::MORPH_CROSS, cv::Size(2*(erosion_size + 0.3), 2*(erosion_size + 0.3)), cv::Point(erosion_size, erosion_size));
  erode(input,output, element);
  namedWindow("Erosion result", WINDOW_AUTOSIZE);
  imshow("Erosion result", output);
  waitKey();
  destroyWindow("Erosion result");
  return output;
}
Mat dilationOperation(Mat input){ //substract  //not useful in these case //brightens
  Mat output;
  float dilation_size=0.938;
  Mat element = getStructuringElement(cv::MORPH_CROSS, cv::Size(2*dilation_size - 0.3,2*dilation_size - 0.3), cv::Point(dilation_size, dilation_size));
  dilate(input,output, element);
  Mat kernel = (Mat_<float>(3,3) << 1,1,1,1,-8,1,1,1,1); //approximation of second derivative
  Mat LaplacianIm, resultIm;
  Mat edgeIm = output;
  filter2D(edgeIm,LaplacianIm,CV_32F,kernel);
  cv::subtract(edgeIm, LaplacianIm, output, Mat(), CV_32F);
  output.convertTo(output,CV_8UC1);
  namedWindow("Dilation result", WINDOW_AUTOSIZE);
  imshow("Dilation result", output);
  waitKey();
  destroyWindow("Dilation result");
  return output;
}
