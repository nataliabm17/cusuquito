// ROS includes
#include <ros/ros.h>
#include <sensor_msgs/image_encodings.h>

// CvBridge imports
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

// OpenCV imports
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utility.hpp>

// C++ includes
#include <fstream>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <vector>
#include <string>
#include <cstdio>
#include <cmath>

using namespace cv;
using namespace std;

Mat HistogramMap(Mat image);
Mat claheEqualization(Mat input);
Mat LogFilter(Mat input), Exponential(Mat input);
Mat medianBlurFilter(Mat input);
Mat bilatFilter(Mat input);
Mat adaptiveThreshold(Mat input);
vector<Point2f> DetectedObjects(Mat input);
int addfiles(string dir, vector<string> &files);
Mat sobelFilter(Mat input);
Point2f colorDetection(Mat input);
Point2f coord;
<<<<<<< HEAD

#define PI 3.141592653589

void FunctionToHandlePublishedImage(const sensor_msgs::ImageConstPtr& msg);

image_transport::Subscriber sub_ourImageTopic_;
cv_bridge::CvImagePtr cv_ptr;
cv_bridge::CvImagePtr cv_ptr_display;

//namespace enc = sensor_msgs::image_encodings;
int contadorDeImagenesRecibidas=0;
=======
float angle(float distance);
>>>>>>> 0df9ff9d2f9a2ff03122af899f83db18eb9c3ea2

#define PI 3.141592653589

main(int argc, char** argv){
<<<<<<< HEAD
=======
  Mat originalHistogram, EqHistogram; //Histograms of readImage and after equalization
  Mat claheEquaIm;         //Original and equalized images
  Mat expoIm, bilatFilterIm, LogFilterIm, dilateIm, erodeIm, medianBlurIm; //mat objects for applied filters
  Mat adapThresholdIm;             //thresholding techniques applied
  Mat PatternIm, adapContoursIm; //mat objects for identification in images
  Mat sobelFiltIm;
  float center = 640;
  float result;
  //Reading images' directory and its content
  string dir;
  if(argc > 1){
    dir = argv[1];
  }
  vector<Mat> originalIms;                  //Mat vector to store the images
  vector<Mat> perf;                     //Mat vector to store segmented images
  vector<string> files = vector<string>(); //Vector where the files' names are kept
  vector<Point2f> coordinates;
  addfiles(dir,files);

  //Obtaining Mat array with the original images
  for(int i = 0; i<files.size();i++){
    //show images' paths, read them all and get them into a Mat vector
    //push_back adds new elements at the end of the vector
    originalIms.push_back(imread(files[i], IMREAD_COLOR));
  }
>>>>>>> 0df9ff9d2f9a2ff03122af899f83db18eb9c3ea2

    //Levantando del nodo ROS llamado "image_segmentation_node".
    ros::init(argc, argv, "image_segmentation_node");
    ros::NodeHandle nh_;

    //Dandole al nodo la capacidad de recepcion de
    //mensajes con imagenes
    image_transport::ImageTransport it_(nh_);

    //Subscripción al topico "/usb_cam/image_raw", a traves del cual
    //se recibiran las imagenes capturadas por la camara usb de su
    //laptop. Se define un buffer de entrada de máximo 1 imágenes
    sub_ourImageTopic_ = it_.subscribe("/usb_cam/image_raw", 1, FunctionToHandlePublishedImage);
    //La función geoFunctionToHandlePublishedImage se ejecutara cada vez
    //que un mensaje se reciba a través del tópico "/usb_cam/image_raw".

    //Creando ventanas OpenCV para desplegar imagenes
    //windowName0 = "Imagen de Intensidad";
    //windowName1 = "Imagen Segmentada";
	//printf("No me pegué ");
    cvWaitKey(30); //Esta funcion ademas de hacer esperar al
    //programa 30 ms, tambien fuerza a OpenCv a crear
    //inmediatamente las ventanas

    //El valor de X en la función ros::Rate loop_rate(X)
    //indica el número de ciclos "while (ros::ok())" que ROS
    //deberá realizar por segundo aproximadamente. Esta función
    //trabaja en forma conjunta con la función loop_rate.sleep()
    ros::Rate loop_rate(30);

    //ros::ok() es cero cuando ctrl+c es presionado en el teclado.
    //Utilice esa combinación de teclas para salirse del programa.
    while (ros::ok()){
         //Dentro de la funcion "ros::spinOnce()" ROS ejecuta
         //sus funciones. Los mensajes se atenderán solamente
         //dentro de "ros::spinOnce()".
         ros::spinOnce();
         loop_rate.sleep();
	}

    return 0;
}

void FunctionToHandlePublishedImage(const sensor_msgs::ImageConstPtr& msg){
	Mat claheEquaIm;         //Original and equalized images
	Mat adapThresholdIm;             //thresholding techniques applied

    //Extrayendo la imagen rgb del mensaje recibido
//printf(" No me pegué x2 ");
    try{
      //cv_prt es un puntero a una estructura ROS que
      //contiene un puntero a otra estructura OpenCV que
      //a su vez contiene un puntero a la imagen rgb
      //recibida.
	  Mat im1;
	  Mat im2;
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
	  //cv_ptr =toCvCopy(msg, sensor_msgs::image_encodings::BGR8);

<<<<<<< HEAD
      //cv_prt_display es una copia de cv_prt. Esta copia
      //se usará únicamente para visualización
      cv_ptr_display = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);

		im1=cv_ptr->image;
		im2=cv_ptr_display->image;

 		//HistogramMap(originalIms[i]).copyTo(EqHistogram);
    	//Applying CLAHE equalization to each image
    	claheEqualization(im2).copyTo(im2);

	    //Histogram obtained after CLAHE equalization
    	//HistogramMap(claheEquaIm).copyTo(EqHistogram);
    	//adaptiveThreshold(claheEquaIm).copyTo(adapThresholdIm)

    	coord = colorDetection(im2);

		  result = coord.x - center;
    if(result > 0){
      //Object to the right
      result = abs(result);           //distance the car should move to one side or the other
      result = angle(result);         //angle the car should move to the right
      //Send message to move left motor RESULT angles to move to the right

    }else if(result < 0){
      //Object to the left
      result = abs(result);
      result = angle(result)*-1;        //angle the car should move to the left
=======
    //Histogram obtained after CLAHE equalization
    //HistogramMap(claheEquaIm).copyTo(EqHistogram);
    //sobelFilter(claheEquaIm).copyTo(sobelFiltIm);
    //adaptiveThreshold(claheEquaIm).copyTo(adapThresholdIm);


    coord = colorDetection(claheEquaIm);
    /*for(int i=coord.x-50;i<coord.x+50;i++){
        for(int j=coord.y-50;i<coord.y+50;j++){
          Vec3b color = image.at<Vec3b>(Point(i,j));
          image.at<Vec3b>(Point(i,j)) = color;
          //claheEquaIm.at<Vec3b>(j,i)=Vec3b(Point(0,0,255));
        }
    }*/

    //Comparacion de coordenada x con posicion del centro
    result = coord.x - center;
    if(result > 0){
      //Object to the right
      result = abs(result);           //distance the car should move to one side or the other
      result = angle(result);         //angle the car should move to the right
      //Send message to move left motor RESULT angles to move to the right

    }else if(result < 0){
      //Object to the left
      result = abs(result);
      result = angle(result);        //angle the car should move to the left
>>>>>>> 0df9ff9d2f9a2ff03122af899f83db18eb9c3ea2
      //Send message to move right motor RESULT angles to move to the left
    }else{                          //Object = Result
      //Send message to keep moving with both motors working ======> Envia 0,0 (cero de direccion, cero de angulo)
    }
<<<<<<< HEAD
    	imshow("Coordinates", claheEquaIm); //show the thresholded image
    	waitKey();
    	destroyWindow("Coordinates");
    	//equalizeHist(im2,im2);
    	//namedWindow("Equalized image", WINDOW_AUTOSIZE);
    	//imshow("Equalized image",im2);
    	//cvWaitKey(30);
		

    }
    catch (cv_bridge::Exception& e){
      ROS_ERROR("lalalalalalalcv_bridge exception: %s", e.what());
      return;
    }

    contadorDeImagenesRecibidas++;
}


=======
    imshow("Coordinates", claheEquaIm); //show the thresholded image
    waitKey();
    destroyWindow("Coordinates");

  }
  return 0;
}



>>>>>>> 0df9ff9d2f9a2ff03122af899f83db18eb9c3ea2
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
  //cv::cvtColor(equalizedIm, equalizedIm, CV_BGR2HSV);
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
  int LowB, HighB, LowG, HighG, LowR, HighR;
  LowB, LowG = 0;
  HighB = 190;
  HighG = 155;
  LowR = 92;
  HighR = 255;
  Mat imgThreshold;
  int iLastX = -1;
  int iLastY = -1;
  Mat imgLines = Mat::zeros(input.size(), CV_8UC3);
  Point2f finalPos;

  inRange(input,Scalar(LowB,LowG,LowR),Scalar(HighB,HighG,HighR),imgThreshold);

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
<<<<<<< HEAD

  if(finalPos.x == 0 && finalPos.y == 0){
        //Manda msj de 'NO ENCUENTRA BANDERA': gira cada 45 grados tomando una foto
		//hasta encontrar la bandera
  }
=======
>>>>>>> 0df9ff9d2f9a2ff03122af899f83db18eb9c3ea2
  imshow("Thresholded Image", imgThreshold); //show the thresholded image
  waitKey();
  cout << finalPos.x << ", " << finalPos.y << endl;
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

float angle(float distance){
  float output;
  float hfov = (54.61*(PI/180));       //angulo de hfov en radianes
  float b = 1280;                      //pixeles totales en micrometros
  float cita = hfov/2;
  output = (atan((2*distance*tan(cita*(PI/180.0)))/b)*180/PI);
  return output;                       //returns angle from center of the camera to one side or the other
}
