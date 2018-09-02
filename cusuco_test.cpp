// ROS includes
#include <ros/ros.h>
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/Float32.h>

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

#define PI 3.141592653589

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
std_msgs::Float32 ang_msg;
float center = 640;
float result;
int areaFinal, areaDetect; /////////////////////////////////////DECLARAR
int lowB, highB, lowG, highG, lowR, highR;

void FunctionToHandlePublishedImage(const sensor_msgs::ImageConstPtr& msg);

image_transport::Subscriber sub_ourImageTopic_;
cv_bridge::CvImagePtr cv_ptr;
cv_bridge::CvImagePtr cv_ptr_display;

//namespace enc = sensor_msgs::image_encodings;
int contadorDeImagenesRecibidas=0;

main(int argc, char** argv){
  leerParametrosDeControlDeArchivoDeTexto();
  //Levantando del nodo ROS llamado "image_segmentation_node".
  ros::init(argc, argv, "image_segmentation_node");
  ros::NodeHandle nh_;

  //Dandole al nodo la capacidad de recepcion de
  //mensajes con imagenes
  image_transport::ImageTransport it_(nh_);

  //Subscripción al topico "/usb_cam/image_raw", a traves del cual se recibiran las imagenes capturadas por la camara usb de su
  //laptop. Se define un buffer de entrada de máximo 1 imágenes
  sub_ourImageTopic_ = it_.subscribe("/usb_cam/image_raw", 1, FunctionToHandlePublishedImage);
  //La función geoFunctionToHandlePublishedImage se ejecutara cada vez
  //que un mensaje se reciba a través del tópico "/usb_cam/image_raw".

  //Creando ventanas OpenCV para desplegar imagenes
  //windowName0 = "Imagen de Intensidad";
  //windowName1 = "Imagen Segmentada";
  cvWaitKey(30); //Esta funcion ademas de hacer esperar al programa 30 ms, tambien fuerza a OpenCv a crear inmediatamente las ventanas

  //El valor de X en la función ros::Rate loop_rate(X) indica el número de ciclos "while (ros::ok())" que ROS
  //deberá realizar por segundo aproximadamente. Esta función trabaja en forma conjunta con la función loop_rate.sleep()
  ros::Rate loop_rate(30);

  //ros::ok() es cero cuando ctrl+c es presionado en el teclado.
  //Utilice esa combinación de teclas para salirse del programa.
  while (ros::ok()){
   //Dentro de la funcion "ros::spinOnce()" ROS ejecuta sus funciones. Los mensajes se atenderán solamente dentro de "ros::spinOnce()".
   ros::spinOnce();
   loop_rate.sleep();
	}
  return 0;
}

void leerParametrosDeControlDeArchivoDeTexto(){
  int numeroDeDatosLeidos;
  FILE *archivo;
  char d1[256], d2[256];
  int res;

  printf("Leyendo los datos de entrada:\n");

  //Abriendo archivo en mode de lectura
  char nombreDeArchivo[256]="cusuco_cv.txt";
  archivo = fopen(nombreDeArchivo, "r");
  if (!archivo) {
    printf("No se pudo abrir el archivo: current_control_parameters.txt\n");
    exit(1);
  }

  //Leyendo datos linea por linea
  //Leyendo path y nombre de imagen de entrada
  res=fscanf(archivo, "%s %s\n", d1, d2);
  if (res==0) {printf("Error leyendo dato No. %d de archivo de parametros de control\n", numeroDeDatosLeidos); exit(0);}
  areaFinal=(int)d2;
  cout << "Area final: " << areaFinal << endl;
  numeroDeDatosLeidos++;

  res=fscanf(archivo, "%s %s\n", d1, d2);
  if (res==0) {printf("Error leyendo dato No. %d de archivo de parametros de control\n", numeroDeDatosLeidos); exit(0);}
  areaDetect=(int)d2;
  cout << "Area deteccion: " << areaDetect << endl;
  numeroDeDatosLeidos++;

  res=fscanf(archivo, "%s %s\n", d1, d2);
  if (res==0) {printf("Error leyendo dato No. %d de archivo de parametros de control\n", numeroDeDatosLeidos); exit(0);}
  lowB=(int)d2;
  cout << "lowB: " << lowB << endl;
  numeroDeDatosLeidos++;

  res=fscanf(archivo, "%s %s\n", d1, d2);
  if (res==0) {printf("Error leyendo dato No. %d de archivo de parametros de control\n", numeroDeDatosLeidos); exit(0);}
  lowG=(int)d2;
  cout << "lowG: " << lowG << endl;
  numeroDeDatosLeidos++;

  res=fscanf(archivo, "%s %s\n", d1, d2);
  if (res==0) {printf("Error leyendo dato No. %d de archivo de parametros de control\n", numeroDeDatosLeidos); exit(0);}
  cout << "lowR: " << lowR << endl;
  numeroDeDatosLeidos++;

  res=fscanf(archivo, "%s %s\n", d1, d2);
  if (res==0) {printf("Error leyendo dato No. %d de archivo de parametros de control\n", numeroDeDatosLeidos); exit(0);}
  highB=(int)d2;
  cout <<"highB: " << highB << endl;
  numeroDeDatosLeidos++;

  res=fscanf(archivo, "%s %s\n", d1, d2);
  if (res==0) {printf("Error leyendo dato No. %d de archivo de parametros de control\n", numeroDeDatosLeidos); exit(0);}
  highG=(int)d2;
  cout << "highG: " << highG << endl;
  numeroDeDatosLeidos++;

  res=fscanf(archivo, "%s %s\n", d1, d2);
  if (res==0) {printf("Error leyendo dato No. %d de archivo de parametros de control\n", numeroDeDatosLeidos); exit(0);}
  highR=(int)d2;
  cout << "highR: " << highR << endl;
  numeroDeDatosLeidos++;

  fclose(archivo);
}

void FunctionToHandlePublishedImage(const sensor_msgs::ImageConstPtr& msg){
	Mat claheEquaIm;         //Original and equalized images
	Mat adapThresholdIm;             //thresholding techniques applied
  //Extrayendo la imagen rgb del mensaje recibido
  //printf(" No me pegué x2 ");
  try{
    //cv_prt es un puntero a una estructura ROS que contiene un puntero a otra estructura OpenCV que
    //a su vez contiene un puntero a la imagen rgb recibida.
	  Mat im1;
	  Mat im2;
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
	  //cv_ptr =toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    //cv_prt_display es una copia de cv_prt. Esta copia se usará únicamente para visualización
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
    if(coord.x == 0 && coord.y == 0){
      result = 999.0;
    }else if(coord.x == 777777 && coord.y == 777777){
      result = 888.0;
    }else{
      if(result > 0){
        //Object to the right
        result = abs(result);           //distance the car should move to one side or the other
        result = angle(result);         //angle the car should move to the right
        //Send message to move left motor RESULT angles to move to the right
      }else if(result < 0){
        //Object to the left
        result = abs(result);
        result = angle(result)*-1;        //angle the car should move to the left
        //Send message to move right motor RESULT angles to move to the left
      }else{                          //Object = Result
        result = 0;
        //Send message to keep moving with both motors working ======> Envia 0 (bien centrado)
      }
    }
    ang_msg = result;
    imshow("Coordinates", claheEquaIm); //show the thresholded image
    waitKey();
    destroyWindow("Coordinates");
    //equalizeHist(im2,im2);
    //namedWindow("Equalized image", WINDOW_AUTOSIZE);
    //imshow("Equalized image",im2);
    //cvWaitKey(30);
  }catch (cv_bridge::Exception& e){
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }
  //contadorDeImagenesRecibidas++;
  angle_pub.publish(ang_msg);
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

Point2f colorDetection(Mat input){
  Mat output;
  Mat imgThreshold;
  int iLastX = -1;
  int iLastY = -1;
  Mat imgLines = Mat::zeros(input.size(), CV_8UC3);
  Point2f finalPos;

  inRange(input,Scalar(lowB,lowG,lowR),Scalar(highB,highG,highR),imgThreshold);
  //morphological opening (removes small objects from the foreground)
  erode(imgThreshold, imgThreshold, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
  dilate(imgThreshold, imgThreshold, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
  //morphological closing (removes small holes from the foreground)
  dilate(imgThreshold, imgThreshold, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
  erode(imgThreshold, imgThreshold, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));

  //Calculate the moments of the thresholded image
  Moments oMoments = moments(imgThreshold);
  double dM01 = oMoments.m01;
  double dM10 = oMoments.m10;
  double dArea = oMoments.m00;
  cout << "Area detectada: " << dArea << endl;   //PRUEBAAAAAAAAAAAAAAAAAA
  // if the area <= 10000, I consider that there are no object in the image and it's because of the noise, the area is not zero
  //************HACER PRUEBAS CON LA BANDERA CON LA DISTANCIA DE 13 METROS*************
  if(dArea > areaFinal){
    //DETENGASE
    finalPos.x = 777777;
    finalPos.y = 777777;
  }else if(dArea > areaDetect){
    //calculate the position of the ball
    finalPos.x = dM10 / dArea;
    finalPos.y = dM01 / dArea;
  }else{
    cout << "No encontre nada, coordenadas (0,0)" << endl;
  }

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
  float b = 1280;                      //pixeles totales en x
  float cita = hfov/2;
  output = (atan((2*distance*tan(cita*(PI/180.0)))/b)*180/PI);
  return output;                       //returns angle from center of the camera to one side or the other
}
