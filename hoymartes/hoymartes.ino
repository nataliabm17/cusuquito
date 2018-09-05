//∗
//∗ Blink
//∗
//∗ Turns on an LED on for one second, then off ffor one second, repeatedly. ∗/

#include "Arduino.h"
#include <Wire.h>
#include <Encoder.h>
#define MOTOR_IN1 9
#define MOTOR_IN2 10
#define MOTOR_IN3 5
#define MOTOR_IN4 6
#include <SFE_BMP180.h>
//este header es para el barometro

#define SLAVE_ADDRESS 0x04
#define ADDRESS_SENSOR 0X77

int number = 0;
int state = 0 ;
int vel_final;
boolean   condicion = false;
void sendData( ) ;
void receiveData (int byteCount);

Encoder knobLeft(3, 8);
Encoder knobRight(2, 7);

long positionLeft  = -999;
long positionRight = -999;
int compensacion = 50; //compensacion a la calibracion de la llantas 
                       // pensar en la compensacion por medio del angulo\\
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
//DECLARACION DE VARIABLES PARA EL BAROMETRO
SFE_BMP180 pressure;

double baseline; // baseline pressure
int tomas[15];
boolean mandardatos=false;
// int vamoaver=0;
/////////////////////////////////////////////////////////////////////////////////////////////////////////////

void setup(){
pinMode (13 , OUTPUT);//pin que energiza la odroid
Serial.begin (38400); // start serial for output \\ a mayor vel mejor preciosion
// initialize i2c as slave
Wire.begin(SLAVE_ADDRESS);
//Wire.begin(ADDRESS_SENSOR);


// define callbacks for i2c communication
Wire.onReceive(receiveData);
Wire.onRequest(sendData);
Serial.println("Ready !");
pinMode(MOTOR_IN1, OUTPUT);
  pinMode(MOTOR_IN2, OUTPUT);

  
//presion inicial del barometro
  if (pressure.begin()){
    baseline = getPressure();
  //presion base
  }
  
}




void loop(){

  
}

///////////////////////////////
//FUNCIONES//
//////////////////////////////
void chequeoBarometro(){
  double a,P;
  // Get a new pressure reading:
 int vamoaver=0;
  for(int i=0; i<15;i++){
    tomas[i]=getPressure(); //vamos llenando el vector de tomas de presiones
    delay(1000);//en intervalos de 1 segundo revisara una nueva presion
    }
  //viene la comparacion mistica

     for(int j=0;j<15;j++){
       if(tomas[14]>tomas[j]){
        vamoaver++;
        }
     }
    
 if(vamoaver==14 ){//los nuevos valores tienen una presion menor a la actual
  //la presion disminuye con la altura
  //encender el pin ya o muy inseguro?
  
  if(tomas[14]< baseline+5 && tomas[14]>baseline-5){//rango de presion atmosferica
  digitalWrite(13,HIGH);
  mandardatos=true; //poner un if dentro del loop .....si mandardatos es true entonces empiece a....mandar datos a algun lado de la odroid
  
  }
  
  }
 // else{
 //   chequeoBarometro();
 //   }
 
}
/////////////////////////
//este metodo getPressure es hermano del de chequeo del barometro
double getPressure()
{
  char status;
  double T,P,p0,a;
  status = pressure.startTemperature();
  if (status != 0)
  {
    delay(status);
    status = pressure.getTemperature(T);
    if (status != 0)
    {
      status = pressure.startPressure(3);
      if (status != 0)
      {
      
        delay(status);

        status = pressure.getPressure(P,T);
        if (status != 0)
        {
          return(P);
        }
        else Serial.println("error retrieving pressure measurement\n");
      }
      else Serial.println("error starting pressure measurement\n");
    }
    else Serial.println("error retrieving temperature measurement\n");
  }
  else Serial.println("error starting temperature measurement\n");
}
//////////////////////////////////


///////
//MOVIMIENTO MOTORES
//////


// la funcion da una compemsacion si las cuentas de los encoder no son iguales
void cambio(){
  if(positionLeft != positionRight){
    if(posicionLeft < positionRigth){
    while(positionLeft != positionRight){
      analogWrite(MOTOR_IN2, number);
      analogWrite(MOTOR_IN4, number-compensacion);
       leerEncoder();
    }
  }
    else{
      if(posicionLeft < positionRigth){
         while(positionLeft != positionRight){
           analogWrite(MOTOR_IN2, number-compensacion);
           analogWrite(MOTOR_IN4, number);
           leerEncoder();
      }
      }
    }
  }
}

 void moverMotor(){
  for (int i=0; i<number; i++) {
    analogWrite(MOTOR_IN2, i);
    analogWrite(MOTOR_IN4, i);
      leerEncoder();
    delay(10);
    while(i==number-1){
      analogWrite(MOTOR_IN2, number);
      analogWrite(MOTOR_IN4, number);
       leerEncoder();
       cambio();
    }
  }
}


/////////////////
//COMUNICACION I2C
/////////////////////////

// callback for received data
void receiveData (int byteCount) {
  while ( Wire.available()){
    for(i=0,i<2,i++){
    number[i] = Wire.read( );
        }
      }
     }
  


// callback for sending data
  void sendData(){
    Wire.write(number) ;
  }
  
 
  delay(1000);








//////////////////////
///// ENCODER ////// 
 ///////////////////////

  void leerEncoder(){
  long newLeft, newRight;
  newLeft = knobLeft.read();
  newRight = knobRight.read();
  if (newLeft != positionLeft || newRight != positionRight) {
    Serial.print("Left = ");
    Serial.print(newLeft);
    Serial.print(", Right = ");
    Serial.print(newRight);
    Serial.println();
    positionLeft = newLeft;
    positionRight = newRight;
  }
  // if a character is sent from the serial monitor,
  // reset both back to zero.
  if (Serial.available()) {
    Serial.read();
    Serial.println("Reset both knobs to zero");
    knobLeft.write(0);
    knobRight.write(0);
  }
  }
  
  
  ///////////////////////////
  //ALGORITMO BANDERA ROJA
  /////////////////////////
  
  

