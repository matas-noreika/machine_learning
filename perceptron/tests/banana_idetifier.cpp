/*
 * Programmer: Matas Noreika 2025-10-25 01:17
 * Purpose: Application that will use a perceptron with predefined weights to identify a banana based on weight and length
*/

#include <iostream>
#include <cstdlib> //rand
#include "Perceptron.h"

#define WEIGHTMAX 119
#define LENGTHMAX 21
#define NUMDATAPOINTS 500000

int main(int argc, char** argv){
  
  //set weight target as <= 118 and length target as <= 20
  std::vector<double> trainingData = {10, 15};
  std::vector<double> inputData = {50, 15}; //set starter training data

  //create a perceptron with 2 inputs and learning rate of 0.5
  Perceptron perceptron(2, 0.01);
  
  //train perceptron on random data set
  for(long i = 0; i < NUMDATAPOINTS; i++){

    trainingData[0] = rand() % (2*WEIGHTMAX);
    trainingData[1] = rand() % (2*LENGTHMAX);
    
    //print training data
    /*std::cout <<  '{' << trainingData[0]
              << ',' << trainingData[1] << "}\n";*/

    bool expected_output = (trainingData[0] <= WEIGHTMAX) && (trainingData[1] <= LENGTHMAX);
    
    /*//print associated expected Output
    std::cout << "Expected Output: " << expected_output << '\n';*/
    
    //train of training data
    perceptron.fit(trainingData, expected_output);
  }
  //print out final weights and bias
  perceptron.print_weights();
  
  double correct = 0;

  for(int i = 0; i < NUMDATAPOINTS; i++){
    //generate random data
    inputData[0] = rand() % (2*WEIGHTMAX);
    inputData[1] = rand() % (2*LENGTHMAX);
    
    bool expected_output = (inputData[0] <= WEIGHTMAX) && (inputData[1] <= LENGTHMAX);

    if(perceptron.predict(inputData) == expected_output){
      correct++;
    }

  }

  std::cout << "Accuracy: " << (correct/NUMDATAPOINTS) * 100 << '\n';
 
  return 0;
}
