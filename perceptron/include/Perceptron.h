/*
 * Programmer: Matas Noreika 2025-10-24 23:19
 * Purpose: Class definition of Perceptron
*/

//start of header guard
#ifndef __PRECEPTRON_H__
#define __PRECEPTRON_H__

#include <iostream>
#include <vector> // class definition for vector

//class definition of Perceptron
class Perceptron{
private:
  std::vector<double> weights; //weights assigned to input data
  int inputs; //number of inputs
  double bias; //bias of activation
  double lr; //learning rate of the Perceptron

  //function to assign weights
  void init_weights();

  //function to assign default value for bias
  void init_bias();

  //Function to calculate z = sum(w.x)
  double calculate_z(std::vector<double>& input_vect);
  
  //function to describe the activation function (step function)
  double activation(double z);
  
  //calculates the loss (describes how should weights and bias be shifted) (basic form of gradient decent)
  double calculate_loss(double prediction, double expected);

  //function to train Perceptron on provided data
  void train(std::vector<double>& input_vect, double expected);
public:

  //constructor
  Perceptron(int inputs, double lr);

  //function to output predict based on weights and bias
  bool predict(std::vector<double>& input_vect);

  //function to train Perceptron on provided data
  void fit(std::vector<double>& input_vect, double expected, int epochs = 20);

  void print_weights(void);

};

#endif
//end of header guard
