/*
 * Programmer: Matas Noreika 2025-10-25 00:26
 * Purpose: Implementation of Perceptron class
*/

#include "Perceptron.h"

//definition of constructor
Perceptron::Perceptron(int inputs, double lr): inputs(inputs), lr(lr) {
  std::cout << inputs << " " << lr << '\n';
  //call initialiser functions
  init_weights();
  init_bias();
}

void Perceptron::init_weights(){
  for(int i = 0; i < inputs; i++){
    weights.push_back(1); // assign default weight as 1
  }
}

void Perceptron::init_bias(){
  bias = 0; //set bias as 0 by default
}

double Perceptron::calculate_z(std::vector<double>& input_vect){
  //std::cout << "Calcualting Z\n";
  double z_sum = 0;
  for(int i = 0; i < inputs; i++){
    z_sum += input_vect[i] * weights[i];
  }
  return z_sum + bias;
}

double Perceptron::calculate_loss(double prediction, double expected){
  return expected - prediction;
}

double Perceptron::activation(double z){
  return z >= 0; //only output 1 if the value greater than bias offset
}

bool Perceptron::predict(std::vector<double>& input_vect){
  return activation(calculate_z(input_vect));
}

void Perceptron::train(std::vector<double>& input_vect, double expected){
  double loss = calculate_loss(predict(input_vect), expected);
  //std::cout << "Loss: " << loss << '\n';
  //updated weights
  for(int i = 0; i < weights.size(); i++){
    weights[i] += lr * loss * input_vect[i];
  }

  //updated bias
  bias += lr * loss;
}

void Perceptron::fit(std::vector<double>& input_vect, double expected, int epochs){
  for(int i = 0; i < epochs; i++){
    train(input_vect, expected);
    //std::cout << "Finished epoch\n";
  }
}

void Perceptron::print_weights(void){
  
  std::cout << "weights: " << '{';
  for(int i = 0; i < weights.size(); i++){
    std::cout << weights[i] << " ";
  }
  std::cout << "}\n";
  std::cout << "bias: " << bias << '\n';
}
