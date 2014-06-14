/**********************************************************************************************************
  HEADER FILE: Backpropagation Neural Network Implementation
  File: bpnet.h
  Version: 0.1
  Copyright(C) NeuroAI (http://www.learnartificialneuralnetworks.com)
  Documentation:http://www.learnartificialneuralnetworks.com/neural-network-software/backpropagation-source-code/
  NeuroAI Licence:
  Permision granted to use this source code only for non commercial and educational purposes.
  You can distribute this file but you can't take ownership of it or modify it.
  You can include this file as a link on any website but you must link directly to NeuroAI website
  (http://www.learnartificialneuralnetworks.com)
  Written by Daniel Rios <daniel.rios@learnartificialneuralnetworks.com> , June 2013

 /*********************************************************************************************************/


#ifndef BPNET_H
#define BPNET_H

/*********************************Structure representing a neuron******************************/
struct neuron
{
    float *weights; // neuron input weights or synaptic connections
    float *deltavalues; //neuron delta values
    float output; //output value
    float gain;//Gain value
    float wgain;//Weight gain value

    neuron();//Constructor
    ~neuron();//Destructor
    void create(int inputcount);//Allocates memory and initializates values
};
/**************************************Structure representing a layer******************************/
struct layer
{
    neuron **neurons;//The array of neurons
    int neuroncount;//Contains the total number of neurons
    float *layerinput;//The layer input
    int inputcount;//The total count of elements in layerinput

    layer();//Object constructor. Initializates all values as 0

    ~layer();//Destructor. Frees the memory used by the layer

    void create(int inputsize, int _neuroncount);//Creates the layer and allocates memory
    void calculate();//Calculates all neurons performing the network formula
};
/********************************Structure Representing the network********************************/
class bpnet
{
private:
    layer m_inputlayer;//input layer of the network
    layer m_outputlayer;//output layer..contains the result of applying the network
    layer **m_hiddenlayers;//Additional hidden layers
    int m_hiddenlayercount;//the count of additional hidden layers

public:
//function tu create in memory the network structure
    bpnet();//Construction..initialzates all values to 0
    ~bpnet();//Destructor..releases memory
    //Creates the network structure on memory
    void create(int inputcount,int inputneurons,int outputcount,int *hiddenlayers,int hiddenlayercount);

    void propagate(const float *input);//Calculates the network values given an input pattern
    //Updates the weight values of the network given a desired output and applying the backpropagation
    //Algorithm
    float train(const float *desiredoutput,const float *input,float alpha, float momentum);

    //Updates the next layer input values
    void update(int layerindex);

    //Returns the output layer..this is useful to get the output values of the network
    inline layer &getOutput()
    {
        return m_outputlayer;
    }

};

#endif // BPNET_H
