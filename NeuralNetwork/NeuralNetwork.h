#ifndef __NN_NEURAL_NETWORK_H__
#define __NN_NEURAL_NETWORK_H__

#include "Matrix.h"
#include <vector>

namespace nn
{

class NeuralNetwork
{
public:
	NeuralNetwork(int nInputs, int nHidden, int nOutputs);
	
	struct Sample
	{
		nn::Matrix _input;
		nn::Matrix _target;
	};
	
	struct Layer
	{
		nn::Matrix _weights;
		nn::Matrix _biases;
		nn::Matrix _output;
		
		Layer(int nInputs, int nOutputs);
	};
	
	using LayerList = std::vector<Layer>;
	const LayerList &layers() const { return _layers; }
	
	void feedforward(const std::vector<const Sample *> &samples);
	void feedforward(const nn::Matrix &input);
	void feedforward(const nn::Matrix &input, nn::Matrix &output);
	
	void backpropagation(const nn::Matrix &input, const nn::Matrix &target);
	
protected:
	LayerList _layers;
};

}; // namespace nn

#endif // __NN_NEURAL_NETWORK_H__
