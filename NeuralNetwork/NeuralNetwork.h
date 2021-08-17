#ifndef __NN_NEURAL_NETWORK_H__
#define __NN_NEURAL_NETWORK_H__

#include "Matrix.h"
#include <initializer_list>
#include <vector>

namespace nn
{

enum class ActivationFunction
{
	SIGMOID, 
	SOFTMAX
};

enum class LossFunction
{
	MEAN_SQUARE_ERROR, 
	SOFTMAX_CROSS_ENTROPY
};

class NeuralNetwork
{
public:
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
		ActivationFunction _af;
		
		Layer(int nInputs, int nOutputs, ActivationFunction af);
		
		void activate();
		void activation_sigmoid();
		void activation_softmax();
	};
	
	struct LayerInfo
	{
		int units;
		ActivationFunction af;
	};
	
	NeuralNetwork(int nInputs, std::initializer_list<LayerInfo> layers, LossFunction lf)
	{
		for (std::initializer_list<LayerInfo>::iterator it = layers.begin(); it != layers.end(); ++it)
		{
			_layers.push_back(Layer(nInputs, it->units, it->af));
			nInputs = it->units;
		}
		
		_lf = lf;
		
		randomize();
	}
	
	using LayerList = std::vector<Layer>;
	const LayerList &layers() const { return _layers; }
	
	void randomize();
	
	void feed_forward(const nn::Matrix &input);
	
	nn::Matrix::value_type compute_loss(const nn::Matrix &target);
	nn::Matrix::value_type compute_loss_mean_square_error(const nn::Matrix &target);
	nn::Matrix::value_type compute_loss_softmax_cross_entropy(const nn::Matrix &target);
	
	void back_propagation(const nn::Matrix &input, const nn::Matrix &target);
	
	void mutate(double rate);
	
protected:
	LayerList _layers;
	LossFunction _lf;
};

}; // namespace nn

#endif // __NN_NEURAL_NETWORK_H__
