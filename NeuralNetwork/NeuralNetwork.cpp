#include "NeuralNetwork.h"
#include <cmath>
#include <random>

namespace nn
{
std::mt19937 _randomgenerator;
std::uniform_real_distribution<float> _randomdistribution(-1.0, 1.0);

NeuralNetwork::NeuralNetwork(int nInputs, int nHidden, int nOutputs)
{
	_layers.push_back(Layer(nInputs, nHidden));
	_layers.push_back(Layer(nHidden, nOutputs));
	
	
	for (auto &layer : _layers)
	{
		nn::map(layer._weights, [&](nn::Matrix::value_type v) { return _randomdistribution(_randomgenerator); });
		nn::map(layer._biases, [&](nn::Matrix::value_type v) { return _randomdistribution(_randomgenerator); });
	}
}

NeuralNetwork::NeuralNetwork::Layer::Layer(int nInputs, int nOutputs) : 
	_weights(nOutputs, nInputs), 
	_biases(nOutputs, 1), 
	_output(nOutputs, 1)
{
}

void NeuralNetwork::feedforward(const std::vector<const Sample *> &samples)
{
	nn::Matrix output;
	output.resize(samples.front()->_target.numRows(), samples.front()->_target.numColumns());
	
	for (const Sample *sample : samples)
	{
		feedforward(sample->_input, output);
	}
}

void NeuralNetwork::feedforward(const nn::Matrix &input)
{
	const nn::Matrix *payload = &input;
	
	for (auto &layer : _layers)
	{
		nn::product(layer._weights, *payload, layer._output);
		nn::add(layer._biases, layer._output, layer._output);
		nn::map(layer._output, [] (nn::Matrix::value_type v) { return 1.0f / (1.0f + std::expf(-v)); });
		payload = &layer._output;
	}
}

void NeuralNetwork::feedforward(const nn::Matrix &input, nn::Matrix &output)
{
	feedforward(input);
	nn::copy(_layers.back()._output, output);
}

void NeuralNetwork::backpropagation(const nn::Matrix &input, const nn::Matrix &target)
{
	/* nn::Matrix guess(target.numRows(), target.numColumns());
	feedforward(input, guess);
	
	nn::Matrix error(target.numRows(), target.numColumns());
	nn::subtract(target, guess, error);
	
	for (auto &layer : _layers)
	{
		nn::transpose(layer._weights, layer._tweights);
	} */
}

}; // namespace nn

