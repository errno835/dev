#include "NeuralNetwork.h"
#include <cmath>
#include <cassert>
#include <random>

namespace nn
{
std::default_random_engine _random_generator;
std::uniform_real_distribution<float> _minus_one_one_distribution(-1.0f, 1.0f);
std::uniform_real_distribution<float> _zero_one_distribution(0.0f, 1.0f);

void NeuralNetwork::randomize()
{
	for (auto &layer : _layers)
	{
		nn::map(layer._weights, [&](nn::Matrix::value_type v) { return _minus_one_one_distribution(_random_generator); });
		nn::map(layer._biases, [&](nn::Matrix::value_type v) { return _minus_one_one_distribution(_random_generator); });
	}
}

NeuralNetwork::NeuralNetwork::Layer::Layer(int nInputs, int nOutputs, ActivationFunction af) : 
	_weights(nOutputs, nInputs), 
	_biases(nOutputs, 1), 
	_output(nOutputs, 1), 
	_af(af)
{
}

void NeuralNetwork::Layer::activate()
{
	switch (_af)
	{
		case ActivationFunction::SIGMOID:
			activation_sigmoid();
			break;
		
		case ActivationFunction::SOFTMAX:
			activation_softmax();
			break;
		
		default:
			assert(false);
			break;
	};
}

void NeuralNetwork::Layer::activation_sigmoid()
{
	nn::map(_output, [] (nn::Matrix::value_type v) { return 1.0f / (1.0f + std::expf(-v)); });
}

void NeuralNetwork::Layer::activation_softmax()
{
	nn::map(_output, [] (nn::Matrix::value_type v) { return std::expf(v); });
	nn::Matrix::value_type sum = nn::sum(_output, 0.0f);
	nn::map(_output, [&] (nn::Matrix::value_type v) { return v / sum; });
}

void NeuralNetwork::feed_forward(const nn::Matrix &input)
{
	const nn::Matrix *payload = &input;
	
	for (auto &layer : _layers)
	{
		nn::dot(layer._weights, *payload, layer._output);
		nn::add(layer._biases, layer._output, layer._output);
		layer.activate();
		payload = &layer._output;
	}
}

nn::Matrix::value_type NeuralNetwork::compute_loss(const nn::Matrix &target)
{
	switch (_lf)
	{
		case LossFunction::MEAN_SQUARE_ERROR:
			return compute_loss_mean_square_error(target);
		
		case LossFunction::SOFTMAX_CROSS_ENTROPY:
			return compute_loss_softmax_cross_entropy(target);
		
		default:
			assert(false);
	};
	
	return 0.0;
}

nn::Matrix::value_type NeuralNetwork::compute_loss_mean_square_error(const nn::Matrix &target)
{
	nn::Matrix::value_type v = (nn::Matrix::value_type)0.0;
	nn::map(_layers.back()._output, target, [&] (nn::Matrix::value_type a, nn::Matrix::value_type b) {
		v += (a - b) * (a - b);
	});
	v /= (target.numRows() * target.numColumns());
	return v;
}

nn::Matrix::value_type NeuralNetwork::compute_loss_softmax_cross_entropy(const nn::Matrix &target)
{
	nn::Matrix::value_type v = (nn::Matrix::value_type)0.0;
	nn::map(_layers.back()._output, target, [&] (nn::Matrix::value_type a, nn::Matrix::value_type b) {
		v += a * std::log(b);
	});
	return -v;
}

void NeuralNetwork::back_propagation(const nn::Matrix &input, const nn::Matrix &target)
{
	/* nn::Matrix guess(target.numRows(), target.numColumns());
	feed_forward(input, guess);
	
	nn::Matrix error(target.numRows(), target.numColumns());
	nn::subtract(target, guess, error);
	
	for (auto &layer : _layers)
	{
		nn::transpose(layer._weights, layer._tweights);
	} */
}

void NeuralNetwork::mutate(double rate)
{
	// printf("NeuralNetwork::mutate(%f): ", rate);
	int nmutations = 0, nvalues = 0;
	
	for (Layer &layer : _layers)
	{
		nn::map(layer._weights, [&] (nn::Matrix::value_type v) {
			float p = _zero_one_distribution(_random_generator);
			if (p <= rate)
			{
				// v += 0.5f * _minus_one_one_distribution(_random_generator);
				v = _minus_one_one_distribution(_random_generator);
				nmutations += 1;
			}
			return v;
		});
		nvalues += layer._weights.numRows() * layer._weights.numColumns();
		
		nn::map(layer._biases, [&] (nn::Matrix::value_type v) {
			float p = _zero_one_distribution(_random_generator);
			if (p <= rate)
			{
				// v += 0.5f * _minus_one_one_distribution(_random_generator);
				v = _minus_one_one_distribution(_random_generator);
				nmutations += 1;
			}
			return v;
		});
		nvalues += layer._biases.numRows() * layer._biases.numColumns();
	}
	
	// printf("%d / %d (%f)\n", nmutations, nvalues, (double)nmutations / (double)nvalues);
}

}; // namespace nn

