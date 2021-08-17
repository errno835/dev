#ifndef __NN_POPULATION_H__
#define __NN_POPULATION_H__

#include "NeuralNetwork.h"
#include <initializer_list>

namespace nn
{

class Population
{
public:
	Population(int n, int nInputs, std::initializer_list<NeuralNetwork::LayerInfo> layers)
	{
		_subjects.resize(n);
		
		for (int i = 0; i < n; ++i)
			_subjects[i] = new Subject(nInputs, layers);
	}
	
	using Sample = NeuralNetwork::Sample;
	
	struct Subject
	{
		nn::NeuralNetwork _brain;
		double _score;
		
		Subject(int nInputs, std::initializer_list<NeuralNetwork::LayerInfo> layers) : _brain(nInputs, layers), _score(0.0)
		{
		}
	};
	
	using SubjectList = std::vector<Subject *>;
	const SubjectList &subjects() const { return _subjects; }
	
	void feedforward(const std::vector<const Sample *> &samples);
	
	struct Statistics
	{
		double _score;
	};
	
	Statistics computePopulationStatistics() const;
	
	void nextgeneration();
	
protected:
	SubjectList _subjects;
};

}; // namespace nn

#endif // __NN_POPULATION_H__
