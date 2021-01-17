#ifndef __NN_POPULATION_H__
#define __NN_POPULATION_H__

#include "NeuralNetwork.h"

namespace nn
{

class Population
{
public:
	Population(int n, int nInputs, int nHidden, int nOutputs);
	
	using Sample = NeuralNetwork::Sample;
	
	struct Subject
	{
		nn::NeuralNetwork _brain;
		double _score;
		
		Subject(int nInputs, int nHidden, int nOutputs) : _brain(nInputs, nHidden, nOutputs), _score(0.0)
		{
		}
	};
	
	using SubjectList = std::vector<Subject *>;
	const SubjectList &subjects() const { return _subjects; }
	
	void feedforward(const std::vector<const Sample *> &samples);
	
	struct Statistics
	{
		double _min;
		double _max;
		double _avg;
	};
	
	Statistics computePopulationStatistics() const;
	
	void nextgeneration();
	
protected:
	SubjectList _subjects;
};

}; // namespace nn

#endif // __NN_POPULATION_H__
