#ifndef __NN_POPULATION_H__
#define __NN_POPULATION_H__

#include "NeuralNetwork.h"
#include <initializer_list>

namespace nn
{

class Population
{
public:
	using Sample = NeuralNetwork::Sample;
	using LayerInfo = NeuralNetwork::LayerInfo;
	
	Population(int n, int nInputs, std::initializer_list<LayerInfo> layers, LossFunction lf)
	{
		_subjects.resize(n);
		
		for (Subject *&subject : _subjects)
		{
			subject = new Subject(nInputs, layers, lf);
		}
	}
	
	struct Subject
	{
		nn::NeuralNetwork _brain;
		double _score;
		
		Subject(int nInputs, std::initializer_list<LayerInfo> layers, LossFunction lf) : _brain(nInputs, layers, lf), _score(0.0)
		{
		}
	};
	
	using SubjectList = std::vector<Subject *>;
	const SubjectList &subjects() const { return _subjects; }
	
	void feed_forward(const std::vector<const Sample *> &samples);
	
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
