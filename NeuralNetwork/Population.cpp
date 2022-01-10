#include "Population.h"
#include <atomic>
#include <thread>
#include <string>
#include <random>
#include <algorithm>

namespace nn
{

namespace
{
	struct TaskRunner
	{
		struct Task
		{
			const Population::Sample *_sample;
			Population::Subject *_subject;
		};
		
		TaskRunner()
		{
			_itask = nullptr;
			_tasks = nullptr;
		}
		
		void set(std::atomic<int> *itask, std::vector<Task> *tasks, int nRows, int nColumns)
		{
			_itask = itask;
			_tasks = tasks;
			_error.resize(nRows, nColumns);
		}
		
		void run()
		{
			while (true)
			{
				int nexttask = _itask->fetch_add(1);
				if (nexttask >= (int)_tasks->size())
					break;
				
				Task &task = (*_tasks)[nexttask];
				const Population::Sample *sample = task._sample;
				Population::Subject *subject = task._subject;
				
				subject->_brain.feed_forward(sample->_input);
				subject->_score += subject->_brain.compute_loss(sample->_target);
			}
		}
		
		std::atomic<int> *_itask;
		std::vector<Task> *_tasks;
		nn::Matrix _error;
		
		std::thread _thread;
	};
	
	std::vector<TaskRunner::Task> _tasks;
	std::vector<TaskRunner> _task_runners;
};

void Population::feed_forward(const std::vector<const Sample *> &samples)
{
	_tasks.resize(samples.size() * _subjects.size());
	
	std::default_random_engine _random_generator;
	std::uniform_int_distribution<int> _distribution(0, samples.size() - 1);
	
	for (int isubject = 0, k = 0; isubject < _subjects.size(); ++isubject)
	{
		for (int isample = 0; isample < samples.size(); ++isample, ++k)
		{
			_tasks[k]._sample = samples[_distribution(_random_generator)];
			_tasks[k]._subject = _subjects[isubject];
		}
	}
	
	for (Subject *subject : _subjects)
	{
		subject->_score = 0.0;
	}
	
	int n = std::thread::hardware_concurrency();
	_task_runners.reserve(n);
	
	std::atomic<int> itask(0);
	
	for (int i = 0; i < n; ++i)
	{
		_task_runners[i].set(&itask, &_tasks, samples.front()->_target.numRows(), samples.front()->_target.numColumns());
		_task_runners[i]._thread = std::thread(&TaskRunner::run, std::ref(_task_runners[i]));
	}
	
	for (int i = 0; i < n; ++i)
	{
		_task_runners[i]._thread.join();
	}
	
	for (Subject *subject : _subjects)
	{
		subject->_score /= (double)samples.size();
	}
}

Population::Statistics Population::computePopulationStatistics() const
{
	double sum = 0.0;
	for (const Subject *subject : _subjects)
	{
		sum += subject->_score;
	}
	
	if (! _subjects.empty())
	{
		sum /= (double)_subjects.size();
	}
	
	Population::Statistics s;
	s._score = sum;
	return s;
}

void Population::nextgeneration()
{
	// std::sort(_subjects.begin(), _subjects.end(), [] (Subject *a, Subject *b) { return a->_score > b->_score; });
	
	double min_mutation_rate = 0.1;
	double max_mutation_rate = 0.5;
	double avg_mutation_rate = 0.0;
	for (Subject *subject : _subjects)
	{
		double t = (subject->_score - 0.0) / (1.0 - 0.0);
		double mutation_rate = min_mutation_rate + (max_mutation_rate - min_mutation_rate) * (1.0 - t);
		avg_mutation_rate += mutation_rate;
		subject->_brain.mutate(mutation_rate);
	}
	
	if (! _subjects.empty())
	{
		avg_mutation_rate /= (double)_subjects.size();
	}
	
	printf("mutation rate: %5.2f", avg_mutation_rate);
}
	
}; // namespace nn
