#include "Population.h"
#include <atomic>
#include <thread>
#include <chrono>
#include <string>

namespace nn
{

Population::Population(int n, int nInputs, int nHidden, int nOutputs)
{
	_subjects.resize(n);
	
	for (int i = 0; i < n; ++i)
		_subjects[i] = new Subject(nInputs, nHidden, nOutputs);
}

namespace
{
	std::string durationstring(const std::chrono::duration<double> &d)
	{
		int h = 0, m = 0;
		double s = d.count();
		
		if (s >= 60.0)
		{
			m = s / 60.0;
			s -= m * 60.0;
		}
		
		if (m >= 60)
		{
			h = m / 60;
			m -= h * 60;
		}
		
		char temp[128];
		char *p = temp;
		
		if (h > 0)
			p += sprintf(p, "%dh ", h);
		
		if (m > 0 || h > 0)
			p += sprintf(p, "%dm ", m);
		
		sprintf(p, "%.2fs", s);
		
		return temp;
	}
	
	struct TaskRunner
	{
		struct Task
		{
			const Population::Sample *_sample;
			Population::Subject *_subject;
		};
		
		TaskRunner(std::atomic<int> &itask, std::vector<Task> &tasks, int nRows, int nColumns) : _itask(itask), _tasks(tasks)
		{
			_output.resize(nRows, nColumns);
			_error.resize(nRows, nColumns);
		}
		
		void run()
		{
			while (true)
			{
				int nexttask = _itask.fetch_add(1);
				if (nexttask >= (int)_tasks.size())
					break;
				
				Task &task = _tasks[nexttask];
				const Population::Sample *sample = task._sample;
				Population::Subject *subject = task._subject;
				
				subject->_brain.feedforward(sample->_input, _output);
				nn::subtract(sample->_target, _output, _error);
				
				double sum = 0.0f;
				nn::map(_error, [&] (nn::Matrix::value_type v) { sum += (double)v * (double)v; return v; });
				
				subject->_score = sum;
			}
		}
		
		std::atomic<int> &_itask;
		std::vector<Task> &_tasks;
		nn::Matrix _output;
		nn::Matrix _error;
		
		std::thread _thread;
	};
};

void Population::feedforward(const std::vector<const Sample *> &samples)
{
	std::vector<TaskRunner::Task> tasks;
	tasks.reserve(samples.size() * _subjects.size());
	
	for (const Sample *sample : samples)
	{
		for (Subject *subject : _subjects)
		{
			tasks.push_back(TaskRunner::Task());
			tasks.back()._sample = sample;
			tasks.back()._subject = subject;
		}
	}
	
	int n = std::thread::hardware_concurrency();
	
	printf("Num threads: %d - Num samples: %d - Num subjects: %d\n", n, (int)samples.size(), (int)_subjects.size());
	fflush(stdout);
	
	std::vector<TaskRunner> taskRunners;
	taskRunners.reserve(n);
	
	std::atomic<int> itask(0);
	
	auto t0 = std::chrono::high_resolution_clock::now();
	
	for (int i = 0; i < n; ++i)
	{
		taskRunners.push_back(TaskRunner(itask, tasks, samples.front()->_target.numRows(), samples.front()->_target.numColumns()));
		taskRunners[i]._thread = std::thread(&TaskRunner::run, std::ref(taskRunners[i]));
	}
	
	for (int i = 0; i < n; ++i)
	{
		taskRunners[i]._thread.join();
	}
	
	auto t1 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed_seconds = t1 - t0;
	std::string s = durationstring(elapsed_seconds);
	
	printf("Time: %s\n", s.c_str());
}

Population::Statistics Population::computePopulationStatistics() const
{
	Population::Statistics s;
	s._min = 1e20;
	s._max = -1e20;
	s._avg = 0.0;
	
	for (const Subject *subject : _subjects)
	{
		if (subject->_score < s._min)
			s._min = subject->_score;
		if (subject->_score > s._max)
			s._max = subject->_score;
		s._avg += subject->_score;
	}
	
	if (! _subjects.empty())
	{
		s._avg /= (double)_subjects.size();
	}
	
	return s;
}

void Population::nextgeneration()
{
}
	
}; // namespace nn
