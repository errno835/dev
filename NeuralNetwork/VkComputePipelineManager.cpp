#include "VkComputePipelineManager.h"

#include <exception>
#include <stdexcept>

namespace wvk
{

ComputePipelineManager::ComputePipelineManager(wvk::Device *device)
{
	_device = device;
}

ComputePipelineManager::~ComputePipelineManager()
{
	if (_device != nullptr)
	{
		destroyAll();
	}
}

ComputePipelineManager::Pipeline *ComputePipelineManager::create(const VkComputePipelineCreateInfo &cInfo)
{
	_pipelines.push_back(new Pipeline);
	Pipeline *p = _pipelines.back();
	
	create(cInfo, *p);
	return p;
}

void ComputePipelineManager::destroy(Pipeline *p)
{
	vkDestroyPipeline(*_device, p->_pipeline, nullptr);
	
	auto it = std::find(_pipelines.begin(), _pipelines.end(), p);
	if (it != _pipelines.end())
	{
		_pipelines.erase(it);
	}
	
	delete p;
}

void ComputePipelineManager::destroyAll()
{
	while (! _pipelines.empty())
	{
		destroy(_pipelines.front());
	}
}

void ComputePipelineManager::create(const VkComputePipelineCreateInfo &cInfo, ComputePipelineManager::Pipeline &p)
{
	if (vkCreateComputePipelines(*_device, VK_NULL_HANDLE, 1, &cInfo, nullptr, &p._pipeline) != VK_SUCCESS)
	{
		throw std::runtime_error("wvk::ComputePipelineManager - failed to create pipeline");
	}
}

}; // namespace wvk
