#ifndef __WVK_COMPUTE_PIPELINE_MANAGER_H__
#define __WVK_COMPUTE_PIPELINE_MANAGER_H__

#include "VkDevice.h"
#include <list>

namespace wvk
{

class ComputePipelineManager
{
public:
	ComputePipelineManager(wvk::Device *device);
	~ComputePipelineManager();
	
	struct Pipeline
	{
		VkPipeline _pipeline;
	};
	
	Pipeline *create(const VkComputePipelineCreateInfo &cInfo);
	void destroy(Pipeline *p);
	
	void destroyAll();
	
protected:
	void create(const VkComputePipelineCreateInfo &cInfo, Pipeline &p);
	
	wvk::Device *_device;
	
	using PipelineList = std::list<Pipeline *>;
	PipelineList _pipelines;
};

}; // namespace wvk

#endif // __WVK_COMPUTE_PIPELINE_MANAGER_H__
