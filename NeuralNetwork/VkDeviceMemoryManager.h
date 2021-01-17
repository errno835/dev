#ifndef __WVK_DEVICE_MEMORY_MANAGER_H__
#define __WVK_DEVICE_MEMORY_MANAGER_H__

#include "VkDevice.h"
#include <vulkan/vulkan.h>
#include <list>

namespace wvk
{

class DeviceMemoryManager
{
public:
	DeviceMemoryManager(wvk::Device *device);
	~DeviceMemoryManager();
	
	void setPageSize(VkDeviceSize size) { _pageSize = size; }
	
	struct MemoryChunk
	{
		VkDeviceMemory _deviceMemory;
		VkMemoryPropertyFlags _properties;
		VkDeviceSize _size;
		uint32_t _memoryTypeIndex;
		
		struct Range
		{
			VkDeviceSize _offset, _size;
		};
		
		using RangeList = std::list<Range>;
		RangeList _availableRanges;
		
		enum class RangeMode
		{
			SORTED_BY_OFFSET, 
			SORTED_BY_SIZE
		};
		
		RangeMode _mode;
	};
	
	struct DeviceMemory
	{
		VkDeviceMemory _deviceMemory;
		VkDeviceSize _offset;
		VkDeviceSize _size;
	};
	
	DeviceMemory allocate(VkMemoryPropertyFlags properties, const VkMemoryRequirements &req);
	void release(DeviceMemory &dm);
	
	void *map(const DeviceMemory &dm);
	void unmap(const DeviceMemory &dm);
	
	void releaseUnusedPages();
	void releaseAll();
	
protected:
	uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
	
	wvk::Device *_device;
	
	VkDeviceSize _pageSize;
	
	using MemoryChunkList = std::list<MemoryChunk>;
	MemoryChunkList _memoryChunks;
};

}; // namespace wvk

#endif // __WVK_DEVICE_MEMORY_MANAGER_H__