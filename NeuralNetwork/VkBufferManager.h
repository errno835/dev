#ifndef __WVK_BUFFER_MANAGER_H__
#define __WVK_BUFFER_MANAGER_H__

#include "VkDevice.h"
#include "VkDeviceMemoryManager.h"
#include <list>

namespace wvk
{

struct Buffer
{
	VkBuffer _handle;
	DeviceMemoryManager::DeviceMemory _dm;
	VkBufferUsageFlags _usage;
	VkMemoryPropertyFlags _properties;
	
	operator VkBuffer () { return _handle; }
};

class BufferManager
{
public:
	BufferManager(wvk::Device *device);
	~BufferManager();
	
	Buffer *create(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties);
	void destroy(Buffer *buffer);
	
	void destroyAll();
	
protected:
	wvk::Device *_device;
	wvk::DeviceMemoryManager *_devicemm;
	
	using BufferList = std::list<Buffer>;
	BufferList _buffers;
};

}; // namespace wvk

#endif // __WVK_BUFFER_MANAGER_H__
