#include "VkBufferManager.h"

#include <algorithm>

namespace wvk
{

BufferManager::BufferManager(wvk::Device *device)
{
	_device = device;
	_devicemm = _device->memoryManager();
}

BufferManager::~BufferManager()
{
	if (_device != nullptr && _devicemm != nullptr)
	{
		destroyAll();
	}
}

Buffer *BufferManager::create(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties)
{
	_buffers.push_back(Buffer());
	Buffer &buffer = _buffers.back();
	
	VkBufferCreateInfo bufferCreateInfo = {};
	bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufferCreateInfo.size = size;
	bufferCreateInfo.usage = usage;
	bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	
	if (vkCreateBuffer(*_device, &bufferCreateInfo, nullptr, &buffer._handle) != VK_SUCCESS)
	{
		throw std::runtime_error("wvk::BufferManager - failed to create buffer");
	}
	
	VkMemoryRequirements req;
	vkGetBufferMemoryRequirements(*_device, buffer._handle, &req);
	
	buffer._dm = _devicemm->allocate(properties, req);
	buffer._usage = usage;
	buffer._properties = properties;
	
	vkBindBufferMemory(*_device, buffer._handle, buffer._dm._deviceMemory, buffer._dm._offset);
	
	return &buffer;
}

void BufferManager::destroy(Buffer *buffer)
{
	vkDestroyBuffer(*_device, buffer->_handle, nullptr);
	_devicemm->release(buffer->_dm);
	
	for (auto it = _buffers.begin(); it != _buffers.end(); ++it)
	{
		if (it->_handle == buffer->_handle)
		{
			_buffers.erase(it);
			break;
		}
	}
}

void BufferManager::destroyAll()
{
	while (! _buffers.empty())
	{
		destroy(&_buffers.front());
	}
}

}; // namespace wvk
