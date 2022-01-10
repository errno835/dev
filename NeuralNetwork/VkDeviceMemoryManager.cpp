#include "VkDeviceMemoryManager.h"

#include <cassert>
#include <exception>
#include <stdexcept>

namespace wvk
{

DeviceMemoryManager::DeviceMemoryManager(wvk::Device *device)
{
	_device = device;
	_pageSize = 16 * 1024 * 1024;
}

DeviceMemoryManager::~DeviceMemoryManager()
{
	if (_device != nullptr)
	{
		releaseAll();
	}
}

uint32_t DeviceMemoryManager::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties)
{
	VkPhysicalDeviceMemoryProperties memProperties;
	vkGetPhysicalDeviceMemoryProperties(*_device, &memProperties);

	for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
	{
		if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
		{
			return i;
		}
	}

	throw std::runtime_error("wvk::DeviceMemoryManager - failed to find suitable memory type");
	return ~0;
}

namespace
{
	bool align(VkDeviceSize alignment, VkDeviceSize size, VkDeviceSize &offset, VkDeviceSize &storageSize)
	{
		VkDeviceSize _Off = offset & (alignment - 1);
		if (0 < _Off)
			_Off = alignment - _Off;
		
		if (storageSize < _Off || storageSize - _Off < size)
			return false;
		
		offset += _Off;
		storageSize -= _Off;
		return true;
	}
	
	void setMode(DeviceMemoryManager::MemoryChunk &chunk, DeviceMemoryManager::MemoryChunk::RangeMode mode)
	{
		if (chunk._mode == mode)
			return;
		
		if (mode == DeviceMemoryManager::MemoryChunk::RangeMode::SORTED_BY_OFFSET)
		{
			chunk._availableRanges.sort([] (const DeviceMemoryManager::MemoryChunk::Range &a, const DeviceMemoryManager::MemoryChunk::Range &b) {
				return a._offset < b._offset;
			});
		}
		else if (mode == DeviceMemoryManager::MemoryChunk::RangeMode::SORTED_BY_SIZE)
		{
			chunk._availableRanges.sort([] (const DeviceMemoryManager::MemoryChunk::Range &a, const DeviceMemoryManager::MemoryChunk::Range &b) {
				return a._size < b._size;
			});
		}
		
		chunk._mode = mode;
	}
	
	void compactAvailableRanges(DeviceMemoryManager::MemoryChunk &chunk)
	{
		setMode(chunk, DeviceMemoryManager::MemoryChunk::RangeMode::SORTED_BY_OFFSET);
		
		for (DeviceMemoryManager::MemoryChunk::RangeList::iterator it = chunk._availableRanges.begin(); it != chunk._availableRanges.end(); )
		{
			if (it->_size == 0)
			{
				it = chunk._availableRanges.erase(it);
				continue;
			}
			
			DeviceMemoryManager::MemoryChunk::RangeList::iterator nextit = it;
			++nextit;
			
			if (nextit != chunk._availableRanges.end() && it->_offset + it->_size >= nextit->_offset)
			{
				nextit->_offset = it->_offset;
				nextit->_size += it->_size;
				it = chunk._availableRanges.erase(it);
				continue;
			}
			
			++it;
		}
	}

	bool splitRange
	(
		const VkMemoryRequirements &req, const DeviceMemoryManager::MemoryChunk::Range &range, 
		DeviceMemoryManager::MemoryChunk::Range &p1, DeviceMemoryManager::MemoryChunk::Range &p2, DeviceMemoryManager::MemoryChunk::Range &p3
	)
	{
		VkDeviceSize p = range._offset;
		VkDeviceSize s = range._size;
		
		if (align(req.alignment, req.size, p, s))
		{
			// The chunk can be split in 3 parts:
			// range._offset                                                           range._offset + range._size - 1
			// [ <-------- 1 --------> +                       + <-------- 3 --------> ]
			// [ --------------------- + --------------------- + --------------------- ]
			// [                       + <-------- 2 --------> +                       ]
			//                         p                                               p + s - 1
			
			p1._offset = range._offset;
			p1._size = (VkDeviceSize)(size_t)p - range._offset;
			
			p2._offset = p1._offset + p1._size;
			p2._size = req.size;
			
			p3._offset = p2._offset + p2._size;
			p3._size = range._size - (p1._size + p2._size);
			
			return true;
		}
		
		return false;
	}
};

DeviceMemoryManager::DeviceMemory DeviceMemoryManager::allocate(VkMemoryPropertyFlags properties, const VkMemoryRequirements &req)
{
	uint32_t memoryTypeIndex = findMemoryType(req.memoryTypeBits, properties);
	
	for (MemoryChunk &chunk : _memoryChunks)
	{
		if (chunk._properties != properties)
			continue;
		if (chunk._memoryTypeIndex != memoryTypeIndex)
			continue;
		if (chunk._size < req.size)
			continue;
		
		setMode(chunk, MemoryChunk::RangeMode::SORTED_BY_SIZE);
		
		for (MemoryChunk::RangeList::iterator it = chunk._availableRanges.begin(); it != chunk._availableRanges.end(); ++it)
		{
			MemoryChunk::Range &range = *it;
			
			MemoryChunk::Range p1, p2, p3;
			if (splitRange(req, range, p1, p2, p3))
			{
				chunk._availableRanges.erase(it);
				
				if (p1._size > 0)
					chunk._availableRanges.push_back(p1);
				
				if (p3._size > 0)
					chunk._availableRanges.push_back(p3);
				
				compactAvailableRanges(chunk);
				
				DeviceMemory dm;
				dm._deviceMemory = chunk._deviceMemory;
				dm._offset = p2._offset;
				dm._size = req.size;
				return dm;
			}
		}
	}
	
	_memoryChunks.push_back(MemoryChunk());
	MemoryChunk &chunk = _memoryChunks.back();
	
	chunk._properties = properties;
	chunk._memoryTypeIndex = memoryTypeIndex;
	chunk._size = req.size;
	if (chunk._size < _pageSize)
		chunk._size = _pageSize;
	chunk._mode = MemoryChunk::RangeMode::SORTED_BY_OFFSET;
	
	VkMemoryAllocateInfo memAllocInfo = {};
	memAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	memAllocInfo.allocationSize = chunk._size;
	memAllocInfo.memoryTypeIndex = memoryTypeIndex;

	if (vkAllocateMemory(*_device, &memAllocInfo, nullptr, &chunk._deviceMemory) != VK_SUCCESS)
	{
		throw std::runtime_error("wvk::DeviceMemoryManager - failed to allocate device memory");
	}
	
	MemoryChunk::Range range;
	range._offset = 0;
	range._size = chunk._size;
	
	MemoryChunk::Range p1, p2, p3;
	splitRange(req, range, p1, p2, p3);
	
	if (p1._size > 0)
		chunk._availableRanges.push_back(p1);
	
	if (p3._size > 0)
		chunk._availableRanges.push_back(p3);
	
	DeviceMemory dm;
	dm._deviceMemory = chunk._deviceMemory;
	dm._offset = p2._offset;
	dm._size = req.size;
	return dm;
}

void DeviceMemoryManager::release(DeviceMemory &dm)
{
	for (MemoryChunk &chunk : _memoryChunks)
	{
		if (chunk._deviceMemory == dm._deviceMemory)
		{
			chunk._availableRanges.push_back(MemoryChunk::Range());
			MemoryChunk::Range &range = chunk._availableRanges.back();
			range._offset = dm._offset;
			range._size = dm._size;
			
			compactAvailableRanges(chunk);
			
			dm._deviceMemory = VK_NULL_HANDLE;
			dm._offset = 0;
			dm._size = 0;
			
			break;
		}
	}
}

void *DeviceMemoryManager::map(const DeviceMemory &dm)
{
	void *data;
	if (vkMapMemory(*_device, dm._deviceMemory, dm._offset, dm._size, 0, &data) != VK_SUCCESS)
	{
		throw std::runtime_error("wvk::DeviceMemoryManager - failed to map device memory");
	}
	return data;
}

void DeviceMemoryManager::unmap(const DeviceMemory &dm)
{
	vkUnmapMemory(*_device, dm._deviceMemory);
}

void DeviceMemoryManager::releaseUnusedPages()
{
	for (auto it = _memoryChunks.begin(); it != _memoryChunks.end(); )
	{
		if (it->_availableRanges.size() == 1)
		{
			MemoryChunk &chunk = *it;
			
			assert(chunk._availableRanges.front()._offset == 0 && chunk._availableRanges.front()._size == it->_size);
			
			vkFreeMemory(*_device, chunk._deviceMemory, nullptr);
			it = _memoryChunks.erase(it);
			continue;
		}
		
		++it;
	}
}

void DeviceMemoryManager::releaseAll()
{
	for (MemoryChunk &chunk : _memoryChunks)
	{
		vkFreeMemory(*_device, chunk._deviceMemory, nullptr);
	}
	
	_memoryChunks.clear();
}

}; // namespace wvk
