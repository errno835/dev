#include "VkImageManager.h"

#include <algorithm>

namespace wvk
{

ImageManager::ImageManager(wvk::Device *device)
{
	_device = device;
	_devicemm = _device->memoryManager();
}

ImageManager::~ImageManager()
{
	if (_device != nullptr && _devicemm != nullptr)
	{
		destroyAll();
	}
}

Image *ImageManager::create1D(
	uint32_t width, 
	uint32_t mipLevels, 
	VkSampleCountFlagBits numSamples, 
	VkFormat format, 
	VkImageTiling tiling, 
	VkImageUsageFlags usage, 
	VkMemoryPropertyFlags properties)
{
	_images.push_back(Image());
	Image &image = _images.back();
	
	image._ci = {};
	image._ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	image._ci.imageType = VK_IMAGE_TYPE_1D;
	image._ci.extent.width = width;
	image._ci.extent.height = 1;
	image._ci.extent.depth = 1;
	image._ci.mipLevels = mipLevels;
	image._ci.arrayLayers = 1;
	image._ci.format = format;
	image._ci.tiling = tiling;
	image._ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	image._ci.usage = usage;
	image._ci.samples = numSamples;
	image._ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	
	try
	{
		create(image, properties);
		return &image;
	}
	catch (...)
	{
		_images.pop_back();
		throw;
	}
	
	return nullptr;
}

Image *ImageManager::create2D(
	uint32_t width, 
	uint32_t height, 
	uint32_t mipLevels, 
	VkSampleCountFlagBits numSamples, 
	VkFormat format, 
	VkImageTiling tiling, 
	VkImageUsageFlags usage, 
	VkMemoryPropertyFlags properties)
{
	_images.push_back(Image());
	Image &image = _images.back();
	
	image._ci = {};
	image._ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	image._ci.imageType = VK_IMAGE_TYPE_2D;
	image._ci.extent.width = width;
	image._ci.extent.height = height;
	image._ci.extent.depth = 1;
	image._ci.mipLevels = mipLevels;
	image._ci.arrayLayers = 1;
	image._ci.format = format;
	image._ci.tiling = tiling;
	image._ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	image._ci.usage = usage;
	image._ci.samples = numSamples;
	image._ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	
	try
	{
		create(image, properties);
		return &image;
	}
	catch (...)
	{
		_images.pop_back();
		throw;
	}
	
	return nullptr;
}

Image *ImageManager::create3D(
	uint32_t width, 
	uint32_t height, 
	uint32_t depth, 
	uint32_t mipLevels, 
	VkSampleCountFlagBits numSamples, 
	VkFormat format, 
	VkImageTiling tiling, 
	VkImageUsageFlags usage, 
	VkMemoryPropertyFlags properties)
{
	_images.push_back(Image());
	Image &image = _images.back();
	
	image._ci = {};
	image._ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	image._ci.imageType = VK_IMAGE_TYPE_3D;
	image._ci.extent.width = width;
	image._ci.extent.height = height;
	image._ci.extent.depth = depth;
	image._ci.mipLevels = mipLevels;
	image._ci.arrayLayers = 1;
	image._ci.format = format;
	image._ci.tiling = tiling;
	image._ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	image._ci.usage = usage;
	image._ci.samples = numSamples;
	image._ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	
	try
	{
		create(image, properties);
		return &image;
	}
	catch (...)
	{
		_images.pop_back();
		throw;
	}
	
	return nullptr;
}

void ImageManager::create(Image &image, VkMemoryPropertyFlags properties)
{
	if (vkCreateImage(*_device, &image._ci, nullptr, &image._handle) != VK_SUCCESS)
	{
		throw std::runtime_error("wvk::ImageManager - failed to create image");
	}
	
	VkMemoryRequirements req;
	vkGetImageMemoryRequirements(*_device, image._handle, &req);
	
	image._dm = _devicemm->allocate(properties, req);
	image._properties = properties;
	image._layout = VK_IMAGE_LAYOUT_UNDEFINED;
	
	vkBindImageMemory(*_device, image._handle, image._dm._deviceMemory, image._dm._offset);
}

void ImageManager::destroy(Image *image)
{
	vkDestroyImage(*_device, image->_handle, nullptr);
	_devicemm->release(image->_dm);
	
	for (auto it = _images.begin(); it != _images.end(); ++it)
	{
		if (it->_handle == image->_handle)
		{
			_images.erase(it);
			break;
		}
	}
}

void ImageManager::destroyAll()
{
	while (! _images.empty())
	{
		destroy(&_images.front());
	}
}

}; // namespace wvk
