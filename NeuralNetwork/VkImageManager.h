#ifndef __WVK_IMAGE_MANAGER_H__
#define __WVK_IMAGE_MANAGER_H__

#include "VkDevice.h"
#include "VkDeviceMemoryManager.h"
#include <list>

namespace wvk
{

struct Image
{
	VkImage _handle;
	DeviceMemoryManager::DeviceMemory _dm;
	VkImageCreateInfo _ci;
	VkMemoryPropertyFlags _properties;
	VkImageLayout _layout;
	
	operator VkImage () { return _handle; }
};

class ImageManager
{
public:
	ImageManager(wvk::Device *device);
	~ImageManager();
	
	Image *create1D(
		uint32_t width, 
		uint32_t mipLevels, 
		VkSampleCountFlagBits numSamples, 
		VkFormat format, 
		VkImageTiling tiling, 
		VkImageUsageFlags usage, 
		VkMemoryPropertyFlags properties);
	
	Image *create2D(
		uint32_t width, 
		uint32_t height, 
		uint32_t mipLevels, 
		VkSampleCountFlagBits numSamples, 
		VkFormat format, 
		VkImageTiling tiling, 
		VkImageUsageFlags usage, 
		VkMemoryPropertyFlags properties);
	
	Image *create3D(
		uint32_t width, 
		uint32_t height, 
		uint32_t depth, 
		uint32_t mipLevels, 
		VkSampleCountFlagBits numSamples, 
		VkFormat format, 
		VkImageTiling tiling, 
		VkImageUsageFlags usage, 
		VkMemoryPropertyFlags properties);
	
	void destroy(Image *image);
	
	void destroyAll();
	
protected:
	void create(Image &image, VkMemoryPropertyFlags properties);
	
	wvk::Device *_device;
	wvk::DeviceMemoryManager *_devicemm;
	
	using ImageList = std::list<Image>;
	ImageList _images;
};

}; // namespace wvk

#endif // __WVK_IMAGE_MANAGER_H__
