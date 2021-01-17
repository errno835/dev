#ifndef __WVK_DEVICE_H__
#define __WVK_DEVICE_H__

#include <vulkan/vulkan.h>
#include <vector>
#include <map>
#include <string>

namespace wvk
{
class DeviceMemoryManager;
class BufferManager;
class Buffer;
class ImageManager;
class Image;
class ComputePipelineManager;

class Device
{
public:
	Device();
	~Device();
	
	void setValidationEnabled(bool enabled);
	
	void create();
	void destroy();
	
	DeviceMemoryManager *memoryManager() { return _memoryManager; }
	BufferManager *bufferManager() { return _bufferManager; }
	ImageManager *imageManager() { return _imageManager; }
	ComputePipelineManager *computePipelineManager() { return _computePipelineManager; }
	
	struct CommandBuffer
	{
		VkCommandBuffer _buffer;
		
		enum class State
		{
			UNDEFINED, 
			RECORDING, 
			RECORDED, 
			SUBMITTED
		};
		
		State _state;
		
		operator VkCommandBuffer () { return _buffer; }
	};
	
	CommandBuffer *getOrCreateComputeCommandBuffer(const std::string &key);
	void destroyComputeCommandBuffer(const std::string &key);
	void destroyAllCommandBuffers();
	
	void beginRecordCommands(CommandBuffer *cb, VkCommandBufferUsageFlags usage);
	void endRecordCommands(CommandBuffer *cb);
	void submitComputeCommands(CommandBuffer *cb, VkFence fence = VK_NULL_HANDLE);
	void waitComputeQueueIdle();
	
	CommandBuffer *beginSingleTimeCommands();
	void endSingleTimeCommands(CommandBuffer *cb);
	
	struct ShaderModule
	{
		VkShaderModule _module;
	};
	
	const ShaderModule &getOrCreateShaderModule(const std::string &spvFileName);
	void destroyAllShaderModules();
	
	void registerDescriptorSetLayout(VkDescriptorSetLayout h);
	void destroyAllDescriptorSetLayouts();
	
	void registerPipelineLayout(VkPipelineLayout h);
	void destroyAllPipelineLayouts();
	
	template <class... Args> void immediate(void (Device::*f)(CommandBuffer *cb, Args... args), Args... args)
	{
		CommandBuffer *cb = beginSingleTimeCommands();
		(this->*f)(cb, args...);
		endSingleTimeCommands(cb);
	}
	
	void copy(CommandBuffer *cb, Buffer *src, Buffer *dst);
	void copy(CommandBuffer *cb, Buffer *src, Buffer *dst, VkDeviceSize size);
	void copy(CommandBuffer *cb, Buffer *src, Buffer *dst, VkDeviceSize srcOffset, VkDeviceSize dstOffset, VkDeviceSize size);
	
	void copy(CommandBuffer *cb, Image *src, Image *dst);
	void copy(CommandBuffer *cb, Buffer *src, Image *dst);
	void copy(CommandBuffer *cb, Image *src, Buffer *dst);
	
	void commit(CommandBuffer *cb, Image *image, VkImageLayout newLayout, VkPipelineStageFlags srcStageMask, VkPipelineStageFlags dstStageMask);
	
	std::vector<const char *> _instanceExtensions;
	std::vector<const char *> _instanceLayers;
	std::vector<const char *> _deviceExtensions;
	
	operator VkPhysicalDevice() { return _physicalDevice; }
	operator VkDevice() { return _device; }
	
protected:
	void setImageLayout(
		VkCommandBuffer cmdbuffer,
		VkImage image,
		VkImageLayout oldImageLayout,
		VkImageLayout newImageLayout,
		VkImageSubresourceRange subresourceRange,
		VkPipelineStageFlags srcStageMask,
		VkPipelineStageFlags dstStageMask
	);
	
	VkInstance _instance = VK_NULL_HANDLE;
	VkPhysicalDevice _physicalDevice = VK_NULL_HANDLE;
	VkDevice _device = VK_NULL_HANDLE;
	int _computeQueueFamilyIndex = -1;
	VkQueue _computeQueue = VK_NULL_HANDLE;
	
	bool _validationEnabled = false;
	VkDebugUtilsMessengerEXT _debugMessenger = VK_NULL_HANDLE;
	
	VkCommandPool _computeCommandPool = VK_NULL_HANDLE;
	
	using ShaderModules = std::map<std::string, ShaderModule>;
	ShaderModules _shaderModules;
	
	struct DescriptorSetLayout
	{
		VkDescriptorSetLayout _handle;
	};
	
	using DescriptorSetLayouts = std::vector<DescriptorSetLayout>;
	DescriptorSetLayouts _registeredDescriptorSetLayouts;
	
	struct PipelineLayout
	{
		VkPipelineLayout _handle;
	};
	
	using PipelineLayouts = std::vector<PipelineLayout>;
	PipelineLayouts _registeredPipelineLayouts;
	
	using CommandBuffers = std::map<std::string, CommandBuffer>;
	CommandBuffers _computeCommandBuffers;
	
	wvk::DeviceMemoryManager *_memoryManager = nullptr;
	wvk::BufferManager *_bufferManager = nullptr;
	wvk::ImageManager *_imageManager = nullptr;
	wvk::ComputePipelineManager *_computePipelineManager = nullptr;
};

}; // namespace wvk

#endif // __WVK_DEVICE_H__
