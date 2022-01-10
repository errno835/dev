#include "VkDevice.h"
#include "VkDeviceMemoryManager.h"
#include "VkBufferManager.h"
#include "VkImageManager.h"
#include "VkComputePipelineManager.h"

#include <set>
#include <algorithm>
#include <exception>
#include <stdexcept>

namespace wvk
{

Device::Device()
{
}

Device::~Device()
{
	if (_device != VK_NULL_HANDLE)
	{
		destroy();
	}
}

void Device::setValidationEnabled(bool enabled)
{
	_validationEnabled = enabled;
}

namespace
{
	int findQueueFamilyIndex(VkPhysicalDevice physicalDevice, VkFlags queueFlags)
	{
		uint32_t queueFamilyCount = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
		
		std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
		vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());
		
		int i = 0;
		for (const auto &queueFamily : queueFamilies)
		{
			if (queueFamily.queueCount > 0 && (queueFamily.queueFlags & queueFlags) == queueFlags)
			{
				return i;
			}
			
			++i;
		}
		
		return -1;
	}
	
	VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback
	(
		VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, 
		VkDebugUtilsMessageTypeFlagsEXT messageType, 
		const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData, 
		void *pUserData
	)
	{
		printf("==========================================================================\n");
		printf("Validation layer: %s\n\n", pCallbackData->pMessage);
		
		return VK_FALSE;
	}
	
	struct str_cmp
	{
		bool operator () (const char *a, const char *b) const
		{
			return strcmp(a, b) < 0;
		}
	};
	
	void removeDuplicatedEntries(std::vector<const char *> &tab)
	{
		std::vector<const char *> wtab;
		wtab.swap(tab);
		
		std::set<const char *, str_cmp> wset;
		
		for (const char *entry : wtab)
		{
			if (wset.insert(entry).second)
			{
				tab.push_back(entry);
			}
		}
	}
};

void Device::create()
{
	if (_validationEnabled)
	{
		_instanceLayers.push_back("VK_LAYER_KHRONOS_validation");
		_instanceExtensions.push_back("VK_EXT_debug_utils");
	}
	
	removeDuplicatedEntries(_instanceLayers);
	removeDuplicatedEntries(_instanceExtensions);
	
	uint32_t layerCount;
	vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
	
	std::vector<VkLayerProperties> availableLayers(layerCount);
	vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());
	
	for (const char *layerName : _instanceLayers)
	{
		bool found = false;
		
		for (VkLayerProperties &lp : availableLayers)
		{
			if (strcmp(layerName, lp.layerName) == 0)
			{
				found = true;
				break;
			}
		}
		
		if (! found)
		{
			char buffer[1024];
			sprintf(buffer, "wvk::Device - missing '%s' instance layer", layerName);
			throw std::runtime_error(buffer);
		}
	}
	
	/* printf("Available instance layers:\n");
	for (VkLayerProperties &lp : availableLayers)
	{
		printf("   - '%s'\n", lp.layerName);
	} */
	
	VkApplicationInfo appInfo = {};
	appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
	appInfo.pApplicationName = "Neural network";
	appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
	appInfo.pEngineName = "No Engine";
	appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
	appInfo.apiVersion = VK_API_VERSION_1_0;
	
	VkInstanceCreateInfo instanceCreateInfo = {};
	instanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	instanceCreateInfo.pApplicationInfo = &appInfo;
	instanceCreateInfo.enabledExtensionCount = (uint32_t)_instanceExtensions.size();
	instanceCreateInfo.ppEnabledExtensionNames = _instanceExtensions.empty() ? nullptr : _instanceExtensions.data();
	instanceCreateInfo.enabledLayerCount = (uint32_t)_instanceLayers.size();
	instanceCreateInfo.ppEnabledLayerNames = _instanceLayers.empty() ? nullptr : _instanceLayers.data();
	
	VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo = {};
	if (_validationEnabled)
	{
		debugCreateInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
		debugCreateInfo.messageSeverity = 
			VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | 
			VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | 
			VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
		debugCreateInfo.messageType = 
			VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | 
			VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | 
			VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
		debugCreateInfo.pfnUserCallback = debugCallback;
		
		instanceCreateInfo.pNext = &debugCreateInfo;
	}
	
	if (vkCreateInstance(&instanceCreateInfo, nullptr, &_instance) != VK_SUCCESS)
	{
		throw std::runtime_error("wvk::Device - failed to create instance");
	}
	
	if (_validationEnabled)
	{
		PFN_vkCreateDebugUtilsMessengerEXT func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
			_instance, 
			"vkCreateDebugUtilsMessengerEXT");
		if (func == nullptr || func(_instance, &debugCreateInfo, nullptr, &_debugMessenger) != VK_SUCCESS)
		{
			throw std::runtime_error("wvk::Device - failed to set up debug messenger");
		}
	}
	
	uint32_t physicalDeviceCount = 0;
	vkEnumeratePhysicalDevices(_instance, &physicalDeviceCount, nullptr);
	
	if (physicalDeviceCount == 0)
	{
		throw std::runtime_error("wvk::Device - failed to find GPUs with Vulkan support");
	}
	
	std::vector<VkPhysicalDevice> physicaldevices(physicalDeviceCount);
	vkEnumeratePhysicalDevices(_instance, &physicalDeviceCount, physicaldevices.data());
	
	for (auto physicalDevice : physicaldevices)
	{
		VkPhysicalDeviceProperties deviceProperties;
		vkGetPhysicalDeviceProperties(physicalDevice, &deviceProperties);
		
		if (deviceProperties.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
			continue;
		
		int computeQueueFamilyIndex = findQueueFamilyIndex(physicalDevice, VK_QUEUE_COMPUTE_BIT);
		
		if (computeQueueFamilyIndex < 0)
			continue;
		
		_physicalDevice = physicalDevice;
		_computeQueueFamilyIndex = computeQueueFamilyIndex;
		break;
	}
	
	if (_physicalDevice == VK_NULL_HANDLE || _computeQueueFamilyIndex < 0)
	{
		throw std::runtime_error("wvk::Device - failed to find discrete GPU with compute support");
	}
	
	std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
	std::set<int> uniqueQueueFamilies = { _computeQueueFamilyIndex };

	float queuePriority = 1.0f;
	for (int queueFamily : uniqueQueueFamilies)
	{
		VkDeviceQueueCreateInfo queueCreateInfo = {};
		queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		queueCreateInfo.queueFamilyIndex = queueFamily;
		queueCreateInfo.queueCount = 1;
		queueCreateInfo.pQueuePriorities = &queuePriority;
		queueCreateInfos.push_back(queueCreateInfo);
	}
	
	VkDeviceCreateInfo deviceCreateInfo = {};
	deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
	deviceCreateInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
	deviceCreateInfo.pQueueCreateInfos = queueCreateInfos.data();
	
	VkPhysicalDeviceFeatures deviceFeatures = {};
	deviceCreateInfo.pEnabledFeatures = &deviceFeatures;
	
	deviceCreateInfo.enabledLayerCount = static_cast<uint32_t>(_instanceLayers.size());
	deviceCreateInfo.ppEnabledLayerNames = _instanceLayers.empty() ? nullptr : _instanceLayers.data();
	deviceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(_deviceExtensions.size());
	deviceCreateInfo.ppEnabledExtensionNames = _deviceExtensions.empty() ? nullptr : _deviceExtensions.data();
	
	if (vkCreateDevice(_physicalDevice, &deviceCreateInfo, nullptr, &_device) != VK_SUCCESS)
	{
		throw std::runtime_error("wvk::Device - failed to create logical device");
	}
	
	vkGetDeviceQueue(_device, _computeQueueFamilyIndex, 0, &_computeQueue);
	
	VkCommandPoolCreateInfo commandPoolCreateInfo = {};
	commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	commandPoolCreateInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
	commandPoolCreateInfo.queueFamilyIndex = _computeQueueFamilyIndex;
	
	if (vkCreateCommandPool(_device, &commandPoolCreateInfo, nullptr, &_computeCommandPool) != VK_SUCCESS)
	{
		throw std::runtime_error("wvk::Device - failed to create compute command pool");
	}
	
	_memoryManager = new DeviceMemoryManager(this);
	_bufferManager = new BufferManager(this);
	_imageManager = new ImageManager(this);
	_computePipelineManager = new ComputePipelineManager(this);
}

void Device::destroy()
{
	if (_device != VK_NULL_HANDLE)
	{
		delete _computePipelineManager;
		_computePipelineManager = nullptr;
		
		delete _imageManager;
		_imageManager = nullptr;
		
		delete _bufferManager;
		_bufferManager = nullptr;
		
		delete _memoryManager;
		_memoryManager = nullptr;
		
		destroyAllCommandBuffers();
		destroyAllPipelineLayouts();
		destroyAllDescriptorSetLayouts();
		destroyAllShaderModules();
		
		vkDestroyCommandPool(_device, _computeCommandPool, nullptr);
		_computeCommandPool = VK_NULL_HANDLE;
		
		vkDestroyDevice(_device, nullptr);
		_device = VK_NULL_HANDLE;
		_physicalDevice = VK_NULL_HANDLE;
		_computeQueueFamilyIndex = -1;
		_computeQueue = VK_NULL_HANDLE;
	}
	
	if (_instance != VK_NULL_HANDLE)
	{
		if (_validationEnabled)
		{
			PFN_vkDestroyDebugUtilsMessengerEXT func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
				_instance, 
				"vkDestroyDebugUtilsMessengerEXT");
			if (func != nullptr)
			{
				func(_instance, _debugMessenger, nullptr);
				_debugMessenger = VK_NULL_HANDLE;
			}
		}
		
		vkDestroyInstance(_instance, nullptr);
		_instance = VK_NULL_HANDLE;
	}
}

const std::string _beginSingleTimeCommandsKey = "single time commands";

Device::CommandBuffer *Device::beginSingleTimeCommands()
{
	CommandBuffer *cb = getOrCreateComputeCommandBuffer(_beginSingleTimeCommandsKey);
	beginRecordCommands(cb, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
	return cb;
}

void Device::endSingleTimeCommands(Device::CommandBuffer *cb)
{
	endRecordCommands(cb);
	submitComputeCommands(cb);
	waitComputeQueueIdle();
	
	destroyComputeCommandBuffer(_beginSingleTimeCommandsKey);
}

const Device::ShaderModule &Device::getOrCreateShaderModule(const std::string &spvFileName)
{
	auto it = _shaderModules.find(spvFileName);
	if (it != _shaderModules.end())
		return it->second;
	
	FILE *fd = fopen(spvFileName.c_str(), "rb");
	if (fd == nullptr)
	{
		throw std::runtime_error("wvk::Device - failed to read SPIR-V file");
	}
	
	fseek(fd, 0, SEEK_END);
	size_t len = ftell(fd);
	fseek(fd, 0, SEEK_SET);
	
	char *buffer = new char[len];
	fread(buffer, 1, len, fd);
	
	fclose(fd);
	
	VkShaderModuleCreateInfo createInfo = {};
	createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	createInfo.flags = 0;
	createInfo.codeSize = len;
	createInfo.pCode = reinterpret_cast<const uint32_t *>(buffer);

	VkShaderModule shaderModule;
	if (vkCreateShaderModule(_device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS)
	{
		throw std::runtime_error("wvk::Device - failed to create shader module");
	}
	
	delete [] buffer;
	
	ShaderModule &sm = _shaderModules[spvFileName];
	sm._module = shaderModule;
	
	return sm;
}

void Device::destroyAllShaderModules()
{
	for (auto &item : _shaderModules)
	{
		vkDestroyShaderModule(_device, item.second._module, nullptr);
	}
	
	_shaderModules.clear();
}

void Device::registerDescriptorSetLayout(VkDescriptorSetLayout h)
{
	_registeredDescriptorSetLayouts.push_back(DescriptorSetLayout());
	DescriptorSetLayout &l = _registeredDescriptorSetLayouts.back();
	l._handle = h;
}

void Device::destroyAllDescriptorSetLayouts()
{
	for (DescriptorSetLayout &l : _registeredDescriptorSetLayouts)
	{
		vkDestroyDescriptorSetLayout(_device, l._handle, nullptr);
	}
	
	_registeredDescriptorSetLayouts.clear();
}

void Device::registerPipelineLayout(VkPipelineLayout h)
{
	_registeredPipelineLayouts.push_back(PipelineLayout());
	PipelineLayout &l = _registeredPipelineLayouts.back();
	l._handle = h;
}

void Device::destroyAllPipelineLayouts()
{
	for (PipelineLayout &l : _registeredPipelineLayouts)
	{
		vkDestroyPipelineLayout(_device, l._handle, nullptr);
	}
	
	_registeredPipelineLayouts.clear();
}

Device::CommandBuffer *Device::getOrCreateComputeCommandBuffer(const std::string &key)
{
	auto it = _computeCommandBuffers.find(key);
	if (it != _computeCommandBuffers.end())
		return &it->second;
	
	VkCommandBufferAllocateInfo allocInfo = {};
	allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	allocInfo.commandPool = _computeCommandPool;
	allocInfo.commandBufferCount = 1;
	
	VkCommandBuffer commandBuffer;
	if (vkAllocateCommandBuffers(_device, &allocInfo, &commandBuffer) != VK_SUCCESS)
	{
		throw std::runtime_error("wvk::Device - failed to allocate command buffer");
	}
	
	CommandBuffer &cb = _computeCommandBuffers[key];
	cb._buffer = commandBuffer;
	cb._state = CommandBuffer::State::UNDEFINED;
	
	return &cb;
}

void Device::destroyComputeCommandBuffer(const std::string &key)
{
	auto it = _computeCommandBuffers.find(key);
	if (it != _computeCommandBuffers.end())
	{
		vkFreeCommandBuffers(_device, _computeCommandPool, 1, &it->second._buffer);
		_computeCommandBuffers.erase(it);
	}
}

void Device::destroyAllCommandBuffers()
{
	while (! _computeCommandBuffers.empty())
	{
		destroyComputeCommandBuffer(_computeCommandBuffers.begin()->first);
	}
}

void Device::beginRecordCommands(CommandBuffer *cb, VkCommandBufferUsageFlags usage)
{
	VkCommandBufferBeginInfo beginInfo = {};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	beginInfo.flags = usage;
	
	vkBeginCommandBuffer(cb->_buffer, &beginInfo);
	cb->_state = CommandBuffer::State::RECORDING;
}

void Device::endRecordCommands(CommandBuffer *cb)
{
	vkEndCommandBuffer(cb->_buffer);
	cb->_state = CommandBuffer::State::RECORDED;
}

void Device::submitComputeCommands(CommandBuffer *cb, VkFence fence)
{
	VkSubmitInfo submitInfo = {};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &cb->_buffer;
	
	vkQueueSubmit(_computeQueue, 1, &submitInfo, fence);
	cb->_state = CommandBuffer::State::SUBMITTED;
}

void Device::waitComputeQueueIdle()
{
	vkQueueWaitIdle(_computeQueue);
}

void Device::copy(CommandBuffer *cb, Buffer *src, Buffer *dst)
{
	copy(cb, src, dst, 0, 0, std::min(src->_dm._size, dst->_dm._size));
}

void Device::copy(CommandBuffer *cb, Buffer *src, Buffer *dst, VkDeviceSize size)
{
	copy(cb, src, dst, (VkDeviceSize)0, (VkDeviceSize)0, size);
}

void Device::copy(CommandBuffer *cb, Buffer *src, Buffer *dst, VkDeviceSize srcOffset, VkDeviceSize dstOffset, VkDeviceSize size)
{
	if (srcOffset + size > src->_dm._size || dstOffset + size > dst->_dm._size)
	{
		throw std::runtime_error("wvk::BufferManager - failed to copy buffers - buffer overflow");
	}
	
	VkBufferCopy copyRegion = {};
	copyRegion.srcOffset = srcOffset;
	copyRegion.dstOffset = dstOffset;
	copyRegion.size = size;
	
	vkCmdCopyBuffer(cb->_buffer, *src, *dst, 1, &copyRegion);
}

void Device::copy(CommandBuffer *cb, Image *src, Image *dst)
{
	VkImageCopy copyRegion = {};
	copyRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	copyRegion.srcSubresource.mipLevel = 0;
	copyRegion.srcSubresource.baseArrayLayer = 0;
	copyRegion.srcSubresource.layerCount = 1;
	copyRegion.srcOffset.x = 0;
	copyRegion.srcOffset.y = 0;
	copyRegion.srcOffset.z = 0;
	copyRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	copyRegion.dstSubresource.mipLevel = 0;
	copyRegion.dstSubresource.baseArrayLayer = 0;
	copyRegion.dstSubresource.layerCount = 1;
	copyRegion.dstOffset.x = 0;
	copyRegion.dstOffset.y = 0;
	copyRegion.dstOffset.z = 0;
	copyRegion.extent.width = std::min(src->_ci.extent.width, dst->_ci.extent.width);
	copyRegion.extent.height = std::min(src->_ci.extent.height, dst->_ci.extent.height);
	copyRegion.extent.depth = std::min(src->_ci.extent.depth, dst->_ci.extent.depth);
	
	vkCmdCopyImage(cb->_buffer, *src, src->_layout, *dst, dst->_layout, 1, &copyRegion);
}

void Device::copy(CommandBuffer *cb, Buffer *src, Image *dst)
{
	VkBufferImageCopy copyRegion = {};
	copyRegion.bufferOffset = 0;
	copyRegion.bufferRowLength = 0;
	copyRegion.bufferImageHeight = 0;
	copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	copyRegion.imageSubresource.mipLevel = 0;
	copyRegion.imageSubresource.baseArrayLayer = 0;
	copyRegion.imageSubresource.layerCount = 1;
	copyRegion.imageOffset.x = 0;
	copyRegion.imageOffset.y = 0;
	copyRegion.imageOffset.z = 0;
	copyRegion.imageExtent.width = dst->_ci.extent.width;
	copyRegion.imageExtent.height = dst->_ci.extent.height;
	copyRegion.imageExtent.depth = dst->_ci.extent.depth;
	
	vkCmdCopyBufferToImage(cb->_buffer, *src, *dst, dst->_layout, 1, &copyRegion);
}

void Device::copy(CommandBuffer *cb, Image *src, Buffer *dst)
{
	VkBufferImageCopy copyRegion = {};
	copyRegion.bufferOffset = 0;
	copyRegion.bufferRowLength = 0;
	copyRegion.bufferImageHeight = 0;
	copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	copyRegion.imageSubresource.mipLevel = 0;
	copyRegion.imageSubresource.baseArrayLayer = 0;
	copyRegion.imageSubresource.layerCount = 1;
	copyRegion.imageOffset.x = 0;
	copyRegion.imageOffset.y = 0;
	copyRegion.imageOffset.z = 0;
	copyRegion.imageExtent.width = src->_ci.extent.width;
	copyRegion.imageExtent.height = src->_ci.extent.height;
	copyRegion.imageExtent.depth = src->_ci.extent.depth;
	
	vkCmdCopyImageToBuffer(cb->_buffer, *src, src->_layout, *dst, 1, &copyRegion);
}

void Device::commit(CommandBuffer *cb, Image *image, VkImageLayout newLayout, VkPipelineStageFlags srcStageMask, VkPipelineStageFlags dstStageMask)
{
	VkImageMemoryBarrier barrier = {};
	barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	
	// Whole image
	barrier.subresourceRange = {};
	barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	barrier.subresourceRange.baseMipLevel = 0;
	barrier.subresourceRange.levelCount = 1;
	barrier.subresourceRange.baseArrayLayer = 0;
	barrier.subresourceRange.layerCount = 1;
	
	barrier.oldLayout = image->_layout;
	barrier.newLayout = newLayout;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.image = *image;
	
	barrier.srcAccessMask = 0;
	barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
	
	vkCmdPipelineBarrier(
		*cb, 				// command buffer
		srcStageMask, 	// source stage mask
		dstStageMask, 	// destination stage mask
		0, 				// dependency flags
		0, nullptr, 	// memory barriers
		0, nullptr, 	// buffer memory barriers
		1, &barrier 	// image memory barriers
	);
	
	image->_layout = newLayout;
}

}; // namespace wvk
