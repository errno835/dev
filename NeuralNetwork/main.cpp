#include "NeuralNetwork.h"
#include "Population.h"
#include "VkDevice.h"
#include "VkDeviceMemoryManager.h"
#include "VkBufferManager.h"
#include "VkImageManager.h"
#include "VkComputePipelineManager.h"

#include <cstdio>
#include <random>
#include <exception>
#include <set>
#include <algorithm>
#include <array>
#include <chrono>
#include <cassert>

#define NOMINMAX
#include <Windows.h>

uint32_t bigToLittleEndian(uint32_t v)
{
	uint32_t vv;
	((unsigned char *)&vv)[0] = ((unsigned char *)&v)[3];
	((unsigned char *)&vv)[1] = ((unsigned char *)&v)[2];
	((unsigned char *)&vv)[2] = ((unsigned char *)&v)[1];
	((unsigned char *)&vv)[3] = ((unsigned char *)&v)[0];
	return vv;
}

bool readMNIST(const char *s, std::vector<nn::Population::Sample> &samples)
{
	char imagesFileName[1024], labelsFileName[1024];
	sprintf(imagesFileName, "%s-images.idx3-ubyte", s);
	sprintf(labelsFileName, "%s-labels.idx1-ubyte", s);
	
	FILE *imagesfd = fopen(imagesFileName, "rb");
	if (imagesfd == nullptr)
	{
		printf("Error: unable to open '%s'\n", imagesFileName);
		return false;
	}
	
	uint32_t temp;
	
	fread(&temp, 1, sizeof(uint32_t), imagesfd);
	uint32_t imagescc = bigToLittleEndian(temp);
	
	if (imagescc != 0x00000803)
	{
		printf("Error: invalid images 4CC: 0x%x (expecting 0x%x)\n", imagescc, 0x00000803);
		fclose(imagesfd);
		return false;
	}
	
	fread(&temp, 1, sizeof(uint32_t), imagesfd);
	uint32_t imagescount = bigToLittleEndian(temp);
	
	fread(&temp, 1, sizeof(uint32_t), imagesfd);
	uint32_t imagesheight = bigToLittleEndian(temp);
	
	fread(&temp, 1, sizeof(uint32_t), imagesfd);
	uint32_t imageswidth = bigToLittleEndian(temp);
	
	FILE *labelsfd = fopen(labelsFileName, "rb");
	if (labelsfd == nullptr)
	{
		printf("Error: unable to open '%s'\n", labelsFileName);
		fclose(imagesfd);
		return false;
	}
	
	fread(&temp, 1, sizeof(uint32_t), labelsfd);
	uint32_t labelscc = bigToLittleEndian(temp);
	
	if (labelscc != 0x00000801)
	{
		printf("Error: invalid labels 4CC: 0x%x (expecting 0x%x)\n", labelscc, 0x00000801);
		fclose(imagesfd);
		fclose(labelsfd);
		return false;
	}
	
	fread(&temp, 1, sizeof(uint32_t), labelsfd);
	uint32_t labelscount = bigToLittleEndian(temp);
	
	if (imagescount != labelscount)
	{
		printf("Error: images and labels count mismatch (%d, %d)\n", imagescount, labelscount);
		fclose(imagesfd);
		fclose(labelsfd);
		return false;
	}
	
	printf("Reading %d images and labels from '%s'\n", imagescount, s);
	fflush(stdout);
	
	uint8_t *pixelbuffer = new uint8_t[imagescount * imageswidth * imagesheight];
	fread(pixelbuffer, imagescount * imageswidth * imagesheight, sizeof(uint8_t), imagesfd);
	uint8_t *pixel = pixelbuffer;
	
	uint8_t *labelbuffer = new uint8_t[imagescount];
	fread(labelbuffer, imagescount, sizeof(uint8_t), labelsfd);
	uint8_t *label = labelbuffer;
	
	samples.reserve(samples.size() + imagescount);
	
	std::set<uint8_t> labelset;
	for (uint32_t i = 0; i < imagescount; ++i)
		labelset.insert(labelbuffer[i]);
	
	for (uint32_t i = 0; i < imagescount; ++i)
	{
		samples.push_back(nn::Population::Sample());
		nn::Population::Sample &sample = samples.back();
		
		sample._input.resize(imagesheight * imageswidth, 1);
		nn::map(sample._input, [&] (float v) { return *pixel++ / 255.0f; });
		
		sample._target.resize(labelset.size(), 1);
		sample._target(*label++, 0) = 1.0;
	}
	
	delete [] pixelbuffer;
	delete [] labelbuffer;
	
	fclose(imagesfd);
	fclose(labelsfd);
	
	return true;
}

std::string durationstring(const std::chrono::duration<double> &d)
{
	int h = 0, m = 0;
	double s = d.count();
	
	if (s >= 60.0)
	{
		m = s / 60.0;
		s -= m * 60.0;
	}
	
	if (m >= 60)
	{
		h = m / 60;
		m -= h * 60;
	}
	
	char temp[128];
	char *p = temp;
	
	if (h > 0)
		p += sprintf(p, "%dh ", h);
	
	if (m > 0 || h > 0)
		p += sprintf(p, "%dm ", m);
	
	sprintf(p, "%.2fs", s);
	
	return temp;
}

int nSamples = 2;
int nSubjects = 1;
int nInputs = 28*28;
int nHidden = 28*28;
int nOutputs = 10;

void parse_arguments(int argc, char *argv[])
{
	int iarg = 1;
	while (iarg < argc)
	{
		if (strcmp(argv[iarg], "--nSamples") == 0)
		{
			if (iarg + 1 < argc)
			{
				nSamples = atoi(argv[iarg + 1]);
				++iarg;
			}
			++iarg;
		}
		else if (strcmp(argv[iarg], "--nSubjects") == 0)
		{
			if (iarg + 1 < argc)
			{
				nSubjects = atoi(argv[iarg + 1]);
				++iarg;
			}
			++iarg;
		}
		else if (strcmp(argv[iarg], "--nHidden") == 0)
		{
			if (iarg + 1 < argc)
			{
				nHidden = atoi(argv[iarg + 1]);
				++iarg;
			}
			++iarg;
		}
		else
		{
			++iarg;
		}
	}
}

void copyPopulationData(float *data, const nn::Population &population)
{
	float *p = data;
	
	for (const nn::Population::Subject *subject : population.subjects())
	{
		const nn::NeuralNetwork &brain = subject->_brain;
		for (const nn::NeuralNetwork::Layer &layer : brain.layers())
		{
			memcpy(p, layer._weights.ptr(), sizeof(float) * layer._weights.numRows() * layer._weights.numColumns());
			p += layer._weights.numRows() * layer._weights.numColumns();
			
			memcpy(p, layer._biases.ptr(), sizeof(float) * layer._biases.numRows() * layer._biases.numColumns());
			p += layer._biases.numRows() * layer._biases.numColumns();
		}
	}
}

void copySampleData(float *data, const std::vector<const nn::Population::Sample *> &samples)
{
	float *p = data;
	
	for (const nn::Population::Sample *sample : samples)
	{
		memcpy(p, sample->_input.ptr(), sizeof(float) * sample->_input.numRows() * sample->_input.numColumns());
		p += sample->_input.numRows() * sample->_input.numColumns();
	}
}

int main(int argc, char *argv[])
{
	try
	{
		parse_arguments(argc, argv);
		
		std::vector<nn::Population::Sample> trainingsamples, testsamples;
		readMNIST("MNIST/train", trainingsamples);
		readMNIST("MNIST/t10k", testsamples);
		
		// std::vector<const nn::Population::Sample *> samples(trainingsamples.size());
		std::vector<const nn::Population::Sample *> samples(std::min((size_t)nSamples, trainingsamples.size()));
		for (size_t i = 0; i < samples.size(); ++i)
			samples[i] = &trainingsamples[i];
		
		nn::Population population(
			nSubjects, 
			nInputs, {
				{ nHidden, nn::ActivationFunction::SIGMOID }, 
				{ nOutputs, nn::ActivationFunction::SOFTMAX }
			}, 
			nn::LossFunction::SOFTMAX_CROSS_ENTROPY
		);
		
		for (int i = 0; i < 10; ++i)
		{
			printf("Generation %3d - ", i);
			fflush(stdout);
			
			auto t0 = std::chrono::high_resolution_clock::now();
			population.feed_forward(samples);
			auto t1 = std::chrono::high_resolution_clock::now();
			
			std::chrono::duration<double> elapsed_seconds = t1 - t0;
			std::string d = durationstring(elapsed_seconds);
			
			nn::Population::Statistics s = population.computePopulationStatistics();
			
			printf("duration: %s, score: %5.1f%%, ", d.c_str(), 100.0 * s._score);
			population.nextgeneration();
			printf("\n");
		}
		
		return 0;
		
// #define VK_BACKEND
#ifdef VK_BACKEND
		uint32_t inputToHiddenWeightsSize = nInputs * nHidden;
		uint32_t inputToHiddenBiasesSize = nHidden;
		uint32_t hiddenToOutputWeightsSize = nHidden * nOutputs;
		uint32_t hiddenToOutputBiasesSize = nOutputs;
		uint32_t subjectSize = 
			inputToHiddenWeightsSize + inputToHiddenBiasesSize + 
			hiddenToOutputWeightsSize + hiddenToOutputBiasesSize;
		uint32_t populationSize = nSubjects * subjectSize;
		
		uint32_t singleSampleSize = nInputs;
		uint32_t sampleSize = nSamples * singleSampleSize;
		
		printf("Input to hidden weights size:  %s\n", nn::HumanReadableSize(inputToHiddenWeightsSize).str());
		printf("Input to hidden biases size:   %s\n", nn::HumanReadableSize(inputToHiddenBiasesSize).str());
		printf("Hidden to output weights size: %s\n", nn::HumanReadableSize(hiddenToOutputWeightsSize).str());
		printf("Hidden to output biases size:  %s\n", nn::HumanReadableSize(hiddenToOutputBiasesSize).str());
		printf("Subject size:                  %s\n", nn::HumanReadableSize(subjectSize).str());
		printf("Population size:               %s\n", nn::HumanReadableSize(populationSize).str());
		printf("Single sample size:            %s\n", nn::HumanReadableSize(singleSampleSize).str());
		printf("Total sample size:             %s\n", nn::HumanReadableSize(sampleSize).str());
		
		wvk::Device device;
		device.setValidationEnabled(true);
		device.create();
		
		wvk::DeviceMemoryManager *mmanager = device.memoryManager();
		wvk::BufferManager *bmanager = device.bufferManager();
		wvk::ImageManager *imanager = device.imageManager();
		
		wvk::Buffer *stagingbuffer = bmanager->create(
			sizeof(float) * std::max(populationSize, sampleSize), 
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, 
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
		);
		
		// #define DATA_BUFFER_BACKED
		#define DATA_IMAGE_BACKED
		
		#ifdef DATA_BUFFER_BACKED
			wvk::Buffer *populationBuffer = bmanager->create(
				sizeof(float) * populationSize, 
				VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
			);
			
			wvk::Buffer *sampleBuffer = bmanager->create(
				sizeof(float) * sampleSize, 
				VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
			);
			
			float *populationData = (float *)mmanager->map(stagingbuffer->_dm);
			copyPopulationData(populationData, population);
			mmanager->unmap(stagingbuffer->_dm);
			device.immediate(&wvk::Device::copy, stagingbuffer, populationBuffer, (VkDeviceSize)(sizeof(float) * populationSize));
			
			float *sampleData = (float *)mmanager->map(stagingbuffer->_dm);
			copySampleData(sampleData, samples);
			mmanager->unmap(stagingbuffer->_dm);
			device.immediate(&wvk::Device::copy, stagingbuffer, sampleBuffer, (VkDeviceSize)(sizeof(float) * sampleSize));
		#endif
		
		#ifdef DATA_IMAGE_BACKED
			wvk::Image *populationImage = imanager->create1D(
				populationSize, 
				1, 
				VK_SAMPLE_COUNT_1_BIT, 
				VK_FORMAT_R32_SFLOAT, 
				VK_IMAGE_TILING_OPTIMAL, 
				VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, 
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
			);
			
			wvk::Image *sampleImage = imanager->create1D(
				sampleSize, 
				1, 
				VK_SAMPLE_COUNT_1_BIT, 
				VK_FORMAT_R32_SFLOAT, 
				VK_IMAGE_TILING_OPTIMAL, 
				VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, 
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
			);
			
			VkFence bufferToImageFence = VK_NULL_HANDLE;
			{
				VkFenceCreateInfo bufferToImageFenceCI = {};
				bufferToImageFenceCI.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
				bufferToImageFenceCI.flags  = 0;
				
				if (vkCreateFence(device, &bufferToImageFenceCI, nullptr, &bufferToImageFence) != VK_SUCCESS)
				{
					throw std::runtime_error("failed to create buffer-to-image fence");
				}
			}
			
			float *populationData = (float *)mmanager->map(stagingbuffer->_dm);
			copyPopulationData(populationData, population);
			mmanager->unmap(stagingbuffer->_dm);
			{
				wvk::Device::CommandBuffer *cb = device.getOrCreateComputeCommandBuffer("buffer-to-image");
				device.beginRecordCommands(cb, 0);
					// device.setImageLayout(cb, populationImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
					device.copy(cb, stagingbuffer, populationImage);
					// device.setImageLayout(cb, populationImage, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
				device.endRecordCommands(cb);
				device.submitComputeCommands(cb, bufferToImageFence);
			}
			
			if (vkWaitForFences(device, 1, &bufferToImageFence, VK_TRUE, ~0) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to wait for buffer-to-image fence");
			}
			vkResetFences(device, 1, &bufferToImageFence);
			
			float *sampleData = (float *)mmanager->map(stagingbuffer->_dm);
			copySampleData(sampleData, samples);
			mmanager->unmap(stagingbuffer->_dm);
			{
				wvk::Device::CommandBuffer *cb = device.getOrCreateComputeCommandBuffer("buffer-to-image");
				device.beginRecordCommands(cb, 0);
					// device.setImageLayout(cb, sampleImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
					device.copy(cb, stagingbuffer, sampleImage);
					// device.setImageLayout(cb, sampleImage, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
				device.endRecordCommands(cb);
				device.submitComputeCommands(cb, bufferToImageFence);
			}
			
			if (vkWaitForFences(device, 1, &bufferToImageFence, VK_TRUE, ~0) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to wait for buffer-to-image fence");
			}
			vkResetFences(device, 1, &bufferToImageFence);
		#endif
		
		// ---------------------------------------------------------------------------------------------------------------
		std::array<VkDescriptorSetLayoutBinding, 2> bindings = {};
		
		#ifdef DATA_BUFFER_BACKED
			bind ings[0].binding = 0;
			bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			bindings[0].descriptorCount = 1;
			bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
			bindings[0].pImmutableSamplers = nullptr;
			bindings[1].binding = 1;
			bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			bindings[1].descriptorCount = 1;
			bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
			bindings[1].pImmutableSamplers = nullptr;
		#endif
		
		#ifdef DATA_IMAGE_BACKED
			bindings[0].binding = 0;
			bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			bindings[0].descriptorCount = 1;
			bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
			bindings[0].pImmutableSamplers = nullptr;
			bindings[1].binding = 1;
			bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			bindings[1].descriptorCount = 1;
			bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
			bindings[1].pImmutableSamplers = nullptr;
		#endif
		
		VkDescriptorSetLayoutCreateInfo dslcInfo = {};
		dslcInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		dslcInfo.flags = 0;
		dslcInfo.bindingCount = (uint32_t)bindings.size();
		dslcInfo.pBindings = bindings.data();
		
		VkDescriptorSetLayout descriptorSetLayout;
		if (vkCreateDescriptorSetLayout(device, &dslcInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create descriptor set layout");
		}
		
		device.registerDescriptorSetLayout(descriptorSetLayout);
		
		// ---------------------------------------------------------------------------------------------------------------
		VkPipelineLayoutCreateInfo plcInfo = {};
		plcInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		plcInfo.flags = 0;
		plcInfo.setLayoutCount = 1;
		plcInfo.pSetLayouts = &descriptorSetLayout;
		plcInfo.pushConstantRangeCount = 0;
		plcInfo.pPushConstantRanges = nullptr;
		
		VkPipelineLayout pipelineLayout;
		if (vkCreatePipelineLayout(device, &plcInfo, nullptr, &pipelineLayout) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create pipeline layout");
		}
		
		device.registerPipelineLayout(pipelineLayout);
		
		// ---------------------------------------------------------------------------------------------------------------
		std::array<VkDescriptorPoolSize, 2> descriptorPoolSizes = {};
		
		#ifdef DATA_BUFFER_BACKED
			descriptorPoolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			descriptorPoolSizes[0].descriptorCount = 1;
			descriptorPoolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			descriptorPoolSizes[1].descriptorCount = 1;
		#endif
		
		#ifdef DATA_IMAGE_BACKED
			descriptorPoolSizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			descriptorPoolSizes[0].descriptorCount = 1;
			descriptorPoolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			descriptorPoolSizes[1].descriptorCount = 1;
		#endif
		
		VkDescriptorPoolCreateInfo dpcInfo = {};
		dpcInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		dpcInfo.maxSets = 1;
		dpcInfo.poolSizeCount = static_cast<uint32_t>(descriptorPoolSizes.size());
		dpcInfo.pPoolSizes = descriptorPoolSizes.data();
		
		VkDescriptorPool descriptorPool;
		if (vkCreateDescriptorPool(device, &dpcInfo, nullptr, &descriptorPool) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create descriptor pool");
		}
		
		// ---------------------------------------------------------------------------------------------------------------
		// SpecializationData sdata;
		// sdata._numInputs = nInputs;
		// sdata._numHidden = nHidden;
		// sdata._numOutputs = nOutputs;
		
		// VkSpecializationMapEntry specializationEntry = {};
		// specializationEntry.constantID = 0;
		// specializationEntry.offset = offsetof(SpecializationData, _wireframe);
		// specializationEntry.size = sizeof(sdata._wireframe);
		
		// VkSpecializationInfo specializationInfo = {};
		// specializationInfo.mapEntryCount = 1;
		// specializationInfo.pMapEntries = &specializationEntry;
		// specializationInfo.dataSize = sizeof(SpecializationData);
		// specializationInfo.pData = &sdata;
		
		// ---------------------------------------------------------------------------------------------------------------
		VkComputePipelineCreateInfo cpcInfo = {};
		cpcInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
		cpcInfo.flags = 0;
		cpcInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		cpcInfo.stage.flags = 0;
		cpcInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		cpcInfo.stage.module = device.getOrCreateShaderModule("shaders/feed_forward.spirv")._module;
		cpcInfo.stage.pName = "main";
		cpcInfo.stage.pSpecializationInfo = nullptr;
		cpcInfo.layout = pipelineLayout;
		cpcInfo.basePipelineHandle = VK_NULL_HANDLE;
		cpcInfo.basePipelineIndex = -1;
		
		wvk::ComputePipelineManager *cpManager = device.computePipelineManager();
		wvk::ComputePipelineManager::Pipeline *pipeline = cpManager->create(cpcInfo);
		
		// ---------------------------------------------------------------------------------------------------------------
		VkDescriptorSetAllocateInfo dsacInfo = {};
		dsacInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		dsacInfo.descriptorPool = descriptorPool;
		dsacInfo.descriptorSetCount = 1;
		dsacInfo.pSetLayouts = &descriptorSetLayout;
		
		VkDescriptorSet descriptorSet;
		if (vkAllocateDescriptorSets(device, &dsacInfo, &descriptorSet) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to allocate descriptor sets");
		}
		
		// ---------------------------------------------------------------------------------------------------------------
		#ifdef DATA_BUFFER_BACKED
			VkDescriptorBufferInfo populationBufferInfo = {};
			populationBufferInfo.buffer = populationBuffer->_handle;
			populationBufferInfo.offset = 0;
			populationBufferInfo.range = VK_WHOLE_SIZE;
			
			VkDescriptorBufferInfo sampleBufferInfo = {};
			sampleBufferInfo.buffer = sampleBuffer->_handle;
			sampleBufferInfo.offset = 0;
			sampleBufferInfo.range = VK_WHOLE_SIZE;
			
			std::array<VkWriteDescriptorSet, 2> descriptorWrites = {};
			
			descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[0].dstSet = descriptorSet;
			descriptorWrites[0].dstBinding = 0;
			descriptorWrites[0].dstArrayElement = 0;
			descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			descriptorWrites[0].descriptorCount = 1;
			descriptorWrites[0].pImageInfo = nullptr;
			descriptorWrites[0].pBufferInfo = &populationBufferInfo;
			descriptorWrites[0].pTexelBufferView = nullptr;
			
			descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[1].dstSet = descriptorSet;
			descriptorWrites[1].dstBinding = 1;
			descriptorWrites[1].dstArrayElement = 0;
			descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			descriptorWrites[1].descriptorCount = 1;
			descriptorWrites[1].pImageInfo = nullptr;
			descriptorWrites[1].pBufferInfo = &sampleBufferInfo;
			descriptorWrites[1].pTexelBufferView = nullptr;
		#endif
		
		#ifdef DATA_IMAGE_BACKED
			VkSampler nearestSampler = VK_NULL_HANDLE;
			{
				VkSamplerCreateInfo nearestSamplerCI = {};
				nearestSamplerCI.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
				nearestSamplerCI.minFilter = VK_FILTER_NEAREST;
				nearestSamplerCI.magFilter = VK_FILTER_NEAREST;
				nearestSamplerCI.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
				nearestSamplerCI.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
				nearestSamplerCI.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
				nearestSamplerCI.anisotropyEnable = VK_FALSE;
				nearestSamplerCI.maxAnisotropy = 0;
				nearestSamplerCI.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
				nearestSamplerCI.unnormalizedCoordinates = VK_FALSE;
				nearestSamplerCI.compareEnable = VK_FALSE;
				nearestSamplerCI.compareOp = VK_COMPARE_OP_ALWAYS;
				nearestSamplerCI.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
				nearestSamplerCI.minLod = 0.0f;
				nearestSamplerCI.maxLod = 0.0f;
				nearestSamplerCI.mipLodBias = 0.0f;
				
				if (vkCreateSampler(device, &nearestSamplerCI, nullptr, &nearestSampler) != VK_SUCCESS)
				{
					throw std::runtime_error("failed to create texture sampler!");
				}
			}
			
			VkImageView populationImageView = VK_NULL_HANDLE;
			{
				VkImageViewCreateInfo populationImageViewCI = {};
				populationImageViewCI.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
				populationImageViewCI.image = *populationImage;
				populationImageViewCI.viewType = VK_IMAGE_VIEW_TYPE_1D;
				populationImageViewCI.format = populationImage->_ci.format;
				populationImageViewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
				populationImageViewCI.subresourceRange.baseMipLevel = 0;
				populationImageViewCI.subresourceRange.levelCount = 1;
				populationImageViewCI.subresourceRange.baseArrayLayer = 0;
				populationImageViewCI.subresourceRange.layerCount = 1;
				
				if (vkCreateImageView(device, &populationImageViewCI, nullptr, &populationImageView) != VK_SUCCESS)
				{
					throw std::runtime_error("failed to create texture image view!");
				}
			}
			
			VkImageView sampleImageView = VK_NULL_HANDLE;
			{
				VkImageViewCreateInfo sampleImageViewCI = {};
				sampleImageViewCI.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
				sampleImageViewCI.image = *sampleImage;
				sampleImageViewCI.viewType = VK_IMAGE_VIEW_TYPE_1D;
				sampleImageViewCI.format = sampleImage->_ci.format;
				sampleImageViewCI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
				sampleImageViewCI.subresourceRange.baseMipLevel = 0;
				sampleImageViewCI.subresourceRange.levelCount = 1;
				sampleImageViewCI.subresourceRange.baseArrayLayer = 0;
				sampleImageViewCI.subresourceRange.layerCount = 1;
				
				if (vkCreateImageView(device, &sampleImageViewCI, nullptr, &sampleImageView) != VK_SUCCESS)
				{
					throw std::runtime_error("failed to create texture image view!");
				}
			}
			
			VkDescriptorImageInfo populationImageInfo = {};
			populationImageInfo.sampler = nearestSampler;
			populationImageInfo.imageView = populationImageView;
			populationImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			
			VkDescriptorImageInfo sampleImageInfo = {};
			sampleImageInfo.sampler = nearestSampler;
			sampleImageInfo.imageView = sampleImageView;
			sampleImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			
			std::array<VkWriteDescriptorSet, 2> descriptorWrites = {};
			
			descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[0].dstSet = descriptorSet;
			descriptorWrites[0].dstBinding = 0;
			descriptorWrites[0].dstArrayElement = 0;
			descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			descriptorWrites[0].descriptorCount = 1;
			descriptorWrites[0].pImageInfo = &populationImageInfo;
			descriptorWrites[0].pBufferInfo = nullptr;
			descriptorWrites[0].pTexelBufferView = nullptr;
			
			descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[1].dstSet = descriptorSet;
			descriptorWrites[1].dstBinding = 1;
			descriptorWrites[1].dstArrayElement = 0;
			descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			descriptorWrites[1].descriptorCount = 1;
			descriptorWrites[1].pImageInfo = &sampleImageInfo;
			descriptorWrites[1].pBufferInfo = nullptr;
			descriptorWrites[1].pTexelBufferView = nullptr;
		#endif
		
		vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
		
		// ---------------------------------------------------------------------------------------------------------------
		wvk::Device::CommandBuffer *ff = device.getOrCreateComputeCommandBuffer("feed forward");
		device.beginRecordCommands(ff, 0);
			vkCmdBindPipeline(ff->_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->_pipeline);
			
			vkCmdBindDescriptorSets(
				ff->_buffer, 							// commandBuffer
				VK_PIPELINE_BIND_POINT_COMPUTE, 	// pipelineBindPoint
				pipelineLayout, 						// layout
				0, 										// firstSet
				1, 										// descriptorSetCount
				&descriptorSet, 						// pDescriptorSets
				0, 										// dynamicOffsetCount
				nullptr 									// pDynamicOffsets
			);
			
			vkCmdDispatch(ff->_buffer, 32, 1, 1);
		device.endRecordCommands(ff);
		
		device.submitComputeCommands(ff);
		device.waitComputeQueueIdle();
		device.destroyComputeCommandBuffer("feed forward");
		
		// ---------------------------------------------------------------------------------------------------------------
		
		/* #ifdef DATA_BUFFER_BACKED
			device.immediate(&wvk::Device::copy, outbuffer, stagingbuffer);
		#endif
		
		{
			float *data = (float *)mmanager->map(stagingbuffer->_dm);
			
			int i = 0;
			for (; i < 4 * 1024 * 1024; ++i)
			{
				if (data[i] != (float)i)
				{
					break;
				}
			}
			printf("OK up to %d\n", i);
			
			mmanager->unmap(stagingbuffer->_dm);
		} */
				
		// ---------------------------------------------------------------------------------------------------------------
		vkDestroyDescriptorPool(device, descriptorPool, nullptr);
		
		cpManager->destroy(pipeline);
		
		#ifdef DATA_BUFFER_BACKED
			bmanager->destroy(populationBuffer);
			bmanager->destroy(sampleBuffer);
		#endif
		
		#ifdef DATA_IMAGE_BACKED
			vkDestroyFence(device, bufferToImageFence, nullptr);
			vkDestroySampler(device, nearestSampler, nullptr);
			vkDestroyImageView(device, populationImageView, nullptr);
			vkDestroyImageView(device, sampleImageView, nullptr);
			imanager->destroy(populationImage);
			imanager->destroy(sampleImage);
		#endif
		
		bmanager->destroy(stagingbuffer);
		
		device.destroy();
#endif // VK_BACKEND
	}
	catch (const std::exception &ex)
	{
		printf("Error: %s\n", ex.what());
	}
	
	return 0;
}
