CC = /c/LLVM/bin/clang
# VULKAN_SDK = /c/VulkanSDK/1.0.46.0
VULKAN_SDK = /c/VulkanSDK/1.2.135.0

# CXXFLAGS = -c -g -gcodeview -Wno-deprecated
# LINKSFLAGS = -g

CXXFLAGS = -c -Wno-deprecated -I$(VULKAN_SDK)/Include
CXX_DEBUG_FLAGS = $(CXXFLAGS) -g -gcodeview
CXX_RELEASE_FLAGS = $(CXXFLAGS) -O3

LINKFLAGS = -fuse-ld=lld -L$(VULKAN_SDK)/Lib -lvulkan-1
LINK_DEBUG_FLAGS = $(LINKFLAGS) -g
LINK_RELEASE_FLAGS = $(LINKFLAGS)

sources =	main.cpp Matrix.cpp NeuralNetwork.cpp Population.cpp \
				VkDevice.cpp \
				VkDeviceMemoryManager.cpp \
				VkBufferManager.cpp \
				VkImageManager.cpp \
				VkComputePipelineManager.cpp
shaders_src = shaders/feed_forward.comp

objects = $(addprefix obj/,$(sources:.cpp=.obj))
objectsd = $(addprefix objd/,$(sources:.cpp=.obj))
dependencies = $(addprefix dep/,$(sources:.cpp=.d))
dependenciesd = $(addprefix depd/,$(sources:.cpp=.d))
shaders_spirv = $(shaders_src:.comp=.spirv)

.PHONY: clean release debug all shaders

shaders/%.spirv: shaders/%.comp
	$(VULKAN_SDK)/Bin/glslc $< -o $@

obj/%.obj: %.cpp
	@mkdir -p obj dep
	$(CC) $(CXX_RELEASE_FLAGS) -MMD -MF $(addprefix dep/,$(<:.cpp=.d)) $< -o $@

objd/%.obj: %.cpp
	@mkdir -p objd depd
	$(CC) $(CXX_DEBUG_FLAGS) -MMD -MF $(addprefix depd/,$(<:.cpp=.d)) $< -o $@

shaders: $(shaders_spirv)

nn.exe: $(objects)
	$(CC) $(LINK_RELEASE_FLAGS) $^ -o $@

nnd.exe: $(objectsd)
	$(CC) $(LINK_DEBUG_FLAGS) $^ -o $@

clean:
	rm -fr obj dep objd depd nn.exe nnd.exe nnd.pdb shaders/*.spirv

release: nn.exe $(shaders_spirv)
debug: nnd.exe $(shaders_spirv)
all: release

-include $(dependencies)
-include $(dependenciesd)
