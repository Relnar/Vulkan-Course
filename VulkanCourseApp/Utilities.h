#pragma once

#include <fstream>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <vector>

const int MAX_FRAME_DRAWS = 2;
const int MAX_OBJECTS = 10;

const std::vector<const char*> deviceExtensions
{
  VK_KHR_SWAPCHAIN_EXTENSION_NAME,
};

// Vertex representation
struct Vertex
{
  glm::vec3 pos;      // Vertex position (x, y, z)
  glm::vec3 col;      // Vertex color (r, g, b)
};

// Indices (locations of Queue Families (if they exist at all)
struct QueueFamilyIndices
{
  int graphicsFamily = -1;         // Location of Graphics Queue Family
  int presentationFamily = -1;     // Location of Presentation Queue Family

  // Check if queue families are valid
  bool isValid()
  {
    return graphicsFamily >= 0 && presentationFamily >= 0;
  }
};

struct SwapChainDetails
{
  VkSurfaceCapabilitiesKHR surfaceCapabilities;     // Surface properties, e.g. image size/extent
  std::vector<VkSurfaceFormatKHR> formats;          // Surface image formats, e.g. RGBA and sizeof of each color
  std::vector<VkPresentModeKHR> presentationModes;
};

struct SwapchainImage
{
  VkImage image;
  VkImageView imageView;
};

static std::vector<char> readFile(const std::string& filename)
{
  // std::ios::ate tells stream reading from end of file
  std::ifstream file(filename, std::ios::binary | std::ios::ate);
  if (!file.is_open())
  {
    throw std::runtime_error("Failed to open file");
  }

  size_t fileSize = (size_t)file.tellg();
  std::vector<char> fileBuffer(fileSize);

  // Reset position to the beginning of the file
  file.seekg(0);
  file.read(fileBuffer.data(), fileSize);

  file.close();

  return fileBuffer;
}

static uint32_t findMemoryTypeIndex(VkPhysicalDevice physicalDevice, uint32_t allowedTypes, VkMemoryPropertyFlags properties)
{
  VkPhysicalDeviceMemoryProperties memProperties;
  vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

  for (uint32_t i = 0; i < memProperties.memoryTypeCount; ++i)
  {
    if ((allowedTypes & (1 << i)) &&
        (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
    {
      return i;
    }
  }

  return uint32_t();
}

static void createBuffer(VkPhysicalDevice physicalDevice,
                         VkDevice device,
                         VkDeviceSize bufferSize,
                         VkBufferUsageFlags bufferUsage,
                         VkMemoryPropertyFlags bufferProperties,
                         VkBuffer* buffer,
                         VkDeviceMemory* bufferMemory,
                         VkAllocationCallbacks* a_pAllocCB)
{
  // Information to create a buffer (doesn't include assigning memory)
  VkBufferCreateInfo bufferInfo = {};
  bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferInfo.size = bufferSize;
  bufferInfo.usage = bufferUsage;
  bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;           // Similar to swap chain image, can share vertex buffers

  if (vkCreateBuffer(device, &bufferInfo, a_pAllocCB, buffer) != VK_SUCCESS)
  {
    throw std::runtime_error("Unable to create buffer");
  }

  // Get buffer memory requirements
  VkMemoryRequirements memReqs = {};
  vkGetBufferMemoryRequirements(device, *buffer, &memReqs);

  // Allocate memory to buffer
  VkMemoryAllocateInfo memAllocInfo = {};
  memAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  memAllocInfo.allocationSize = memReqs.size;
  memAllocInfo.memoryTypeIndex = findMemoryTypeIndex(physicalDevice,
                                                     memReqs.memoryTypeBits,
                                                     bufferProperties);

  // Allocate memory to VkDeviceMemory
  if (vkAllocateMemory(device, &memAllocInfo, a_pAllocCB, bufferMemory) != VK_SUCCESS)
  {
    throw std::runtime_error("Failed to allocate vertex buffer memory");
  }

  // Allocate memory to given vertex buffer
  if (vkBindBufferMemory(device, *buffer, *bufferMemory, 0) != VK_SUCCESS)
  {
    throw std::runtime_error("Failed to bind buffer memory vertex buffer");
  }
}

static VkCommandBuffer beginCommandBuffer(VkDevice device, VkCommandPool commandPool)
{
  VkCommandBuffer commandBuffer;
  VkCommandBufferAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandPool = commandPool;
  allocInfo.commandBufferCount = 1;
  vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

  VkCommandBufferBeginInfo bufferBeginInfo = {};
  bufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  bufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;      // Only using the buffer once

  // Begin recording transfer commands
  vkBeginCommandBuffer(commandBuffer, &bufferBeginInfo);

  return commandBuffer;
}

static void endCommandBuffer(VkDevice device, VkCommandBuffer commandBuffer, VkQueue commandQueue, VkCommandPool commandPool)
{
  vkEndCommandBuffer(commandBuffer);

  // Queue submission
  VkSubmitInfo submitInfo = {};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &commandBuffer;
  vkQueueSubmit(commandQueue, 1, &submitInfo, VK_NULL_HANDLE);

  // Wait for the queue to finish
  vkQueueWaitIdle(commandQueue);

  // Release temporary buffer
  vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
}

static void copyBuffer(VkDevice device,
                       VkQueue transferQueue,
                       VkCommandPool transferCmdPool,
                       VkBuffer srcBuffer,
                       VkBuffer dstBuffer,
                       VkDeviceSize bufferSize)
{
  auto transferCmdBuffer = beginCommandBuffer(device, transferCmdPool);

  // Copy the src buffer in the dst buffer
  VkBufferCopy region = {};
  region.srcOffset = 0;
  region.dstOffset = 0;
  region.size = bufferSize;
  vkCmdCopyBuffer(transferCmdBuffer, srcBuffer, dstBuffer, 1, &region);

  endCommandBuffer(device, transferCmdBuffer, transferQueue, transferCmdPool);
}

static void copyImageBuffer(VkDevice device,
                            VkQueue transferQueue,
                            VkCommandPool transferCmdPool,
                            VkBuffer srcBuffer,
                            VkImage dstImage,
                            uint32_t width,
                            uint32_t height)
{
  auto commandBuffer = beginCommandBuffer(device, transferCmdPool);

  VkBufferImageCopy region = {};
  region.bufferOffset = 0;                                            // Offset into buffer data
  region.bufferRowLength = 0;                                         // Row length of data to calculate data spacing
  region.bufferImageHeight = 0;                                       // Image height to calculate data spacing
  region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;     // Aspect of image to copy
  region.imageSubresource.mipLevel = 0;
  region.imageSubresource.baseArrayLayer = 0;
  region.imageSubresource.layerCount = 1;
  region.imageOffset = {0, 0, 0};
  region.imageExtent = { width, height, 1};

  // Copy buffer to given image
  vkCmdCopyBufferToImage(commandBuffer, srcBuffer, dstImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

  endCommandBuffer(device, commandBuffer, transferQueue, transferCmdPool);
}

static void transitionImageLayout(VkDevice device,
                                  VkQueue queue,
                                  VkCommandPool commandPool,
                                  VkImage image,
                                  VkImageLayout oldLayout,
                                  VkImageLayout newLayout)
{
  VkImageMemoryBarrier imgMemoryBarrier = {};

  VkPipelineStageFlags srcStage;
  VkPipelineStageFlags dstStage;

  // If transitioning from new image to image ready to receive data
  if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED &&
      newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
  {
    imgMemoryBarrier.srcAccessMask = 0;                              // Memory access stage transition must after...
    imgMemoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;   // Memory access stage transition must before...

    srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    dstStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
  }
  else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
  {
    imgMemoryBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;   // Memory access stage transition must after...
    imgMemoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;      // Memory access stage transition must before...

    srcStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    dstStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
  }
  else
  {
    assert(!"Unexpected old/new layouts");
    return;
  }

  VkCommandBuffer cmdBuffer = beginCommandBuffer(device, commandPool);

  imgMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  imgMemoryBarrier.oldLayout = oldLayout;
  imgMemoryBarrier.newLayout = newLayout;
  imgMemoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;    // Queue family to transiton from
  imgMemoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;    // Queue family to transiton to
  imgMemoryBarrier.image = image;
  imgMemoryBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  imgMemoryBarrier.subresourceRange.baseMipLevel = 0;
  imgMemoryBarrier.subresourceRange.levelCount = 1;
  imgMemoryBarrier.subresourceRange.baseArrayLayer = 0;
  imgMemoryBarrier.subresourceRange.layerCount = 1;

  vkCmdPipelineBarrier(cmdBuffer,
                       srcStage, dstStage,  // Pipeline stages (match to src and dst AccessMasks)
                       0,                   // Dependency flags
                       0, nullptr,          // Memory barrier
                       0, nullptr,          // Buffer memory barrier
                       1, &imgMemoryBarrier);  // Image memory barrier

  endCommandBuffer(device, cmdBuffer, queue, commandPool);
}