#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "Utilities.h"

#include <vector>

class Mesh
{
public:
  Mesh(VkPhysicalDevice newPhysicalDevice, VkDevice newDevice, const std::vector<Vertex>& vertices, VkAllocationCallbacks* a_pAllocCB = nullptr);
  ~Mesh();

  void destroyVertexBuffer();

  int getVertexCount() { return vertexCount; }
  VkBuffer getVertexBuffer() { return vertexBuffer; }

private:
  int vertexCount;
  VkBuffer vertexBuffer;
  VkDeviceMemory vertexBufferMemory;

  VkPhysicalDevice physicalDevice;
  VkDevice device;
  VkAllocationCallbacks* m_pAllocCB;

  void createVertexbuffer(const std::vector<Vertex>& vertices);
  uint32_t findMemoryTypeIndex(uint32_t allowedTypes, VkMemoryPropertyFlags properties);
};

