#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "Utilities.h"

#include <vector>

struct Model
{
  glm::mat4 model;
};

class Mesh
{
public:
  Mesh(VkPhysicalDevice newPhysicalDevice,
       VkDevice newDevice,
       VkQueue transferQueue,
       VkCommandPool transferCmdPool,
       const std::vector<Vertex>& vertices,
       const std::vector<uint32_t>& indices,
       int newTexId,
       VkAllocationCallbacks* a_pAllocCB = nullptr);
  ~Mesh();

  void setModel(const glm::mat4& newModel) { model.model = newModel; }
  const Model& getModel() const { return model; }

  int getTexId() const { return texId; }

  void destroyBuffers();

  int getVertexCount() const { return vertexCount; }
  VkBuffer getVertexBuffer() const { return vertexBuffer; }

  int getIndexCount() const { return indexCount; }
  VkBuffer getIndexBuffer() const { return indexBuffer; }

private:
  Model model;

  int texId;

  int vertexCount;
  VkBuffer vertexBuffer;
  VkDeviceMemory vertexBufferMemory;

  int indexCount;
  VkBuffer indexBuffer;
  VkDeviceMemory indexBufferMemory;

  VkQueue transferQueue;
  VkCommandPool transferCmdPool;

  VkPhysicalDevice physicalDevice;
  VkDevice device;
  VkAllocationCallbacks* m_pAllocCB;

  void initBuffer(VkBuffer& buffer,
                  VkDeviceMemory& deviceMemory,
                  const void* srcData,
                  VkDeviceSize bufferSize,
                  VkBufferUsageFlagBits bufferUsage,
                  VkQueue transferQueue,
                  VkCommandPool transferCmdPool);
};

