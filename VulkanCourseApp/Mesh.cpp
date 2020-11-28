#include "Mesh.h"

Mesh::Mesh(VkPhysicalDevice newPhysicalDevice,
           VkDevice newDevice,
           VkQueue transferQueue,
           VkCommandPool transferCmdPool,
           const std::vector<Vertex>& vertices,
           const std::vector<uint32_t>& indices,
           VkAllocationCallbacks* a_pAllocCB)
: vertexCount(vertices.size())
, physicalDevice(newPhysicalDevice)
, device(newDevice)
, indexCount(indices.size())
{
  uboModel.model = glm::mat4(1.0f);

  if (vertices.size() != 0)
  {
    initBuffer(vertexBuffer, vertexBufferMemory, vertices.data(), sizeof(vertices[0]) * vertices.size(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, transferQueue, transferCmdPool);
  }
  if (indices.size() != 0)
  {
    initBuffer(indexBuffer, indexBufferMemory, indices.data(), sizeof(indices[0]) * indices.size(), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, transferQueue, transferCmdPool);
  }
}

Mesh::~Mesh()
{
  destroyBuffers();
}

void Mesh::destroyBuffers()
{
  if (vertexBuffer != VK_NULL_HANDLE)
  {
    vkDestroyBuffer(device, vertexBuffer, m_pAllocCB);
    vkFreeMemory(device, vertexBufferMemory, m_pAllocCB);
    vertexBuffer = VK_NULL_HANDLE;
    vertexCount = 0;
  }

  if (indexBuffer != VK_NULL_HANDLE)
  {
    vkDestroyBuffer(device, indexBuffer, m_pAllocCB);
    vkFreeMemory(device, indexBufferMemory, m_pAllocCB);
    indexBuffer = VK_NULL_HANDLE;
    indexCount = 0;
  }
}

void Mesh::initBuffer(VkBuffer& buffer,
                      VkDeviceMemory& deviceMemory,
                      const void* srcData,
                      VkDeviceSize bufferSize,
                      VkBufferUsageFlagBits bufferUsage,
                      VkQueue transferQueue,
                      VkCommandPool transferCmdPool)
{
  // Temporary buffer to stage vertex data before transferring to GPU
  VkBuffer stagingBuffer;
  VkDeviceMemory stagingBufferMemory;
  createBuffer(physicalDevice,
               device,
               bufferSize,
               VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
               &stagingBuffer,
               &stagingBufferMemory,
               m_pAllocCB);

  // Map vertex data to vertex buffer
  void* dstData;
  vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &dstData);
  memcpy(dstData, srcData, static_cast<size_t>(bufferSize));
  vkUnmapMemory(device, stagingBufferMemory);

  // Buffer only visible on GPU that would receive the data from the CPU visible buffer
  createBuffer(physicalDevice,
               device,
               bufferSize,
               bufferUsage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
               &buffer,
               &deviceMemory,
               m_pAllocCB);

  // Copy the buffer to the GPU
  copyBuffer(device, transferQueue, transferCmdPool, stagingBuffer, buffer, bufferSize);

  // Delete the staging buffer
  vkDestroyBuffer(device, stagingBuffer, m_pAllocCB);
  vkFreeMemory(device, stagingBufferMemory, m_pAllocCB);
}
