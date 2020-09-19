#include "Mesh.h"

Mesh::Mesh(VkPhysicalDevice newPhysicalDevice,
           VkDevice newDevice,
           const std::vector<Vertex>& vertices,
           VkAllocationCallbacks* a_pAllocCB)
: vertexCount(vertices.size())
, physicalDevice(newPhysicalDevice)
, device(newDevice)
{
  createVertexbuffer(vertices);
}

Mesh::~Mesh()
{
  destroyVertexBuffer();
}

void Mesh::destroyVertexBuffer()
{
  if (vertexBuffer != VK_NULL_HANDLE)
  {
    vkDestroyBuffer(device, vertexBuffer, m_pAllocCB);
    vkFreeMemory(device, vertexBufferMemory, m_pAllocCB);
    vertexBuffer = VK_NULL_HANDLE;
  }
}

void Mesh::createVertexbuffer(const std::vector<Vertex>& vertices)
{
  // Information to create a buffer (doesn't include assigning memory)
  VkBufferCreateInfo bufferInfo = {};
  bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferInfo.size = sizeof(Vertex) * vertices.size();
  bufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
  bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;           // Similar to swap chain image, can share vertex buffers

  if (vkCreateBuffer(device, &bufferInfo, m_pAllocCB, &vertexBuffer) != VK_SUCCESS)
  {
    throw std::runtime_error("Unable to create buffer");
  }

  // Get buffer memory requirements
  VkMemoryRequirements memReqs = {};
  vkGetBufferMemoryRequirements(device, vertexBuffer, &memReqs);

  // Allocate memory to buffer
  VkMemoryAllocateInfo memAllocInfo = {};
  memAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  memAllocInfo.allocationSize = memReqs.size;
  memAllocInfo.memoryTypeIndex = findMemoryTypeIndex(memReqs.memoryTypeBits,
                                                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

  // Allocate memory to VkDeviceMemory
  if (vkAllocateMemory(device, &memAllocInfo, m_pAllocCB, &vertexBufferMemory) != VK_SUCCESS)
  {
    throw std::runtime_error("Failed to allocate vertex buffer memory");
  }

  // Allocate memory to given vertex buffer
  if (vkBindBufferMemory(device, vertexBuffer, vertexBufferMemory, 0) != VK_SUCCESS)
  {
    throw std::runtime_error("Failed to bind buffer memory vertex buffer");
  }

  // Map vertex data to vertex buffer
  void* data;
  vkMapMemory(device, vertexBufferMemory, 0, bufferInfo.size, 0, &data);
  memcpy(data, vertices.data(), static_cast<size_t>(bufferInfo.size));
  vkUnmapMemory(device, vertexBufferMemory);
}

uint32_t Mesh::findMemoryTypeIndex(uint32_t allowedTypes, VkMemoryPropertyFlags properties)
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
