#pragma once

#include <fstream>
#include <glm/glm.hpp>
#include <vector>

const int MAX_FRAME_DRAWS = 2;

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