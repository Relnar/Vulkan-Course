#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <stdexcept>
#include <vector>

class VulkanRenderer
{
public:
  VulkanRenderer();
  ~VulkanRenderer();

  int init(GLFWwindow* a_pWindow);

private:
  GLFWwindow* m_pWindow;

  // Vulkan Components
  VkInstance instance;

  // Vulkan Functions
  void createInstance();
};

