#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <stdexcept>
#include <vector>
#include <iostream>

#include "VulkanRenderer.h"

GLFWwindow* initWindow(const std::string& wName = "Test Window", int width = 800, int height = 600)
{
  glfwInit();
  
  GLFWwindow* pWindow = glfwCreateWindow(width, height, wName.c_str(), nullptr, nullptr);

  // Set GLFW to NOT work with OpenGL
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

  return pWindow;
}

int main()
{
  if (GLFWwindow* pWindow = initWindow())
  {
    // Create Vulkan Renderer instance
    VulkanRenderer vulkanRenderer;
    vulkanRenderer.init(pWindow);

    uint32_t extensionCount = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);

    printf("Extension count %u\n", extensionCount);

    // Loop until closed
    while (!glfwWindowShouldClose(pWindow))
    {
      glfwPollEvents();
    }

    // Destroy GLFW window
    glfwDestroyWindow(pWindow);
    pWindow = nullptr;
  }

  // Stop GFLW
  glfwTerminate();

  return 0;
}