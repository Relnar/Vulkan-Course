#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <stdexcept>
#include <vector>
#include <iostream>

#include "VulkanRenderer.h"

GLFWwindow* initWindow(const std::string& wName = "Test Window", int width = 800, int height = 600)
{
  glfwInit();
  
  // Set GLFW to NOT work with OpenGL
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

  return glfwCreateWindow(width, height, wName.c_str(), nullptr, nullptr);
}

int main()
{
  if (GLFWwindow* pWindow = initWindow())
  {
    // Create Vulkan Renderer instance
    VulkanRenderer vulkanRenderer;
    if (vulkanRenderer.init(pWindow) == EXIT_SUCCESS)
    {
      // Loop until closed
      while (!glfwWindowShouldClose(pWindow))
      {
        glfwPollEvents();
        vulkanRenderer.draw();
      }
    }

    vulkanRenderer.cleanup();

    // Destroy GLFW window
    glfwDestroyWindow(pWindow);
    pWindow = nullptr;
  }

  // Stop GFLW
  glfwTerminate();

  return EXIT_SUCCESS;
}