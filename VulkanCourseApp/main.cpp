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
      double angle = 0.0;
      double lastTime = 0.0;

      // Loop until closed
      while (!glfwWindowShouldClose(pWindow))
      {
        glfwPollEvents();

        double now = glfwGetTime();
        double deltaTime = now - lastTime;
        lastTime = now;

        // Update model rotation
        angle = fmod(10.0 * deltaTime + angle, 360.0);
        glm::mat4 mat = glm::rotate(glm::mat4(1.0f), glm::radians(static_cast<float>(angle)), glm::vec3(0.0f, 0.0f, 1.0f));

        vulkanRenderer.updateModel(mat);

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