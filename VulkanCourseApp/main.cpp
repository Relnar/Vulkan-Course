#define STB_IMAGE_IMPLEMENTATION
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
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

      int helicopterModel = vulkanRenderer.createMeshModel("Models/Seahawk.obj");

      // Loop until closed
      while (!glfwWindowShouldClose(pWindow))
      {
        glfwPollEvents();

        double now = glfwGetTime();
        double deltaTime = now - lastTime;
        lastTime = now;

        // Update model rotation
        angle = fmod(10.0 * deltaTime + angle, 360.0);
        glm::mat4 matRotation = glm::translate(glm::mat4(1.0f), glm::vec3(-2.0f, 0.0f, -2.5f));
        matRotation = glm::rotate(matRotation, glm::radians(static_cast<float>(angle*5)), glm::vec3(0.0f, 1.0f, 0.0f));
        vulkanRenderer.updateModel(helicopterModel, matRotation);

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