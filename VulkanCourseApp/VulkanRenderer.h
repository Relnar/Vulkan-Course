#pragma once

//#define VK_USE_PLATFORM_WIN32_KHR
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <stdexcept>
#include <vector>
#include <set>
#include <algorithm>

#include "Mesh.h"
#include "Utilities.h"

class VulkanRenderer
{
public:
  VulkanRenderer();
  ~VulkanRenderer();

  int init(GLFWwindow* a_pWindow);
  void draw();
  void cleanup();

private:
  GLFWwindow* m_pWindow;
  bool m_bValidationLayers;

  int currentFrame = 0;

  std::vector<Mesh*> meshList;

  // Vulkan Components
  VkInstance instance;
  VkAllocationCallbacks* m_pAllocCB;
  VkDebugUtilsMessengerEXT debugMessenger;
  struct
  {
    VkPhysicalDevice physicalDevice;
    VkDevice logicalDevice;
  } mainDevice;
  VkQueue graphicsQueue;
  VkQueue presentationQueue;
  VkSurfaceKHR surface;
  VkSwapchainKHR swapchain;

  std::vector<SwapchainImage> swapChainImages;
  std::vector<VkFramebuffer> swapChainFramebuffers;
  std::vector<VkCommandBuffer> commandBuffers;

  VkCommandPool graphicsCommandPool;

  VkPipeline graphicsPipeline;
  VkPipelineLayout pipelineLayout;
  VkRenderPass renderPass;

  VkFormat swapChainImageFormat;
  VkExtent2D swapChainExtent;

  std::vector<VkSemaphore> imageAvailable;
  std::vector<VkSemaphore> renderFinished;
  std::vector<VkFence> drawFences;

  // Vulkan Functions
  void createInstance();
  void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo);
  void setupDebugMessenger();
  void createLogicalDevice();
  void createSurface();
  void createSwapChain();
  void createRenderPass();
  void createGraphicsPipeline();
  void createFramebuffers();
  void createCommandPool();
  void createCommandBuffers();
  void createSynchronization();

  void recordCommands();

  void getPhysicalDevice();

  bool checkInstanceExtensionSupport(const std::vector<const char*>& a_rExtensions);
  bool checkDeviceExtensionSupport(VkPhysicalDevice device);
  bool checkValidationLayerSupport();
  std::vector<const char*> getRequiredExtensions();
  bool checkSuitableDevice(VkPhysicalDevice device);

  QueueFamilyIndices getQueueFamilies(VkPhysicalDevice device);
  SwapChainDetails getSwapChainDetails(VkPhysicalDevice device);

  VkSurfaceFormatKHR chooseBestSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& formats);
  VkPresentModeKHR chooseBestPresentationMode(const std::vector<VkPresentModeKHR>& presentationModes);
  VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& surfaceCapabilities);

  VkImageView createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags);
  VkShaderModule createShaderModule(const std::vector<char>& code);
};
