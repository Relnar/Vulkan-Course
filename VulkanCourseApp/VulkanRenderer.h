#pragma once

//#define VK_USE_PLATFORM_WIN32_KHR
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <stdexcept>
#include <vector>
#include <set>
#include <algorithm>

#include "Mesh.h"
#include "stb_image.h"
#include "Utilities.h"

class VulkanRenderer
{
public:
  VulkanRenderer();
  ~VulkanRenderer();

  int init(GLFWwindow* a_pWindow);

  void updateModel(unsigned int modelId, const glm::mat4& newModel);

  void draw();
  void cleanup();

private:
  GLFWwindow* m_pWindow;
  bool m_bValidationLayers;

  int currentFrame = 0;

  std::vector<Mesh*> meshList;

  struct UboViewProjection
  {
    glm::mat4 projection;
    glm::mat4 view;
  } uboViewProjection;

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

  VkImage depthBufferImage;
  VkDeviceMemory depthBufferImageMemory;
  VkFormat depthBufferFormat;
  VkImageView depthBufferImageView;

  bool samplerAnisotropySupported;

  VkSampler textureSampler;

  VkDescriptorSetLayout descSetLayout;
  VkDescriptorSetLayout samplerSetLayout;
  VkPushConstantRange pushConstantRange;

  VkDescriptorPool descriptorPool;
  VkDescriptorPool samplerDescriptorPool;
  std::vector<VkDescriptorSet> descriptorSets;
  std::vector<VkDescriptorSet> samplerDescriptorSets;

  VkDeviceSize minUniformBufferOffset;
  size_t modelUniformAlignment;
  Model* modelTransferSpace;

  std::vector<VkBuffer> vpUniformBuffer;
  std::vector<VkDeviceMemory> vpUniformBufferMemory;

  std::vector<VkBuffer> modelUniformBufferDynamic;
  std::vector<VkDeviceMemory> modelUniformBufferMemoryDynamic;

  VkCommandPool graphicsCommandPool;

  std::vector<VkImage> textureImages;
  std::vector<VkDeviceMemory> textureImageMemory;
  std::vector<VkImageView> textureImageViews;

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
  void createDescriptorSetLayout();
  void createPushConstantRange();
  void createGraphicsPipeline();
  void createDepthBuffer();
  void createFramebuffers();
  void createCommandPool();
  void createCommandBuffers();
  void createSynchronization();
  void createTextureSampler();

  void createUniformBuffers();
  void createDescriptorPool();
  void createDescriptorSets();

  void updateUniformBuffers(uint32_t imageIndex);

  void recordCommands(uint32_t currentImage);

  void getPhysicalDevice();

  void allocateDynamicBufferTransferSpace();

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
  VkFormat chooseSupportedFormat(const std::vector<VkFormat>& formats, VkImageTiling tiling, VkFormatFeatureFlags featureFlags);

  std::tuple<VkImage, VkDeviceMemory> createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling,
                                                  VkImageUsageFlags useFlags, VkMemoryPropertyFlags propFlags);
  VkImageView createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags);
  VkShaderModule createShaderModule(const std::vector<char>& code);

  int createTextureImage(const std::string& filename);
  int createTexture(const std::string& filename);
  int createTextureDescriptor(VkImageView textureImage);

  stbi_uc* loadTextureFile(const std::string& filename, int& width, int& height, VkDeviceSize& imageSize);
};
