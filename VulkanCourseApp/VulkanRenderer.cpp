#include "VulkanRenderer.h"
#include <iostream>
#include <array>

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#define USING_PUSH_CONSTANT

static const std::vector<const char*> validationLayers =
{
  "VK_LAYER_KHRONOS_validation"
};

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                                                    VkDebugUtilsMessageTypeFlagsEXT messageType,
                                                    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
                                                    void* pUserData)
{

  std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

  return VK_FALSE;
}

VkResult CreateDebugUtilsMessengerEXT(VkInstance instance,
                                      const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
                                      const VkAllocationCallbacks* pAllocator,
                                      VkDebugUtilsMessengerEXT* pDebugMessenger)
{
  auto func = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
  if (func != nullptr)
  {
    return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
  }
  else
  {
    return VK_ERROR_EXTENSION_NOT_PRESENT;
  }
}

void DestroyDebugUtilsMessengerEXT(VkInstance instance,
                                   VkDebugUtilsMessengerEXT debugMessenger,
                                   const VkAllocationCallbacks* pAllocator)
{
  auto func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
  if (func != nullptr)
  {
    func(instance, debugMessenger, pAllocator);
  }
}

VulkanRenderer::VulkanRenderer()
: m_pWindow(nullptr)
, m_pAllocCB(nullptr)
#ifdef NDEBUG
, m_bValidationLayers(false)
#else
, m_bValidationLayers(true)
#endif
, minUniformBufferOffset(256)
, modelUniformAlignment(256)
, modelTransferSpace(nullptr)
, samplerAnisotropySupported(false)
{
}

VulkanRenderer::~VulkanRenderer()
{
}

int VulkanRenderer::init(GLFWwindow* a_pWindow)
{
  m_pWindow = a_pWindow;

  try
  {
    createInstance();
    setupDebugMessenger();
    createSurface();
    getPhysicalDevice();
    createLogicalDevice();
    createSwapChain();
    createColorBufferImage();
    createDepthBuffer();
    createRenderPass();
    createDescriptorSetLayout();
#ifdef USING_PUSH_CONSTANT
    createPushConstantRange();
#endif
    createGraphicsPipeline();
    createFramebuffers();
    createCommandPool();
    createCommandBuffers();
    createTextureSampler();
#ifndef USING_PUSH_CONSTANT
    allocateDynamicBufferTransferSpace();
#endif
    createUniformBuffers();
    createDescriptorPool();
    createDescriptorSets();
    createInputDescriptorSets();
    createSynchronization();
  }
  catch (const std::runtime_error& e)
  {
    printf("Error: %s\n", e.what());
    return EXIT_FAILURE;
  }

  uboViewProjection.projection = glm::perspective(glm::radians(45.0f), static_cast<float>(swapChainExtent.width) / swapChainExtent.height, 0.1f, 5000.0f);
  uboViewProjection.view = glm::lookAt(glm::vec3(0.0f, 50.0f, 250.0f), glm::vec3(0.0f, 0.0f, -2.0f), glm::vec3(0.0, 1.0f, 0.0f));

  // Vulkan Y-up is inverted compared to OpenGL
  uboViewProjection.projection[1][1] *= -1.0f;

  createTexture("plain.png");

  return EXIT_SUCCESS;
}

void VulkanRenderer::updateModel(unsigned int modelId, const glm::mat4& newModel)
{
  if (modelId >= modelList.size())
  {
    return;
  }

  modelList[modelId]->setModelMatrix(newModel);
}

void VulkanRenderer::draw()
{
  // Wait for given fence to signal (open) from last draw before continuing
  vkWaitForFences(mainDevice.logicalDevice, 1, &drawFences[currentFrame], VK_TRUE, std::numeric_limits<uint64_t>::max());
  // Manually reset (close) the fence
  vkResetFences(mainDevice.logicalDevice, 1, &drawFences[currentFrame]);

  // Get the next available image to draw to and set signal when we're finished with the image (semaphore)
  uint32_t imageIndex = 0;
  vkAcquireNextImageKHR(mainDevice.logicalDevice, swapchain, std::numeric_limits<uint64_t>::max(), imageAvailable[currentFrame], VK_NULL_HANDLE, &imageIndex);

  recordCommands(imageIndex);
  updateUniformBuffers(imageIndex);

  // Submit command buffer to queue for execution, make sure ti waits for image to be signalled as available before drawing
  // and signals when it has finished rendering
  VkSubmitInfo submitInfo = {};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.waitSemaphoreCount = 1;
  submitInfo.pWaitSemaphores = &imageAvailable[currentFrame];
  VkPipelineStageFlags waitStages[] =
  {
    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
  };
  submitInfo.pWaitDstStageMask = waitStages;          // Stages to check semaphores at
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &commandBuffers[imageIndex];
  submitInfo.signalSemaphoreCount = 1;
  submitInfo.pSignalSemaphores = &renderFinished[currentFrame];

  VkResult result = vkQueueSubmit(graphicsQueue, 1, &submitInfo, drawFences[currentFrame]);
  if (result != VK_SUCCESS)
  {
    throw std::runtime_error("Failed to submit command buffer to queue");
  }

  // Present image to screen when it has signalled finished rendering
  VkPresentInfoKHR presentInfo = {};
  presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
  presentInfo.waitSemaphoreCount = 1;
  presentInfo.pWaitSemaphores = &renderFinished[currentFrame];
  presentInfo.swapchainCount = 1;
  presentInfo.pSwapchains = &swapchain;                   // Swapchain to present image to
  presentInfo.pImageIndices = &imageIndex;

  result = vkQueuePresentKHR(presentationQueue, &presentInfo);
  if (result != VK_SUCCESS)
  {
    throw std::runtime_error("Failed to present image");
  }

  currentFrame = (currentFrame + 1) % MAX_FRAME_DRAWS;
}

void VulkanRenderer::cleanup()
{
  // Wait until no actions being run on device before destroying
  vkDeviceWaitIdle(mainDevice.logicalDevice);

#ifndef USING_PUSH_CONSTANT
  _aligned_free(modelTransferSpace);
  modelTransferSpace = nullptr;
#endif

  for (auto& model : modelList)
  {
    delete model;
  }
  modelList.clear();

  vkDestroySampler(mainDevice.logicalDevice, textureSampler, m_pAllocCB);

  for (size_t i = 0; i < textureImages.size(); ++i)
  {
    vkDestroyImageView(mainDevice.logicalDevice, textureImageViews[i], m_pAllocCB);
    vkDestroyImage(mainDevice.logicalDevice, textureImages[i], m_pAllocCB);
    vkFreeMemory(mainDevice.logicalDevice, textureImageMemory[i], m_pAllocCB);
  }
  textureImages.clear();
  textureImageMemory.clear();
  textureImageViews.clear();

  for (size_t i = 0; i < colorBufferImage.size(); ++i)
  {
    vkDestroyImageView(mainDevice.logicalDevice, colorBufferImageView[i], m_pAllocCB);
    vkDestroyImage(mainDevice.logicalDevice, colorBufferImage[i], m_pAllocCB);
    vkFreeMemory(mainDevice.logicalDevice, colorBufferImageMemory[i], m_pAllocCB);
  }

  for (size_t i = 0; i < depthBufferImage.size(); ++i)
  {
    vkDestroyImageView(mainDevice.logicalDevice, depthBufferImageView[i], m_pAllocCB);
    vkDestroyImage(mainDevice.logicalDevice, depthBufferImage[i], m_pAllocCB);
    vkFreeMemory(mainDevice.logicalDevice, depthBufferImageMemory[i], m_pAllocCB);
  }

  vkDestroyDescriptorPool(mainDevice.logicalDevice, descriptorPool, m_pAllocCB);
  vkDestroyDescriptorSetLayout(mainDevice.logicalDevice, descSetLayout, m_pAllocCB);

  vkDestroyDescriptorPool(mainDevice.logicalDevice, samplerDescriptorPool, m_pAllocCB);
  vkDestroyDescriptorSetLayout(mainDevice.logicalDevice, samplerSetLayout, m_pAllocCB);

  vkDestroyDescriptorPool(mainDevice.logicalDevice, inputDescriptorPool, m_pAllocCB);
  vkDestroyDescriptorSetLayout(mainDevice.logicalDevice, inputSetLayout, m_pAllocCB);

  for (size_t i = 0; i < swapChainImages.size(); ++i)
  {
    vkDestroyBuffer(mainDevice.logicalDevice, vpUniformBuffer[i], m_pAllocCB);
    vkFreeMemory(mainDevice.logicalDevice, vpUniformBufferMemory[i], m_pAllocCB);
#ifndef USING_PUSH_CONSTANT
    vkDestroyBuffer(mainDevice.logicalDevice, modelUniformBufferDynamic[i], m_pAllocCB);
    vkFreeMemory(mainDevice.logicalDevice, modelUniformBufferMemoryDynamic[i], m_pAllocCB);
#endif
  }

  for (int i = 0; i < MAX_FRAME_DRAWS; ++i)
  {
    vkDestroySemaphore(mainDevice.logicalDevice, renderFinished[i], m_pAllocCB);
    vkDestroySemaphore(mainDevice.logicalDevice, imageAvailable[i], m_pAllocCB);
    vkDestroyFence(mainDevice.logicalDevice, drawFences[i], m_pAllocCB);
  }
  vkDestroyCommandPool(mainDevice.logicalDevice, graphicsCommandPool, m_pAllocCB);
  for (auto& framebuffer : swapChainFramebuffers)
  {
    vkDestroyFramebuffer(mainDevice.logicalDevice, framebuffer, m_pAllocCB);
  }
  vkDestroyPipeline(mainDevice.logicalDevice, secondPipeline, m_pAllocCB);
  vkDestroyPipelineLayout(mainDevice.logicalDevice, secondPipelineLayout, m_pAllocCB);
  vkDestroyPipeline(mainDevice.logicalDevice, graphicsPipeline, m_pAllocCB);
  vkDestroyPipelineLayout(mainDevice.logicalDevice, pipelineLayout, m_pAllocCB);
  vkDestroyRenderPass(mainDevice.logicalDevice, renderPass, m_pAllocCB);
  for (auto& image : swapChainImages)
  {
    vkDestroyImageView(mainDevice.logicalDevice, image.imageView, m_pAllocCB);
  }
  vkDestroySwapchainKHR(mainDevice.logicalDevice, swapchain, m_pAllocCB);
  vkDestroySurfaceKHR(instance, surface, m_pAllocCB);
  vkDestroyDevice(mainDevice.logicalDevice, m_pAllocCB);

  if (m_bValidationLayers)
  {
    DestroyDebugUtilsMessengerEXT(instance, debugMessenger, m_pAllocCB);
  }

  vkDestroyInstance(instance, m_pAllocCB);
}

void VulkanRenderer::createInstance()
{
  if (!checkValidationLayerSupport())
  {
    throw std::runtime_error("Validation layers requesed, but not available !");
  }

  // Information about the application itself
  // Data is for developer convenience
  VkApplicationInfo appInfo = {};
  appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.pApplicationName = "Vulkan App";
  appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.pEngineName = "No Engine";
  appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  // Vulcan 1.2
  appInfo.apiVersion = VK_API_VERSION_1_2;
 
  // Creation information for a VkInstance
  VkInstanceCreateInfo createInfo = {};
  createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  createInfo.pApplicationInfo = &appInfo;

  // Create list of hold instance extensions
  std::vector<const char*> instanceExtensions = getRequiredExtensions();

  // Check if instance extensions are supported
  if (!checkInstanceExtensionSupport(instanceExtensions))
  {
    throw std::runtime_error("VkInstance doesnt not support required extensions!");
  }

  createInfo.enabledExtensionCount = static_cast<uint32_t>(instanceExtensions.size());
  createInfo.ppEnabledExtensionNames = instanceExtensions.data();

  // Create info must outside the if statement to not be destroyed before vkCreateinstance is called
  VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo;
  if (m_bValidationLayers)
  {
    createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
    createInfo.ppEnabledLayerNames = validationLayers.data();

    populateDebugMessengerCreateInfo(debugCreateInfo);
    createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;
  }
  else
  {
    createInfo.enabledLayerCount = 0;
    createInfo.ppEnabledLayerNames = nullptr;
  }
 
  // Create instance
  if (vkCreateInstance(&createInfo, m_pAllocCB, &instance) != VK_SUCCESS)
  {
    throw std::runtime_error("Failed to create a Vulkan instance");
  }
}

void VulkanRenderer::populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo)
{
  createInfo = {};
  createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
  createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                               VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                               VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
  createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
  createInfo.pfnUserCallback = debugCallback;
  createInfo.pUserData = nullptr; // Optional
}

void VulkanRenderer::setupDebugMessenger()
{
  if (!m_bValidationLayers)
  {
    return;
  }

  VkDebugUtilsMessengerCreateInfoEXT createInfo;
  populateDebugMessengerCreateInfo(createInfo);

  if (CreateDebugUtilsMessengerEXT(instance, &createInfo, m_pAllocCB, &debugMessenger) != VK_SUCCESS)
  {
    throw std::runtime_error("Failed to setup debug messenger");
  }
}

void VulkanRenderer::createLogicalDevice()
{
  QueueFamilyIndices indices = getQueueFamilies(mainDevice.physicalDevice);

  std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
  std::set<int> queueFamilyIndices = { indices.graphicsFamily, indices.presentationFamily };

  for (int queueFamilyIndex : queueFamilyIndices)
  {
    VkDeviceQueueCreateInfo queueCreateInfo = {};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = queueFamilyIndex;
    queueCreateInfo.queueCount = 1;
    float priority = 1.0f;
    queueCreateInfo.pQueuePriorities = &priority;

    queueCreateInfos.push_back(queueCreateInfo);
  }

  VkDeviceCreateInfo deviceCreateInfo = {};
  deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  deviceCreateInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
  deviceCreateInfo.pQueueCreateInfos = queueCreateInfos.data();
  deviceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size()); // Number of enabled logical device extensions.
  deviceCreateInfo.ppEnabledExtensionNames = deviceExtensions.data();

  // Deprecated in Vulkan 1.1
  deviceCreateInfo.enabledLayerCount = 0;
  deviceCreateInfo.ppEnabledLayerNames = nullptr;

  VkPhysicalDeviceFeatures deviceFeatures = {};
  deviceFeatures.samplerAnisotropy = samplerAnisotropySupported ? VK_TRUE : VK_FALSE;
  deviceCreateInfo.pEnabledFeatures = &deviceFeatures;

  // Create the logical device
  if (vkCreateDevice(mainDevice.physicalDevice, &deviceCreateInfo, m_pAllocCB, &mainDevice.logicalDevice) != VK_SUCCESS)
  {
    throw std::runtime_error("Failed to create the logical device");
  }

  // Queues are created at the same time as the device
  vkGetDeviceQueue(mainDevice.logicalDevice, indices.graphicsFamily, 0, &graphicsQueue);
  vkGetDeviceQueue(mainDevice.logicalDevice, indices.presentationFamily, 0, &presentationQueue);
}

void VulkanRenderer::createSurface()
{
  // Create surface (creating a surface create info struct,
  //                 runs the create surface function,
  //                 returns a VkResult
  VkResult result = glfwCreateWindowSurface(instance, m_pWindow, m_pAllocCB, &surface);
  if (result != VK_SUCCESS)
  {
    throw std::runtime_error("Failed to create a surface");
  }
}

void VulkanRenderer::createSwapChain()
{
  SwapChainDetails swapChainDetails = getSwapChainDetails(mainDevice.physicalDevice);

  VkSurfaceFormatKHR surfaceFormat = chooseBestSurfaceFormat(swapChainDetails.formats);

  VkSwapchainCreateInfoKHR swapChainCreateInfo = {};
  swapChainCreateInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
  swapChainCreateInfo.imageFormat = surfaceFormat.format;
  swapChainCreateInfo.imageColorSpace = surfaceFormat.colorSpace;
  swapChainCreateInfo.presentMode = chooseBestPresentationMode(swapChainDetails.presentationModes);
  swapChainCreateInfo.imageExtent = chooseSwapExtent(swapChainDetails.surfaceCapabilities);

  // Get 1 more than the minimum to allow triple buffering
  // Also, maxImageCount can be 0 meaning no limit, so make sure not to set the value 0 for the minImageCount
  swapChainCreateInfo.minImageCount = std::min(swapChainDetails.surfaceCapabilities.minImageCount + 1,
                                               std::max(swapChainDetails.surfaceCapabilities.maxImageCount,
                                                        swapChainDetails.surfaceCapabilities.minImageCount + 1));

  swapChainCreateInfo.imageArrayLayers = 1;                               // Number of layers for each image in chain
  swapChainCreateInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;   // Attachments, usually only color (not often depth)
  swapChainCreateInfo.preTransform = swapChainDetails.surfaceCapabilities.currentTransform;
  swapChainCreateInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  swapChainCreateInfo.clipped = VK_TRUE;                                  // Whether to clip parts of image not in view
  swapChainCreateInfo.surface = surface;

  QueueFamilyIndices indices = getQueueFamilies(mainDevice.physicalDevice);
  uint32_t queueFamilyIndices[] =
  {
    static_cast<uint32_t>(indices.graphicsFamily),
    static_cast<uint32_t>(indices.presentationFamily),
  };

  // If graphics and presentation families are different,
  // then swapchain must let images be shared between families
  if (indices.graphicsFamily != indices.presentationFamily)
  {
    swapChainCreateInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
    swapChainCreateInfo.queueFamilyIndexCount = 2;
    swapChainCreateInfo.pQueueFamilyIndices = queueFamilyIndices;
  }
  else
  {
    swapChainCreateInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    swapChainCreateInfo.queueFamilyIndexCount = 0;
    swapChainCreateInfo.pQueueFamilyIndices = nullptr;
  }

  // If old swap chain been destroyed and this one replaces it,
  // then link old one to quickly hand over responsibilities
  swapChainCreateInfo.oldSwapchain = VK_NULL_HANDLE;

  if (vkCreateSwapchainKHR(mainDevice.logicalDevice, &swapChainCreateInfo, m_pAllocCB, &swapchain) != VK_SUCCESS)
  {
    throw std::runtime_error("Failed to create swapchain");
  }

  swapChainImageFormat = surfaceFormat.format;
  swapChainExtent = swapChainCreateInfo.imageExtent;

  // Get swap chain images
  uint32_t swapChainImageCount;
  vkGetSwapchainImagesKHR(mainDevice.logicalDevice, swapchain, &swapChainImageCount, nullptr);
  std::vector<VkImage> images(swapChainImageCount);
  vkGetSwapchainImagesKHR(mainDevice.logicalDevice, swapchain, &swapChainImageCount, images.data());

  for (VkImage image : images)
  {
    SwapchainImage swapChainImage = {};
    swapChainImage.image = image;
    swapChainImage.imageView = createImageView(image, swapChainImageFormat, VK_IMAGE_ASPECT_COLOR_BIT);

    swapChainImages.push_back(swapChainImage);
  }
}

void VulkanRenderer::createRenderPass()
{
  std::array<VkSubpassDescription, 2> subpasses = {};

  // Subpass 1
  std::array<VkAttachmentDescription, 2> subpass1AttachmentDesc = {};
  std::array<VkAttachmentReference, 2> subpass1AttachmentRef = {};

  // Color attachment
  VkAttachmentDescription& colorAttachment = subpass1AttachmentDesc[0];
  colorAttachment.format = chooseSupportedFormat({ VK_FORMAT_R8G8B8A8_UNORM },
                                                 VK_IMAGE_TILING_OPTIMAL,
                                                 VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
  colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;                        // MSAA count
  colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;                   // Clear color before rendering
  colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;             // What to do after rendering
  colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;              // Image data layout before render pass starts
  colorAttachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; // Image data layout after render pass (to change to)

  // Depth attachment
  VkAttachmentDescription& depthAttachment = subpass1AttachmentDesc[1];
  depthAttachment.format = depthBufferFormat;
  depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
  depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;                   // Clear depth before rendering
  depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;             // What to do after rendering
  depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

  // Attachment reference uses an attachment index that refers to index in the attachment list passed to renderPassCreate
  VkAttachmentReference& colorAttachmentRef = subpass1AttachmentRef[0];
  colorAttachmentRef.attachment = 1;
  colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

  VkAttachmentReference& depthAttachmentRef = subpass1AttachmentRef[1];
  depthAttachmentRef.attachment = 2;
  depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

  VkSubpassDescription& subpass1 = subpasses[0];
  subpass1.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass1.colorAttachmentCount = 1;
  subpass1.pColorAttachments = &colorAttachmentRef;
  subpass1.pDepthStencilAttachment = &depthAttachmentRef;

  // Subpass 2
  std::array<VkAttachmentDescription, 3> renderPassAttachments = { colorAttachment, colorAttachment, depthAttachment };

  // Swapchain color attachment
  VkAttachmentDescription& swapchainColorAttachment = renderPassAttachments[0];
  swapchainColorAttachment.format = swapChainImageFormat;
  swapchainColorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;                    // MSAA count
  swapchainColorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;               // Clear color before rendering
  swapchainColorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;             // What to do after rendering
  swapchainColorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  swapchainColorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

  // Framebuffer data will be stored as an image, but images can be given different data layouts
  swapchainColorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;          // Image data layout before render pass starts
  swapchainColorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;      // Image data layout after render pass (to change to)

  // Attachment reference uses an attachment index that refers to index in the attachment list passed to renderPassCreate
  VkAttachmentReference swapchainColorAttachmentRef = {};
  swapchainColorAttachmentRef.attachment = 0;
  swapchainColorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

  std::array<VkAttachmentReference, 2> inputReferences = {};
  inputReferences[0].attachment = 1;
  inputReferences[0].layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  inputReferences[1].attachment = 2;
  inputReferences[1].layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

  // Information about a particular subpass the render pass is using
  VkSubpassDescription& subpass2 = subpasses[1];
  subpass2.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass2.colorAttachmentCount = 1;
  subpass2.pColorAttachments = &swapchainColorAttachmentRef;
  subpass2.inputAttachmentCount = inputReferences.size();
  subpass2.pInputAttachments = inputReferences.data();

  // Subpass dependencies

  // Need to determine when layout transitions occur using subpass dependencies
  std::array<VkSubpassDependency, 3> subpassDependencies = {};

  // Conversion from VK_IMAGE_LAYOUT_UNDEFINED to VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
  // Transition must happen after ...
  subpassDependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;                      // Subpass index (VK_SUBPASS_EXTERNAL = Special value meaning outside of render pass(
  subpassDependencies[0].srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;   // Pipeline stage
  subpassDependencies[0].srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;             // Stage access mask (memory access)
  // But must happen before ...
  subpassDependencies[0].dstSubpass = 0;
  subpassDependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  subpassDependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  subpassDependencies[0].dependencyFlags = 0;

  // Subpass 1 layout to Subpass 2 layout
  subpassDependencies[1].srcSubpass = 0;
  subpassDependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  subpassDependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  subpassDependencies[1].dstSubpass = 1;
  subpassDependencies[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
	subpassDependencies[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  subpassDependencies[1].dependencyFlags = 0;

  // Conversion from VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL to VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
  subpassDependencies[2].srcSubpass = 0;
  subpassDependencies[2].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  subpassDependencies[2].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  subpassDependencies[2].dstSubpass = VK_SUBPASS_EXTERNAL;
  subpassDependencies[2].dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
  subpassDependencies[2].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
  subpassDependencies[2].dependencyFlags = 0;

  VkRenderPassCreateInfo renderPassCreate = {};
  renderPassCreate.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  renderPassCreate.attachmentCount = renderPassAttachments.size();
  renderPassCreate.pAttachments = renderPassAttachments.data();
  renderPassCreate.subpassCount = subpasses.size();
  renderPassCreate.pSubpasses = subpasses.data();
  renderPassCreate.dependencyCount = static_cast<uint32_t>(subpassDependencies.size());
  renderPassCreate.pDependencies = subpassDependencies.data();

  if (vkCreateRenderPass(mainDevice.logicalDevice, &renderPassCreate, m_pAllocCB, &renderPass) != VK_SUCCESS)
  {
    throw std::runtime_error("Failed to create render pass");
  }
}

void VulkanRenderer::createDescriptorSetLayout()
{
  // Uniform value DescriptorSetLayout
#ifndef USING_PUSH_CONSTANT
  std::array<VkDescriptorSetLayoutBinding, 2> layoutBindings = {};
#else
  std::array<VkDescriptorSetLayoutBinding, 1> layoutBindings = {};
#endif

  // ViewProjection binding info
  VkDescriptorSetLayoutBinding& vpLayoutBinding = layoutBindings[0];
  vpLayoutBinding.binding = 0;           // Must match the binding number in the shader
  vpLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  vpLayoutBinding.descriptorCount = 1;
  vpLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
  vpLayoutBinding.pImmutableSamplers = nullptr;

#ifndef USING_PUSH_CONSTANT
  // Model binding info
  VkDescriptorSetLayoutBinding& modelLayoutBinding = layoutBindings[1];
  modelLayoutBinding.binding = 1;           // Must match the binding number in the shader
  modelLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
  modelLayoutBinding.descriptorCount = 1;
  modelLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
  modelLayoutBinding.pImmutableSamplers = nullptr;

  const VkDescriptorSetLayoutBinding* bindings[] = { &vpLayoutBinding, &modelLayoutBinding };
#else
  const VkDescriptorSetLayoutBinding* bindings[] = { &vpLayoutBinding };
#endif

  VkDescriptorSetLayoutCreateInfo layoutCreateInfo = {};
  layoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layoutCreateInfo.bindingCount = layoutBindings.size();
  layoutCreateInfo.pBindings = layoutBindings.data();

  if (vkCreateDescriptorSetLayout(mainDevice.logicalDevice, &layoutCreateInfo, m_pAllocCB, &descSetLayout) != VK_SUCCESS)
  {
    throw std::runtime_error("Failed to create uniform descriptor set layout");
  }

  // Sampler DescriptorSetLayout

  // Sampler binding info
  VkDescriptorSetLayoutBinding samplerLayoutBinding = {};
  samplerLayoutBinding.binding = 0;           // Must match the binding number in the shader
  samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  samplerLayoutBinding.descriptorCount = 1;
  samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
  samplerLayoutBinding.pImmutableSamplers = nullptr;

  VkDescriptorSetLayoutCreateInfo textureLayoutCreateInfo = {};
  textureLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  textureLayoutCreateInfo.bindingCount = 1;
  textureLayoutCreateInfo.pBindings = &samplerLayoutBinding;

  if (vkCreateDescriptorSetLayout(mainDevice.logicalDevice, &textureLayoutCreateInfo, m_pAllocCB, &samplerSetLayout) != VK_SUCCESS)
  {
    throw std::runtime_error("Failed to create sampler descriptor set layout");
  }

  // Input attachment image descriptor set layout
  std::array<VkDescriptorSetLayoutBinding, 2> inputBindings = {};
  VkDescriptorSetLayoutBinding& colorInputBinding = inputBindings[0];
  colorInputBinding.binding = 0;           // Must match the binding number in the shader
  colorInputBinding.descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
  colorInputBinding.descriptorCount = 1;
  colorInputBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
  colorInputBinding.pImmutableSamplers = nullptr;

  VkDescriptorSetLayoutBinding& depthInputBinding = inputBindings[1];
  depthInputBinding.binding = 1;           // Must match the binding number in the shader
  depthInputBinding.descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
  depthInputBinding.descriptorCount = 1;
  depthInputBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
  depthInputBinding.pImmutableSamplers = nullptr;

  VkDescriptorSetLayoutCreateInfo inputLayoutCreateInfo = {};
  inputLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  inputLayoutCreateInfo.bindingCount = inputBindings.size();
  inputLayoutCreateInfo.pBindings = inputBindings.data();

  if (vkCreateDescriptorSetLayout(mainDevice.logicalDevice, &inputLayoutCreateInfo, m_pAllocCB, &inputSetLayout) != VK_SUCCESS)
  {
    throw std::runtime_error("Failed to create input descriptor set layout");
  }
}

void VulkanRenderer::createPushConstantRange()
{
  pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
  pushConstantRange.offset = 0;
  pushConstantRange.size = sizeof(Model);
}

void VulkanRenderer::createGraphicsPipeline()
{
  // Read shader files
#ifndef USING_PUSH_CONSTANT
  auto vertexShader = readFile("Shaders/vert.spv");
#else
  auto vertexShader = readFile("Shaders/vertPushConstant.spv");
#endif
  auto fragmentShader = readFile("Shaders/frag.spv");

  // Build shader modules to link to graphics pipeline
  VkShaderModule vertexShaderModule = createShaderModule(vertexShader);
  VkShaderModule fragmentShaderModule = createShaderModule(fragmentShader);

  // Shader sage creation information
  VkPipelineShaderStageCreateInfo vertexShaderCreateInfo = {};
  vertexShaderCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  vertexShaderCreateInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
  vertexShaderCreateInfo.module = vertexShaderModule;
  vertexShaderCreateInfo.pName = "main";

  VkPipelineShaderStageCreateInfo fragmentShaderCreateInfo = {};
  fragmentShaderCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  fragmentShaderCreateInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  fragmentShaderCreateInfo.module = fragmentShaderModule;
  fragmentShaderCreateInfo.pName = "main";

  VkPipelineShaderStageCreateInfo shaderStages[] = { vertexShaderCreateInfo, fragmentShaderCreateInfo };

  //
  // Create pipeline
  //
  VkVertexInputBindingDescription bindingDesc = {};
  bindingDesc.binding = 0;
  bindingDesc.stride = sizeof(Vertex);
  bindingDesc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

  std::array<VkVertexInputAttributeDescription, 3> attribDescs = {};

  // Position attribute
  attribDescs[0].binding = 0;
  attribDescs[0].location = 0;
  attribDescs[0].format = VK_FORMAT_R32G32B32_SFLOAT;
  attribDescs[0].offset = offsetof(Vertex, pos);          // Find offset in struct for pos attribute

  // Color attribute
  attribDescs[1].binding = 0;
  attribDescs[1].location = 1;
  attribDescs[1].format = VK_FORMAT_R32G32B32_SFLOAT;
  attribDescs[1].offset = offsetof(Vertex, col);          // Find offset in struct for pos attribute

  // Texcoord attribute
  attribDescs[2].binding = 0;
  attribDescs[2].location = 2;
  attribDescs[2].format = VK_FORMAT_R32G32_SFLOAT;
  attribDescs[2].offset = offsetof(Vertex, tex);          // Find offset in struct for pos attribute

  // Vertex input
  VkPipelineVertexInputStateCreateInfo vertexInputCreateInfo = {};
  vertexInputCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  vertexInputCreateInfo.vertexBindingDescriptionCount = 1;
  vertexInputCreateInfo.pVertexBindingDescriptions = &bindingDesc;    // data spacing, stride info
  vertexInputCreateInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attribDescs.size());
  vertexInputCreateInfo.pVertexAttributeDescriptions = attribDescs.data();   // data format and where to bind to/from

  // Input assembly
  VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
  inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;   // primitive type
  inputAssembly.primitiveRestartEnable = VK_FALSE;

  // Viewport and scissor
  VkViewport viewport = {};
  viewport.x = 0.0f;
  viewport.y = 0.0f;
  viewport.width = (float)swapChainExtent.width;
  viewport.height = (float)swapChainExtent.height;
  viewport.minDepth = 0.0f;
  viewport.maxDepth = 1.0f;

  // Create scissor info struct
  VkRect2D scissor = {};
  scissor.offset = { 0, 0 };
  scissor.extent = swapChainExtent;

  VkPipelineViewportStateCreateInfo viewportCreateInfo = {};
  viewportCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  viewportCreateInfo.viewportCount = 1;
  viewportCreateInfo.pViewports = &viewport;
  viewportCreateInfo.scissorCount = 1;
  viewportCreateInfo.pScissors = &scissor;

  // Dynamic state
//   std::vector<VkDynamicState> dynamicStateEnable =
//   {
//     VK_DYNAMIC_STATE_VIEWPORT,      // Can resize in command buffer: vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
//     VK_DYNAMIC_STATE_SCISSOR        // Can resize in command buffer: vkCmdSetScissor(commandBuffer, 0, 1, &scissor);
//   };
//   VkPipelineDynamicStateCreateInfo dynamicState = {};
//   dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
//   dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStateEnable.size());
//   dynamicState.pDynamicStates = dynamicStateEnable.data();

  // Rasterizer
  VkPipelineRasterizationStateCreateInfo rasterizer = {};
  rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  rasterizer.depthClampEnable = VK_FALSE;         // Change if fragments beyond near/far planes are clipped (default) or clamped to plane
                                                  // Need to enable depthClamp flag in VkPhysicalDeviceFeatures
  rasterizer.rasterizerDiscardEnable = VK_FALSE;  // When not needing to output to a framebuffer
  rasterizer.polygonMode = VK_POLYGON_MODE_FILL;  // Need device feature if using something else than FILL
  rasterizer.lineWidth = 1.0f;
  rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
  rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
  rasterizer.depthBiasEnable = VK_FALSE;          // Set to true to stop shadow acne from shadow mapping

  // Multisampling
  VkPipelineMultisampleStateCreateInfo msaaCreateInfo = {};
  msaaCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  msaaCreateInfo.sampleShadingEnable = VK_FALSE;
  msaaCreateInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;    // MSAA count

  // Blending
  // Blend attachment state
  VkPipelineColorBlendAttachmentState colorState = {};
  colorState.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
                              VK_COLOR_COMPONENT_G_BIT |
                              VK_COLOR_COMPONENT_B_BIT |
                              VK_COLOR_COMPONENT_A_BIT;
  colorState.blendEnable = VK_TRUE;
  // Blending uses equation: (srcColorBlendFactor * new color) colorBlendOp (dstColorBlendFactor * old color)
  colorState.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
  colorState.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
  colorState.colorBlendOp = VK_BLEND_OP_ADD;
  colorState.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
  colorState.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
  colorState.alphaBlendOp = VK_BLEND_OP_ADD;

  // Blend create info
  VkPipelineColorBlendStateCreateInfo blendingCreateInfo = {};
  blendingCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  blendingCreateInfo.logicOpEnable = VK_FALSE;        // Alternative to calculation is to use logical operations
  blendingCreateInfo.logicOp = VK_LOGIC_OP_COPY;
  blendingCreateInfo.attachmentCount = 1;
  blendingCreateInfo.pAttachments = &colorState;

  // Layout
  std::array<VkDescriptorSetLayout, 2> descriptorSetLayouts = { descSetLayout, samplerSetLayout };
  VkPipelineLayoutCreateInfo layoutCreateInfo = {};
  layoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  layoutCreateInfo.setLayoutCount = static_cast<uint32_t>(descriptorSetLayouts.size());
  layoutCreateInfo.pSetLayouts = descriptorSetLayouts.data();
#ifndef USING_PUSH_CONSTANT
  layoutCreateInfo.pushConstantRangeCount = 0;
  layoutCreateInfo.pPushConstantRanges = nullptr;
#else
  layoutCreateInfo.pushConstantRangeCount = 1;
  layoutCreateInfo.pPushConstantRanges = &pushConstantRange;
#endif

  if (vkCreatePipelineLayout(mainDevice.logicalDevice, &layoutCreateInfo, m_pAllocCB, &pipelineLayout) != VK_SUCCESS)
  {
    throw std::runtime_error("Failed to create pipeline layout");
  }
 
  // Depth stencil testing
  VkPipelineDepthStencilStateCreateInfo depthCreateInfo = {};
  depthCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
  depthCreateInfo.depthTestEnable = VK_TRUE;
  depthCreateInfo.depthWriteEnable = VK_TRUE;
  depthCreateInfo.depthCompareOp = VK_COMPARE_OP_LESS;
  depthCreateInfo.depthBoundsTestEnable = VK_FALSE;
  depthCreateInfo.stencilTestEnable = VK_FALSE;

  // Graphics pipeline creation
  VkGraphicsPipelineCreateInfo createInfo = {};
  createInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  createInfo.stageCount = 2;                                // Number of shader stages
  createInfo.pStages = shaderStages;                        // List of shader stages
  createInfo.pVertexInputState = &vertexInputCreateInfo;    // All the fixed function pipeline states
  createInfo.pInputAssemblyState = &inputAssembly;
  createInfo.pViewportState = &viewportCreateInfo;
  createInfo.pDynamicState = nullptr;                       // Set to &dynamicState in uncommented above
  createInfo.pRasterizationState = &rasterizer;
  createInfo.pMultisampleState = &msaaCreateInfo;
  createInfo.pColorBlendState = &blendingCreateInfo;
  createInfo.pDepthStencilState = &depthCreateInfo;
  createInfo.layout = pipelineLayout;
  createInfo.renderPass = renderPass;
  createInfo.subpass = 0;                                   // Subpass of render pass to use with pipeline

  // Pipeline derivatives: Can create multiple pipelines that derive from one another for optimisation
  createInfo.basePipelineHandle = VK_NULL_HANDLE;           // Existing pipeline to derive from ...
  createInfo.basePipelineIndex = -1;                        // or index of pipeline being created to derive from (in case creating multiple at once)

  if (vkCreateGraphicsPipelines(mainDevice.logicalDevice, VK_NULL_HANDLE, 1, &createInfo, m_pAllocCB, &graphicsPipeline) != VK_SUCCESS)
  {
    throw std::runtime_error("Failed to create graphics pipeline");
  }

  // Destroy shader module no longer needed after creating the pipeline
  vkDestroyShaderModule(mainDevice.logicalDevice, fragmentShaderModule, m_pAllocCB);
  vkDestroyShaderModule(mainDevice.logicalDevice, vertexShaderModule, m_pAllocCB);

  // Second pass pipeline
  auto secondVertexShaderCode = readFile("Shaders/second_vert.spv");
  auto secondFragmentShaderCode = readFile("Shaders/second_frag.spv");

  VkShaderModule secondVertexShaderModule = createShaderModule(secondVertexShaderCode);
  VkShaderModule secondFragmentShaderModule = createShaderModule(secondFragmentShaderCode);

  vertexShaderCreateInfo.module = secondVertexShaderModule;
  fragmentShaderCreateInfo.module = secondFragmentShaderModule;

  VkPipelineShaderStageCreateInfo secondShaderStages[] = { vertexShaderCreateInfo, fragmentShaderCreateInfo };

  // No vertex data for second pass
  vertexInputCreateInfo.pVertexBindingDescriptions = nullptr;
  vertexInputCreateInfo.vertexBindingDescriptionCount = 0;
  vertexInputCreateInfo.pVertexAttributeDescriptions = nullptr;
  vertexInputCreateInfo.vertexAttributeDescriptionCount = 0;

  // Don't want to write to depth buffer
  depthCreateInfo.depthWriteEnable = VK_FALSE;
//  depthCreateInfo.depthTestEnable = VK_FALSE;

  VkPipelineLayoutCreateInfo secondPipelineLayoutInfo = {};
  secondPipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  secondPipelineLayoutInfo.setLayoutCount = 1;
  secondPipelineLayoutInfo.pSetLayouts = &inputSetLayout;
  secondPipelineLayoutInfo.pushConstantRangeCount = 0;
  secondPipelineLayoutInfo.pPushConstantRanges = nullptr;

  if (vkCreatePipelineLayout(mainDevice.logicalDevice, &secondPipelineLayoutInfo, m_pAllocCB, &secondPipelineLayout) != VK_SUCCESS)
  {
    throw std::runtime_error("Failed to create second pipeline layout");
  }

  // Pipeline for second pass
  createInfo.pStages = secondShaderStages;
  createInfo.layout = secondPipelineLayout;
  createInfo.subpass = 1;

  if (vkCreateGraphicsPipelines(mainDevice.logicalDevice, VK_NULL_HANDLE, 1, &createInfo, m_pAllocCB, &secondPipeline) != VK_SUCCESS)
  {
    throw std::runtime_error("Failed to create second pipeline");
  }

    // Destroy shader module no longer needed after creating the pipeline
  vkDestroyShaderModule(mainDevice.logicalDevice, secondFragmentShaderModule, m_pAllocCB);
  vkDestroyShaderModule(mainDevice.logicalDevice, secondVertexShaderModule, m_pAllocCB);
}

void VulkanRenderer::createColorBufferImage()
{
  colorBufferImage.resize(swapChainImages.size());
  colorBufferImageMemory.resize(swapChainImages.size());
  colorBufferImageView.resize(swapChainImages.size());

  VkFormat colorFormat = chooseSupportedFormat({ VK_FORMAT_R8G8B8A8_UNORM },
                                               VK_IMAGE_TILING_OPTIMAL,
                                               VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

  for (size_t i = 0; i < swapChainImages.size(); ++i)
  {
    std::tie(colorBufferImage[i], colorBufferImageMemory[i]) = 
      createImage(swapChainExtent.width,
                  swapChainExtent.height,
                  colorFormat,
                  VK_IMAGE_TILING_OPTIMAL,
                  VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT,
                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    colorBufferImageView[i] = createImageView(colorBufferImage[i], colorFormat, VK_IMAGE_ASPECT_COLOR_BIT);
  }
}

void VulkanRenderer::createDepthBuffer()
{
  depthBufferImage.resize(swapChainImages.size());
  depthBufferImageMemory.resize(swapChainImages.size());
  depthBufferImageView.resize(swapChainImages.size());

  VkImageTiling tiling = VK_IMAGE_TILING_OPTIMAL;
  depthBufferFormat = chooseSupportedFormat({ VK_FORMAT_D32_SFLOAT_S8_UINT,
                                              VK_FORMAT_D32_SFLOAT,
                                              VK_FORMAT_D24_UNORM_S8_UINT },
                                            tiling,
                                            VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);

  for (size_t i = 0; i < swapChainImages.size(); ++i)
  {
    std::tie(depthBufferImage[i], depthBufferImageMemory[i]) = 
      createImage(swapChainExtent.width,
                  swapChainExtent.height,
                  depthBufferFormat,
                  tiling,
                  VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT,
                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    depthBufferImageView[i] = createImageView(depthBufferImage[i], depthBufferFormat, VK_IMAGE_ASPECT_DEPTH_BIT);
  }
}

void VulkanRenderer::createFramebuffers()
{
  swapChainFramebuffers.resize(swapChainImages.size());

  for (size_t i = 0; i < swapChainFramebuffers.size(); ++i)
  {
    std::array<VkImageView, 3> attachments =
    {
      swapChainImages[i].imageView,
      colorBufferImageView[i],
      depthBufferImageView[i],
    };

    VkFramebufferCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    createInfo.renderPass = renderPass;
    createInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
    createInfo.pAttachments = attachments.data();
    createInfo.width = swapChainExtent.width;
    createInfo.height = swapChainExtent.height;
    createInfo.layers = 1;

    if (vkCreateFramebuffer(mainDevice.logicalDevice, &createInfo, m_pAllocCB, &swapChainFramebuffers[i]) != VK_SUCCESS)
    {
      throw std::runtime_error("Failed to create framebuffer");
    }
  }
}

void VulkanRenderer::createCommandPool()
{
  QueueFamilyIndices queueFamilyIndices = getQueueFamilies(mainDevice.physicalDevice);

  VkCommandPoolCreateInfo poolInfo = {};
  poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily;

  if (vkCreateCommandPool(mainDevice.logicalDevice, &poolInfo, m_pAllocCB, &graphicsCommandPool) != VK_SUCCESS)
  {
    throw std::runtime_error("Failed to create command pool");
  }
}

void VulkanRenderer::createCommandBuffers()
{
  commandBuffers.resize(swapChainFramebuffers.size());

  VkCommandBufferAllocateInfo cbAllocInfo = {};
  cbAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  cbAllocInfo.commandPool = graphicsCommandPool;
  cbAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;  // VK_COMMAND_BUFFER_LEVEL_PRIMARY  : Buffer you submit directly to queue Can't be called by other buffers
                                                        // VK_COMMAND_BUFFER_LEVEL_SECONDARY: Buffer can't be called directly. Can be called from other buffers via "vkCmdExecuteCommands"
  cbAllocInfo.commandBufferCount = static_cast<uint32_t>(commandBuffers.size());

  if (vkAllocateCommandBuffers(mainDevice.logicalDevice, &cbAllocInfo, commandBuffers.data()) != VK_SUCCESS)
  {
    throw std::runtime_error("Failed to allocate command buffer");
  }
}

void VulkanRenderer::createSynchronization()
{
  imageAvailable.resize(MAX_FRAME_DRAWS);
  renderFinished.resize(MAX_FRAME_DRAWS);
  drawFences.resize(MAX_FRAME_DRAWS);

  // Semaphore creation
  VkSemaphoreCreateInfo semaphoreCreateInfo = {};
  semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

  // Fence creation
  VkFenceCreateInfo fenceCreateInfo = {};
  fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fenceCreateInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;           // Want to start the fence opened

  for (int i = 0; i < MAX_FRAME_DRAWS; ++i)
  {
    if (vkCreateSemaphore(mainDevice.logicalDevice, &semaphoreCreateInfo, m_pAllocCB, &imageAvailable[i]) != VK_SUCCESS ||
        vkCreateSemaphore(mainDevice.logicalDevice, &semaphoreCreateInfo, m_pAllocCB, &renderFinished[i]) != VK_SUCCESS ||
        vkCreateFence(mainDevice.logicalDevice, &fenceCreateInfo, m_pAllocCB, &drawFences[i]) != VK_SUCCESS)
    {
      throw std::runtime_error("Failed to create semaphore");
    }
  }
}

void VulkanRenderer::createTextureSampler()
{
  // Sampler create info
  VkSamplerCreateInfo samplerCreateInfo = {};
  samplerCreateInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  samplerCreateInfo.minLod = 0.0f;
  samplerCreateInfo.maxLod = 0.0f;
  samplerCreateInfo.mipLodBias = 0.0f;
  samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
  samplerCreateInfo.magFilter = VK_FILTER_LINEAR;
  samplerCreateInfo.minFilter = VK_FILTER_LINEAR;
  samplerCreateInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  samplerCreateInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  samplerCreateInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  samplerCreateInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
  samplerCreateInfo.unnormalizedCoordinates = VK_FALSE;
  samplerCreateInfo.anisotropyEnable = VK_TRUE;
  samplerCreateInfo.maxAnisotropy = 8.0f;

  if (vkCreateSampler(mainDevice.logicalDevice, &samplerCreateInfo, m_pAllocCB, &textureSampler) != VK_SUCCESS)
  {
    throw std::runtime_error("Failed to create texture sampler");
  }
}

void VulkanRenderer::createUniformBuffers()
{
  VkDeviceSize vpBufferSize = sizeof(UboViewProjection);

  VkDeviceSize modelBufferSize = modelUniformAlignment * MAX_OBJECTS;

  // One uniform buffer for each image and command buffer
  vpUniformBuffer.resize(swapChainImages.size());
  vpUniformBufferMemory.resize(swapChainImages.size());

#ifndef USING_PUSH_CONSTANT
  modelUniformBufferDynamic.resize(swapChainImages.size());
  modelUniformBufferMemoryDynamic.resize(swapChainImages.size());
#endif

  for (size_t i = 0; i < swapChainImages.size(); ++i)
  {
    createBuffer(mainDevice.physicalDevice,
                 mainDevice.logicalDevice,
                 vpBufferSize,
                 VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 &vpUniformBuffer[i],
                 &vpUniformBufferMemory[i],
                 m_pAllocCB);

#ifndef USING_PUSH_CONSTANT
    createBuffer(mainDevice.physicalDevice,
                 mainDevice.logicalDevice,
                 modelBufferSize,
                 VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 &modelUniformBufferDynamic[i],
                 &modelUniformBufferMemoryDynamic[i],
                 m_pAllocCB);
#endif
  }
}

void VulkanRenderer::createDescriptorPool()
{
  // Uniform descriptor pool
#ifndef USING_PUSH_CONSTANT
  std::array<VkDescriptorPoolSize, 2> poolSizes = {};
#else
  std::array<VkDescriptorPoolSize, 1> poolSizes = {};
#endif

  VkDescriptorPoolSize& vpPoolSize = poolSizes[0];
  vpPoolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  vpPoolSize.descriptorCount = static_cast<uint32_t>(vpUniformBuffer.size());

#ifndef USING_PUSH_CONSTANT
  VkDescriptorPoolSize& modelPoolSize = poolSizes[1];
  modelPoolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
  modelPoolSize.descriptorCount = static_cast<uint32_t>(modelUniformBufferDynamic.size());
#endif
  VkDescriptorPoolCreateInfo poolCreateInfo = {};
  poolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  poolCreateInfo.maxSets = static_cast<uint32_t>(swapChainImages.size());
  poolCreateInfo.poolSizeCount = poolSizes.size();
  poolCreateInfo.pPoolSizes = poolSizes.data();
  poolCreateInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;

  if (vkCreateDescriptorPool(mainDevice.logicalDevice, &poolCreateInfo, m_pAllocCB, &descriptorPool) != VK_SUCCESS)
  {
    throw std::runtime_error("Failed to create descriptor pool");
  }

  // Sampler descriptor pool
  VkDescriptorPoolSize samplerPoolSize = {};
  samplerPoolSize.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  samplerPoolSize.descriptorCount = MAX_OBJECTS;

  VkDescriptorPoolCreateInfo samplerPoolCreateInfo = {};
  samplerPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  samplerPoolCreateInfo.maxSets = MAX_OBJECTS;
  samplerPoolCreateInfo.poolSizeCount = 1;
  samplerPoolCreateInfo.pPoolSizes = &samplerPoolSize;

  if (vkCreateDescriptorPool(mainDevice.logicalDevice, &samplerPoolCreateInfo, m_pAllocCB, &samplerDescriptorPool) != VK_SUCCESS)
  {
    throw std::runtime_error("Failed to create sampler descriptor pool");
  }

  // Input attachment descriptor pool
  std::array<VkDescriptorPoolSize, 2> inputPoolSize = {};
  VkDescriptorPoolSize& colorInputPoolSize = inputPoolSize[0];
  colorInputPoolSize.type = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
  colorInputPoolSize.descriptorCount = colorBufferImageView.size();

  VkDescriptorPoolSize& depthInputPoolSize = inputPoolSize[1];
  depthInputPoolSize.type = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
  depthInputPoolSize.descriptorCount = depthBufferImageView.size();

  VkDescriptorPoolCreateInfo inputPoolCreateInfo = {};
  inputPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  inputPoolCreateInfo.maxSets = swapChainImages.size();
  inputPoolCreateInfo.poolSizeCount = inputPoolSize.size();
  inputPoolCreateInfo.pPoolSizes = inputPoolSize.data();

  if (vkCreateDescriptorPool(mainDevice.logicalDevice, &inputPoolCreateInfo, m_pAllocCB, &inputDescriptorPool) != VK_SUCCESS)
  {
    throw std::runtime_error("Failed to create input descriptor pool");
  }
}

void VulkanRenderer::createDescriptorSets()
{
  descriptorSets.resize(swapChainImages.size());

  std::vector<VkDescriptorSetLayout> setLayouts(swapChainImages.size(), descSetLayout);

  VkDescriptorSetAllocateInfo setAllocInfo = {};
  setAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  setAllocInfo.descriptorPool = descriptorPool;
  setAllocInfo.descriptorSetCount = static_cast<uint32_t>(setLayouts.size());
  setAllocInfo.pSetLayouts = setLayouts.data();

  if (vkAllocateDescriptorSets(mainDevice.logicalDevice, &setAllocInfo, descriptorSets.data()) != VK_SUCCESS)
  {
    throw std::runtime_error("Failed to allocate descriptor sets");
  }

  for (size_t i = 0; i < swapChainImages.size(); ++i)
  {
#ifndef USING_PUSH_CONSTANT
    std::array<VkWriteDescriptorSet, 2> setWrites = {};
#else
    std::array<VkWriteDescriptorSet, 1> setWrites = {};
#endif
    // ViewProjection
    VkDescriptorBufferInfo vpBufferInfo = {};
    vpBufferInfo.buffer = vpUniformBuffer[i];
    vpBufferInfo.offset = 0;
    vpBufferInfo.range = sizeof(UboViewProjection);

    VkWriteDescriptorSet& vpSetWrite = setWrites[0];
    vpSetWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    vpSetWrite.dstSet = descriptorSets[i];       // Descriptor set to update
    vpSetWrite.dstBinding = 0;                   // Must match binding in shader
    vpSetWrite.dstArrayElement = 0;
    vpSetWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    vpSetWrite.descriptorCount = 1;
    vpSetWrite.pBufferInfo = &vpBufferInfo;

#ifndef USING_PUSH_CONSTANT
    // Model
    VkDescriptorBufferInfo modelBufferInfo = {};
    modelBufferInfo.buffer = modelUniformBufferDynamic[i];
    modelBufferInfo.offset = 0;
    modelBufferInfo.range = modelUniformAlignment;

    VkWriteDescriptorSet& modelSetWrite = setWrites[1];
    modelSetWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    modelSetWrite.dstSet = descriptorSets[i];       // Descriptor set to update
    modelSetWrite.dstBinding = 1;                   // Must match binding in shader
    modelSetWrite.dstArrayElement = 0;
    modelSetWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
    modelSetWrite.descriptorCount = 1;
    modelSetWrite.pBufferInfo = &modelBufferInfo;
#endif
    vkUpdateDescriptorSets(mainDevice.logicalDevice, setWrites.size(), setWrites.data(), 0, nullptr);
  }
}

void VulkanRenderer::createInputDescriptorSets()
{
  inputDescriptorSets.resize(swapChainImages.size());

  std::vector<VkDescriptorSetLayout> setLayouts(swapChainImages.size(), inputSetLayout);

  VkDescriptorSetAllocateInfo setAllocInfo = {};
  setAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  setAllocInfo.descriptorPool = inputDescriptorPool;
  setAllocInfo.descriptorSetCount = static_cast<uint32_t>(setLayouts.size());
  setAllocInfo.pSetLayouts = setLayouts.data();

  if (vkAllocateDescriptorSets(mainDevice.logicalDevice, &setAllocInfo, inputDescriptorSets.data()) != VK_SUCCESS)
  {
    throw std::runtime_error("Failed to allocate input descriptor sets");
  }

  for (size_t i = 0; i < swapChainImages.size(); ++i)
  {
    std::array<VkWriteDescriptorSet, 2> setWrites = {};

    // Color attachment write
    VkDescriptorImageInfo colorAttachmentDesc = {};
    colorAttachmentDesc.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    colorAttachmentDesc.imageView = colorBufferImageView[i];
    colorAttachmentDesc.sampler = VK_NULL_HANDLE;

    VkWriteDescriptorSet& colorWrite = setWrites[0];
    colorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    colorWrite.dstSet = inputDescriptorSets[i];
    colorWrite.dstBinding = 0;
    colorWrite.descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
    colorWrite.descriptorCount = 1;
    colorWrite.pImageInfo = &colorAttachmentDesc;

    // Depth attachment write
    VkDescriptorImageInfo depthAttachmentDesc = {};
    depthAttachmentDesc.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    depthAttachmentDesc.imageView = depthBufferImageView[i];
    depthAttachmentDesc.sampler = VK_NULL_HANDLE;

    VkWriteDescriptorSet& depthWrite = setWrites[1];
    depthWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    depthWrite.dstSet = inputDescriptorSets[i];
    depthWrite.dstBinding = 1;
    depthWrite.descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
    depthWrite.descriptorCount = 1;
    depthWrite.pImageInfo = &depthAttachmentDesc;

    vkUpdateDescriptorSets(mainDevice.logicalDevice, setWrites.size(), setWrites.data(), 0, nullptr);
  }
}

void VulkanRenderer::updateUniformBuffers(uint32_t imageIndex)
{
  // Copy VP data
  void* data;
  vkMapMemory(mainDevice.logicalDevice, vpUniformBufferMemory[imageIndex], 0, sizeof(UboViewProjection), 0, &data);
  memcpy(data, &uboViewProjection, sizeof(UboViewProjection));
  vkUnmapMemory(mainDevice.logicalDevice, vpUniformBufferMemory[imageIndex]);

#ifndef USING_PUSH_CONSTANT
  // Copy Model data
  int numMeshes = 0;
  for (size_t j = 0; j < modelList.size(); ++j)
  {
    for (size_t i = 0; i < modelList[j]->getMeshCount(); ++i)
    {
      Model* thisModel = (Model*)((uint64_t)modelTransferSpace + (i * modelUniformAlignment));
      *thisModel = modelList[j]->getMesh(i)->getModel();
    }
    numMeshes += modelList[j]->getMeshCount();
  }
  vkMapMemory(mainDevice.logicalDevice,
              modelUniformBufferMemoryDynamic[imageIndex],
              0,
              modelUniformAlignment * numMeshes,
              0,
              &data);
  memcpy(data, modelTransferSpace, modelUniformAlignment * numMeshes);
  vkUnmapMemory(mainDevice.logicalDevice, modelUniformBufferMemoryDynamic[imageIndex]);
#endif
}

void VulkanRenderer::recordCommands(uint32_t currentImage)
{
  VkCommandBufferBeginInfo beginInfo = {};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
//  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;     // Buffer can be resubmitted when it has already been submitted and waiting execution

  VkRenderPassBeginInfo renderBeginInfo = {};
  renderBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  renderBeginInfo.renderPass = renderPass;                            // Render pass to begin
  renderBeginInfo.renderArea.offset = { 0, 0 };                       // Start point
  renderBeginInfo.renderArea.extent = swapChainExtent;                // Region size

  std::array<VkClearValue, 3> clearValues = {};
  clearValues[0].color = { 0.0f, 0.0f, 0.0f, 1.0f };
  clearValues[1].color = { 0.6f, 0.65f, 0.4f, 1.0f };
  clearValues[2].depthStencil.depth = 1.0f;

  renderBeginInfo.pClearValues = clearValues.data();
  renderBeginInfo.clearValueCount = clearValues.size();

  renderBeginInfo.framebuffer = swapChainFramebuffers[currentImage];
  auto& commandBuffer = commandBuffers[currentImage];

  // Start recording commands to command buffer
  if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS)
  {
    throw std::runtime_error("Failed to begin command buffer");
  }

  // Begin render pass
  vkCmdBeginRenderPass(commandBuffer, &renderBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

  // Bind pipeline to use
  vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

  for (size_t j = 0; j < modelList.size(); ++j)
  {
    MeshModel* meshModel = modelList[j];

    uint32_t* dynamicOffsetPtr = nullptr;
#ifndef USING_PUSH_CONSTANT
    // Dynamic offset amount
    uint32_t dynamicOffset = static_cast<uint32_t>(modelUniformAlignment) * j;
    dynamicOffsetPtr = &dynamicOffset;
#else
    vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(Model), &meshModel->getModelMatrix());
#endif

    for (size_t k = 0; k < meshModel->getMeshCount(); ++k)
    {
      auto* mesh = meshModel->getMesh(k);

      VkBuffer vertexBuffers[] = { mesh->getVertexBuffer() };
      VkDeviceSize offsets[] = { 0 };
      vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);

      if (mesh->getIndexCount() > 0)
      {
        // Bind mesh index buffer
        vkCmdBindIndexBuffer(commandBuffer, mesh->getIndexBuffer(), 0, VK_INDEX_TYPE_UINT32);

        std::array<VkDescriptorSet, 2> descriptorSetGroup =
        {
          descriptorSets[currentImage],
          samplerDescriptorSets[mesh->getTexId()]
        };

        // Bind descriptor sets for uniform buffers
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0,
                                descriptorSetGroup.size(), descriptorSetGroup.data(), 0, dynamicOffsetPtr);

        // Execute pipeline
        vkCmdDrawIndexed(commandBuffer, mesh->getIndexCount(), 1, 0, 0, 0);
      }
      else
      {
        // Execute pipeline
        vkCmdDraw(commandBuffer, mesh->getVertexCount(), 1, 0, 0);
      }
    }
  }

  // Start second subpass
  vkCmdNextSubpass(commandBuffer, VK_SUBPASS_CONTENTS_INLINE);

  vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, secondPipeline);
  vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, secondPipelineLayout, 0,
                          1, &inputDescriptorSets[currentImage], 0, nullptr);

  // Draw fullscreen triangle
  vkCmdDraw(commandBuffer, 3, 1, 0, 0);

  // End render pass
  vkCmdEndRenderPass(commandBuffer);

  // Stop recording to command buffer
  if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS)
  {
    throw std::runtime_error("Failed to end command buffer");
  }
}

void VulkanRenderer::getPhysicalDevice()
{
  // Enumerate physical devices the vkInstance can access
  uint32_t deviceCount = 0;
  vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

  // If no devices available, then none support Vulkan
  if (deviceCount == 0)
  {
    throw std::runtime_error("Can't find GPUs that support Vulkan Instance!");
  }

  // Get list of physical devices
  std::vector<VkPhysicalDevice> deviceList(deviceCount);
  vkEnumeratePhysicalDevices(instance, &deviceCount, deviceList.data());

  for (const auto& device : deviceList)
  {
    if (checkSuitableDevice(device))
    {
      mainDevice.physicalDevice = device;
      break;
    }
  }

  // Information about the device itself (ID, name, type, vendor, etc)
  VkPhysicalDeviceProperties deviceProperties;
  vkGetPhysicalDeviceProperties(mainDevice.physicalDevice, &deviceProperties);

  minUniformBufferOffset = deviceProperties.limits.minUniformBufferOffsetAlignment;
}

void VulkanRenderer::allocateDynamicBufferTransferSpace()
{
#ifndef USING_PUSH_CONSTANT
  modelUniformAlignment = static_cast<size_t>((sizeof(Model) + minUniformBufferOffset - 1) & (~(minUniformBufferOffset - 1)));

  // Create space in memory to hold dynamic buffer that is aligned to our required alignment
  modelTransferSpace = reinterpret_cast<Model*>(_aligned_malloc(modelUniformAlignment * MAX_OBJECTS, modelUniformAlignment));
#endif
}

bool VulkanRenderer::checkInstanceExtensionSupport(const std::vector<const char*>& a_rExtensions)
{
  uint32_t extensionCount = 0;
  if (vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr) != VK_SUCCESS)
  {
    return false;
  }

  std::vector<VkExtensionProperties> extensions(extensionCount);
  vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());

  bool bOk = true;
  for (const auto& checkExtension : a_rExtensions)
  {
    bool bHasExtension = false;
    for (const auto& extension : extensions)
    {
      if (strcmp(extension.extensionName, checkExtension) == 0)
      {
        bHasExtension = true;
        break;
      }
    }

    if (!bHasExtension)
    {
      printf("Extension not supported: %s\n", checkExtension);
      bOk = false;
    }
  }

  return bOk;
}

bool VulkanRenderer::checkDeviceExtensionSupport(VkPhysicalDevice device)
{
  uint32_t extensionCount = 0;
  vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

  if (extensionCount == 0)
  {
    return false;
  }

  std::vector<VkExtensionProperties> extensions(extensionCount);
  vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, extensions.data());

  for (const auto& deviceExtension : deviceExtensions)
  {
    bool hasExtension = false;
    for (const auto& extension : extensions)
    {
      if (strcmp(deviceExtension, extension.extensionName) == 0)
      {
        hasExtension = true;
        break;
      }
    }

    if (!hasExtension)
    {
      return false;
    }
  }

  return true;
}

bool VulkanRenderer::checkValidationLayerSupport()
{
  if (m_bValidationLayers)
  {
    uint32_t layerCount;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

    std::vector<VkLayerProperties> availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

    for (const char* layerName : validationLayers)
    {
      bool layerFound = false;

      for (const auto& layerProperties : availableLayers)
      {
        if (strcmp(layerName, layerProperties.layerName) == 0)
        {
          layerFound = true;
          break;
        }
      }

      if (!layerFound)
      {
        return false;
      }
    }
  }

  return true;
}

std::vector<const char*> VulkanRenderer::getRequiredExtensions()
{
  uint32_t glfwExtensionCount = 0;
  const char** glfwExtensions;
  glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

  std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

  if (m_bValidationLayers)
  {
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  }

  return extensions;
}

bool VulkanRenderer::checkSuitableDevice(VkPhysicalDevice device)
{
  // Information about what the device can do (geo shader, tess shader, wide lines, etc)
  VkPhysicalDeviceFeatures deviceFeatures;
  vkGetPhysicalDeviceFeatures(device, &deviceFeatures);

  QueueFamilyIndices indices = getQueueFamilies(device);

  bool swapChainValid = false;
  if (checkDeviceExtensionSupport(device))
  {
    SwapChainDetails swapChainDetails = getSwapChainDetails(device);
    swapChainValid = !swapChainDetails.presentationModes.empty() &&
                     !swapChainDetails.formats.empty();
  }

  samplerAnisotropySupported = deviceFeatures.samplerAnisotropy == VK_TRUE;

  return indices.isValid() && swapChainValid;
}

QueueFamilyIndices VulkanRenderer::getQueueFamilies(VkPhysicalDevice device)
{
  QueueFamilyIndices indices;

  uint32_t queueFamilyCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

  std::vector<VkQueueFamilyProperties> queueFamilyList(queueFamilyCount);
  vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilyList.data());

  // Go through each queue family and check if it has at least 1 of the required types of queue
  int i = 0;
  for (const auto& queueFamily : queueFamilyList)
  {
    // Check if queue family has at least 1 queue in that family
    // Queue can be multiple type defined through bitfield.
    if (queueFamily.queueCount > 0 && (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0)
    {
      indices.graphicsFamily = i;
    }

    VkBool32 presentationSupport = false;
    vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentationSupport);
    if (queueFamily.queueCount > 0 && presentationSupport)
    {
      indices.presentationFamily = i;
    }

    if (indices.isValid())
    {
      break;
    }

    i++;
  }

  return indices;
}

SwapChainDetails VulkanRenderer::getSwapChainDetails(VkPhysicalDevice device)
{
  SwapChainDetails swapChainDetails;
  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &swapChainDetails.surfaceCapabilities);

  uint32_t formatCount = 0;
  vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);
  if (formatCount != 0)
  {
    swapChainDetails.formats.resize(formatCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, swapChainDetails.formats.data());
  }

  uint32_t presentationCount = 0;
  vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentationCount, nullptr);
  if (presentationCount != 0)
  {
    swapChainDetails.presentationModes.resize(presentationCount);
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentationCount, swapChainDetails.presentationModes.data());
  }

  return swapChainDetails;
}

VkSurfaceFormatKHR VulkanRenderer::chooseBestSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& formats)
{
  // If only 1 format available and it's undefined, it means all formats are supported and
  // Vulkan didn't want to list all of them.
  if (formats.size() == 1 && formats[0].format == VK_FORMAT_UNDEFINED)
  {
    return { VK_FORMAT_R8G8B8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR };
  }

  for (const auto& format : formats)
  {
    if ((format.format == VK_FORMAT_R8G8B8A8_UNORM ||
         format.format == VK_FORMAT_B8G8R8A8_UNORM) &&
        format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
    {
      return format;
    }
  }

  return formats[0];
}

VkPresentModeKHR VulkanRenderer::chooseBestPresentationMode(const std::vector<VkPresentModeKHR>& presentationModes)
{
  for (const auto& presentationMode : presentationModes)
  {
    if (presentationMode == VK_PRESENT_MODE_MAILBOX_KHR)
    {
      return presentationMode;
    }
  }

  // Vulkan states this mode must be supported
  return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D VulkanRenderer::chooseSwapExtent(const VkSurfaceCapabilitiesKHR& surfaceCapabilities)
{
  if (surfaceCapabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
  {
    return surfaceCapabilities.currentExtent;
  }
  else
  {
    // If value can vary, need to set manually
    int width;
    int height;
    glfwGetFramebufferSize(m_pWindow, &width, &height);

    VkExtent2D newExtent;
    newExtent.width = static_cast<uint32_t>(width);
    newExtent.height = static_cast<uint32_t>(height);

    newExtent.width = std::max(surfaceCapabilities.minImageExtent.width,
                               std::min(surfaceCapabilities.maxImageExtent.width,
                                        newExtent.width));
    newExtent.height = std::max(surfaceCapabilities.minImageExtent.height,
                                std::min(surfaceCapabilities.maxImageExtent.height,
                                         newExtent.height));
    return newExtent;
  }
}

VkFormat VulkanRenderer::chooseSupportedFormat(const std::vector<VkFormat>& formats, VkImageTiling tiling, VkFormatFeatureFlags featureFlags)
{
  for (auto format : formats)
  {
    VkFormatProperties properties;
    vkGetPhysicalDeviceFormatProperties(mainDevice.physicalDevice, format, &properties);

    if ((tiling == VK_IMAGE_TILING_LINEAR  && (properties.linearTilingFeatures  & featureFlags) == featureFlags) ||
        (tiling == VK_IMAGE_TILING_OPTIMAL && (properties.optimalTilingFeatures & featureFlags) == featureFlags))
    {
      return format;
    }
  }

  throw std::runtime_error("Failed to fin a matchin format!");
  return VK_FORMAT_UNDEFINED;
}

std::tuple<VkImage, VkDeviceMemory> VulkanRenderer::createImage(uint32_t width,
                                                                uint32_t height,
                                                                VkFormat format,
                                                                VkImageTiling tiling,
                                                                VkImageUsageFlags useFlags,
                                                                VkMemoryPropertyFlags propFlags)
{
  if (width == 0 || height == 0)
  {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory deviceMemory = VK_NULL_HANDLE;
    return std::make_tuple(image, deviceMemory);
  }

  // Create image
  VkImageCreateInfo imageCreateInfo = {};
  imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
  imageCreateInfo.extent.width = width;
  imageCreateInfo.extent.height = height;
  imageCreateInfo.extent.depth = 1;
  imageCreateInfo.mipLevels = 1;
  imageCreateInfo.arrayLayers = 1;
  imageCreateInfo.format = format;
  imageCreateInfo.tiling = tiling;
  imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  imageCreateInfo.usage = useFlags;
  imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
  imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VkImage image;
  if (vkCreateImage(mainDevice.logicalDevice, &imageCreateInfo, m_pAllocCB, &image) != VK_SUCCESS)
  {
    throw std::runtime_error("Failed to create image");
  }

  // Allocate device memory
  VkMemoryRequirements memoryReqs = {};
  vkGetImageMemoryRequirements(mainDevice.logicalDevice, image, &memoryReqs);

  VkDeviceMemory deviceMemory;
  VkMemoryAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = memoryReqs.size;
  allocInfo.memoryTypeIndex = findMemoryTypeIndex(mainDevice.physicalDevice, memoryReqs.memoryTypeBits, propFlags);
  if (vkAllocateMemory(mainDevice.logicalDevice, &allocInfo, m_pAllocCB, &deviceMemory) != VK_SUCCESS)
  {
    throw std::runtime_error("Failed to allocate image device memory");
  }

  // Connect device memory to image
  vkBindImageMemory(mainDevice.logicalDevice, image, deviceMemory, 0);

  return std::make_tuple(image, deviceMemory);
}

VkImageView VulkanRenderer::createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags)
{
  VkImageViewCreateInfo createInfo = {};
  createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  createInfo.image = image;
  createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
  createInfo.format = format;
  createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;      // Allows remapping of the rgba components
  createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
  createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
  createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

  // Subresources
  createInfo.subresourceRange.aspectMask = aspectFlags;         // COLOR_BIT for viewing color
  createInfo.subresourceRange.baseMipLevel = 0;                 // Start mipmap level
  createInfo.subresourceRange.levelCount = 1;                   // Number of mipmap levels to view
  createInfo.subresourceRange.baseArrayLayer = 0;               // Texture array index
  createInfo.subresourceRange.layerCount = 1;

  VkImageView imageView;
  if (vkCreateImageView(mainDevice.logicalDevice, &createInfo, m_pAllocCB, &imageView) != VK_SUCCESS)
  {
    throw std::runtime_error("Unable to create the image view");
  }
  return imageView;
}

VkShaderModule VulkanRenderer::createShaderModule(const std::vector<char>& code)
{
  VkShaderModuleCreateInfo createInfo = {};
  createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  createInfo.codeSize = code.size();
  createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

  VkShaderModule shaderModule;
  vkCreateShaderModule(mainDevice.logicalDevice, &createInfo, m_pAllocCB, &shaderModule);
  return shaderModule;
}

int VulkanRenderer::createTextureImage(const std::string& filename)
{
  int width, height;
  VkDeviceSize imageSize;
  stbi_uc* imageData = loadTextureFile(filename, width, height, imageSize);

  VkBuffer imageStageBuffer;
  VkDeviceMemory imageStageBufferMemory;
  createBuffer(mainDevice.physicalDevice,
               mainDevice.logicalDevice,
               imageSize,
               VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
               &imageStageBuffer,
               &imageStageBufferMemory,
               m_pAllocCB);

  void* data;
  vkMapMemory(mainDevice.logicalDevice, imageStageBufferMemory, 0, imageSize, 0, &data);
  memcpy(data, imageData, static_cast<size_t>(imageSize));
  vkUnmapMemory(mainDevice.logicalDevice, imageStageBufferMemory);

  stbi_image_free(imageData);
  imageData = nullptr;

  VkImage texImage;
  VkDeviceMemory texImageMemory;
  std::tie(texImage, texImageMemory) = createImage(width,
                                                   height,
                                                   VK_FORMAT_R8G8B8A8_UNORM,
                                                   VK_IMAGE_TILING_OPTIMAL,
                                                   VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                                                   VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  // Transition image to be DST for copy operation
  transitionImageLayout(mainDevice.logicalDevice, graphicsQueue, graphicsCommandPool, texImage, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

  // Copy image data
  copyImageBuffer(mainDevice.logicalDevice, graphicsQueue, graphicsCommandPool, imageStageBuffer, texImage, width, height);

  transitionImageLayout(mainDevice.logicalDevice, graphicsQueue, graphicsCommandPool, texImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

  textureImages.push_back(texImage);
  textureImageMemory.push_back(texImageMemory);

  // Destroy staging buffers
  vkDestroyBuffer(mainDevice.logicalDevice, imageStageBuffer, m_pAllocCB);
  vkFreeMemory(mainDevice.logicalDevice, imageStageBufferMemory, m_pAllocCB);

  return textureImages.size() - 1;
}

int VulkanRenderer::createTexture(const std::string& filename)
{
  int textureImageLoc = createTextureImage(filename);

  VkImageView imageView = createImageView(textureImages[textureImageLoc], VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT);
  textureImageViews.push_back(imageView);

  // Create descriptor set
  return createTextureDescriptor(imageView);
}

int VulkanRenderer::createTextureDescriptor(VkImageView textureImage)
{
  VkDescriptorSet descriptorSet;
  
  VkDescriptorSetAllocateInfo descriptorAllocateInfo = {};
  descriptorAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  descriptorAllocateInfo.descriptorPool = samplerDescriptorPool;
  descriptorAllocateInfo.descriptorSetCount = 1;
  descriptorAllocateInfo.pSetLayouts = &samplerSetLayout;

  if (vkAllocateDescriptorSets(mainDevice.logicalDevice, &descriptorAllocateInfo, &descriptorSet))
  {
    throw std::runtime_error("Failed to allocate texture descriptor set");
  }

  VkDescriptorImageInfo imageInfo = {};
  imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;     // layout when in use
  imageInfo.imageView = textureImage;
  imageInfo.sampler = textureSampler;

  VkWriteDescriptorSet descriptorWrite = {};
  descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  descriptorWrite.dstSet = descriptorSet;
  descriptorWrite.dstBinding = 0;
  descriptorWrite.dstArrayElement = 0;
  descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  descriptorWrite.descriptorCount = 1;
  descriptorWrite.pImageInfo = &imageInfo;

  vkUpdateDescriptorSets(mainDevice.logicalDevice, 1, &descriptorWrite, 0, nullptr);

  samplerDescriptorSets.push_back(descriptorSet);

  return samplerDescriptorSets.size() - 1;
}

int VulkanRenderer::createMeshModel(const std::string& modelFile)
{
  // Import model "scene"
  Assimp::Importer importer;
  const aiScene* scene = importer.ReadFile(modelFile, aiProcess_Triangulate | aiProcess_FlipUVs | aiProcess_JoinIdenticalVertices);
  if (!scene)
  {
    throw std::runtime_error("Failed top load model " + modelFile);
  }

  std::vector<std::string> textureNames = MeshModel::LoadMaterials(scene);
  std::vector<int> matToTex(textureNames.size());
  for (size_t i = 0; i < textureNames.size(); ++i)
  {
    // If material had no texture, set '0' to indicated no texture, texture 0 will be reserved for default exture
    if (textureNames[i].empty())
    {
      matToTex[i] = 0;
    }
    else
    {
      matToTex[i] = createTexture(textureNames[i]);
    }
  }

  std::vector<Mesh*> modelMeshes = MeshModel::LoadNode(mainDevice.physicalDevice, mainDevice.logicalDevice, graphicsQueue,
                                                       graphicsCommandPool, scene->mRootNode, scene, matToTex, m_pAllocCB);
  modelList.push_back(new MeshModel(modelMeshes));
  return modelList.size() - 1;
}

stbi_uc* VulkanRenderer::loadTextureFile(const std::string& filename, int& width, int& height, VkDeviceSize& imageSize)
{
  if (filename.length() < 1)
  {
    return nullptr;
  }

  std::string fileLoc = "Textures/" + filename;
  int channels;
  stbi_uc* image = stbi_load(fileLoc.c_str(), &width, &height, &channels, STBI_rgb_alpha);
  if (!image)
  {
    throw std::runtime_error("Failed to load texture file " + filename);
  }

  imageSize = width * height * 4;

  return image;
}
