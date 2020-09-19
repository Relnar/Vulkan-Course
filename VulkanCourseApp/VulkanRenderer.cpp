#include "VulkanRenderer.h"
#include <iostream>
#include <array>

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
    createRenderPass();
    createGraphicsPipeline();
    createFramebuffers();
    createCommandPool();

    std::vector<Vertex> meshVertices =
    {
      {{-0.1, -0.4, 0.0}, {1.0, 0.0, 0.0}},    // 0
      {{-0.1, 0.4, 0.0}, {0.0, 1.0, 0.0}},     // 1
      {{-0.9, 0.4, 0.0}, {0.0, 0.0, 1.0}},    // 2
      {{-0.9, -0.4, 0.0}, {1.0, 1.0, 0.0}},   // 3
    };
    std::vector<Vertex> meshVertices2 =
    {
      {{0.9, -0.4, 0.0}, {1.0, 0.0, 0.0}},    // 0
      {{0.9, 0.4, 0.0}, {0.0, 1.0, 0.0}},     // 1
      {{0.1, 0.4, 0.0}, {0.0, 0.0, 1.0}},    // 2
      {{0.1, -0.4, 0.0}, {1.0, 1.0, 0.0}},   // 3
    };
    std::vector<uint32_t> meshIndices =
    {
      0, 1, 2, 2, 3 ,0
    };
    meshList.push_back(new Mesh(mainDevice.physicalDevice, mainDevice.logicalDevice, graphicsQueue, graphicsCommandPool, meshVertices, meshIndices, m_pAllocCB));
    meshList.push_back(new Mesh(mainDevice.physicalDevice, mainDevice.logicalDevice, graphicsQueue, graphicsCommandPool, meshVertices2, meshIndices, m_pAllocCB));

    createCommandBuffers();
    createSynchronization();
    recordCommands();
  }
  catch (const std::runtime_error& e)
  {
    printf("Error: %s\n", e.what());
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
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

  for (auto mesh : meshList)
  {
    delete mesh;
  }
  meshList.clear();

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
  // Color attachment of the render pass
  VkAttachmentDescription colorAttachment = {};
  colorAttachment.format = swapChainImageFormat;
  colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;                    // MSAA count
  colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;               // Clear color before rendering
  colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;             // What to do after rendering
  colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

  // Framebuffer data will be stored as an image, but images can be given different data layouts
  colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;          // Image data layout before render pass starts
  colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;      // Image data layout after render pass (to change to)

  // Attachment reference uses an attachment index that refers to index in the attachment list passed to renderPassCreate
  VkAttachmentReference colorAttachmentRef = {};
  colorAttachmentRef.attachment = 0;
  colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

  // Information about a particular subpass the render pass is using
  VkSubpassDescription subpass = {};
  subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass.colorAttachmentCount = 1;
  subpass.pColorAttachments = &colorAttachmentRef;

  // Need to determine when layout transitions occur using subpass dependencies
  std::array<VkSubpassDependency, 2> subpassDependencies;

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

  // Conversion from VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL to VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
  subpassDependencies[1].srcSubpass = 0;
  subpassDependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  subpassDependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  subpassDependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
  subpassDependencies[1].dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
  subpassDependencies[1].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
  subpassDependencies[1].dependencyFlags = 0;

  VkRenderPassCreateInfo renderPassCreate = {};
  renderPassCreate.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  renderPassCreate.attachmentCount = 1;
  renderPassCreate.pAttachments = &colorAttachment;
  renderPassCreate.subpassCount = 1;
  renderPassCreate.pSubpasses = &subpass;
  renderPassCreate.dependencyCount = static_cast<uint32_t>(subpassDependencies.size());
  renderPassCreate.pDependencies = subpassDependencies.data();

  if (vkCreateRenderPass(mainDevice.logicalDevice, &renderPassCreate, m_pAllocCB, &renderPass) != VK_SUCCESS)
  {
    throw std::runtime_error("Failed to create render pass");
  }
}

void VulkanRenderer::createGraphicsPipeline()
{
  // Read shader files
  auto vertexShader = readFile("Shaders/vert.spv");
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

  std::array<VkVertexInputAttributeDescription, 2> attribDescs;

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
  rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
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
  VkPipelineLayoutCreateInfo layoutCreateInfo = {};
  layoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  layoutCreateInfo.setLayoutCount = 0;
  layoutCreateInfo. pSetLayouts = nullptr;
  layoutCreateInfo.pushConstantRangeCount = 0;
  layoutCreateInfo.pPushConstantRanges = nullptr;

  if (vkCreatePipelineLayout(mainDevice.logicalDevice, &layoutCreateInfo, m_pAllocCB, &pipelineLayout) != VK_SUCCESS)
  {
    throw std::runtime_error("Failed to create pipeline layout");
  }
 
  // Depth stencil testing
  // TODO: setup depth stencil testing


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
  createInfo.pDepthStencilState = nullptr;
  createInfo.layout = pipelineLayout;
  createInfo.renderPass = renderPass;
  createInfo.subpass = 0;                                   // Subpass of render pass to use with pipeline

  // Pipeline derivatives: Can create multiple pipelines that derive from one another for optimisation
  createInfo.basePipelineHandle = VK_NULL_HANDLE;           // Existing pipeline to derive from ...
  createInfo.basePipelineIndex = -1;                        // or index of pipeline being created to derive from (in case creating multiple at once)

  if (vkCreateGraphicsPipelines(mainDevice.logicalDevice, VK_NULL_HANDLE, 1, &createInfo, m_pAllocCB, &graphicsPipeline) != VK_SUCCESS)
  {
    throw std::runtime_error("Failed to create graphics pipelines");
  }

  // Destroy shader module no longer needed after creating the pipeline
  vkDestroyShaderModule(mainDevice.logicalDevice, fragmentShaderModule, m_pAllocCB);
  vkDestroyShaderModule(mainDevice.logicalDevice, vertexShaderModule, m_pAllocCB);
}

void VulkanRenderer::createFramebuffers()
{
  swapChainFramebuffers.resize(swapChainImages.size());

  for (size_t i = 0; i < swapChainFramebuffers.size(); ++i)
  {
    std::array<VkImageView, 1> attachments =
    {
      swapChainImages[i].imageView,
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

void VulkanRenderer::recordCommands()
{
  VkCommandBufferBeginInfo beginInfo = {};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
//  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;     // Buffer can be resubmitted when it has already been submitted and waiting execution

  VkRenderPassBeginInfo renderBeginInfo = {};
  renderBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  renderBeginInfo.renderPass = renderPass;                            // Render pass to begin
  renderBeginInfo.renderArea.offset = { 0, 0 };                       // Start point
  renderBeginInfo.renderArea.extent = swapChainExtent;                // Region size
  VkClearValue clearValues[] =
  {
    { 0.6f, 0.65f, 0.4f, 1.0f },
  };
  renderBeginInfo.pClearValues = clearValues;
  renderBeginInfo.clearValueCount = 1;

  for (size_t i = 0; i < commandBuffers.size(); ++i)
  {
    renderBeginInfo.framebuffer = swapChainFramebuffers[i];
    auto& commandBuffer = commandBuffers[i];

    // Start recording commands to command buffer
    if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS)
    {
      throw std::runtime_error("Failed to begin command buffer");
    }

    // Begin render pass
    vkCmdBeginRenderPass(commandBuffer, &renderBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

    // Bind pipeline to use
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

    for (auto mesh : meshList)
    {
      VkBuffer vertexBuffers[] = { mesh->getVertexBuffer() };
      VkDeviceSize offsets[] = { 0 };
      vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);

      if (mesh->getIndexCount() > 0)
      {
        vkCmdBindIndexBuffer(commandBuffer, mesh->getIndexBuffer(), 0, VK_INDEX_TYPE_UINT32);

        // Execute pipeline
        vkCmdDrawIndexed(commandBuffer, mesh->getIndexCount(), 1, 0, 0, 0);
      }
      else
      {
        // Execute pipeline
        vkCmdDraw(commandBuffer, mesh->getVertexCount(), 1, 0, 0);
      }
    }

    // End render pass
    vkCmdEndRenderPass(commandBuffer);

    // Stop recording to command buffer
    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS)
    {
      throw std::runtime_error("Failed to end command buffer");
    }
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
/*
  // Information about the device itself (ID, name, type, vendor, etc)
  VkPhysicalDeviceProperties deviceProperties;
  vkGetPhysicalDeviceProperties(device, &deviceProperties);

  // Information about what the device can do (geo shader, tess shader, wide lines, etc)
  VkPhysicalDeviceFeatures deviceFeatures;
  vkGetPhysicalDeviceFeatures(device, &deviceFeatures);
*/

  QueueFamilyIndices indices = getQueueFamilies(device);

  bool swapChainValid = false;
  if (checkDeviceExtensionSupport(device))
  {
    SwapChainDetails swapChainDetails = getSwapChainDetails(device);
    swapChainValid = !swapChainDetails.presentationModes.empty() &&
                     !swapChainDetails.formats.empty();
  }

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
