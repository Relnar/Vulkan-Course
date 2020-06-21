#include "VulkanRenderer.h"


VulkanRenderer::VulkanRenderer()
: m_pWindow(nullptr)
{
}

VulkanRenderer::~VulkanRenderer()
{
}

int VulkanRenderer::init(GLFWwindow* a_pWindow)
{
  m_pWindow = a_pWindow;

  createInstance();

  return 0;
}

void VulkanRenderer::createInstance()
{
}
