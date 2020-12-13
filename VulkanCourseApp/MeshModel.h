#pragma once

#include <glm/glm.hpp>

#include <vector>

struct aiMesh;
struct aiNode;
struct aiScene;
class Mesh;

class MeshModel
{
public:
  MeshModel(const std::vector<Mesh*>& newMeshList);
  ~MeshModel();

  size_t getMeshCount() const { return meshList.size(); }
  Mesh* getMesh(size_t index) const { assert(index < meshList.size()); return meshList[index]; }

  const glm::mat4 getModelMatrix() const { return model; }
  void setModelMatrix(const glm::mat4& newModel) { model = newModel; }

  void destroyMeshModel();

  static std::vector<std::string> LoadMaterials(const aiScene* scene);
  static std::vector<Mesh*> LoadNode(VkPhysicalDevice newPhysicalDevice, VkDevice newDevice, VkQueue transferQueue,
                                     VkCommandPool transferCommandPool, aiNode* node, const aiScene* scene,
                                     const std::vector<int>& matToTex, VkAllocationCallbacks* callback);
  static Mesh* LoadMesh(VkPhysicalDevice newPhysicalDevice, VkDevice newDevice, VkQueue transferQueue,
                        VkCommandPool transferCommandPool, aiMesh* mesh, const aiScene* scene,
                        const std::vector<int>& matToTex, VkAllocationCallbacks* callback);

private:
  std::vector<Mesh*> meshList;
  glm::mat4 model;
};
