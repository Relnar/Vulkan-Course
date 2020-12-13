#include "Mesh.h"
#include "MeshModel.h"

#include <assimp/scene.h>

MeshModel::MeshModel(const std::vector<Mesh*>& newMeshList)
: model(glm::mat4(1.0f))
{
  meshList.reserve(newMeshList.size());

  for (auto& mesh : newMeshList)
  {
    if (mesh)
    {
      meshList.push_back(mesh);
    }
  }
}

MeshModel::~MeshModel()
{
  destroyMeshModel();
}

void MeshModel::destroyMeshModel()
{
  for (auto& mesh : meshList)
  {
    mesh->destroyBuffers();
  }
  meshList.clear();
}

std::vector<std::string> MeshModel::LoadMaterials(const aiScene* scene)
{
  std::vector<std::string> textureList(scene->mNumMaterials);

  for (size_t i = 0; i < scene->mNumMaterials; ++i)
  {
    aiMaterial* material = scene->mMaterials[i];

    textureList[i] = "";

    if (auto numTextures = material->GetTextureCount(aiTextureType_DIFFUSE))
    {
      aiString path;
      if (material->GetTexture(aiTextureType_DIFFUSE, 0, &path) == AI_SUCCESS)
      {
        // Cut off any directory information already present
        int idx = std::string(path.data).rfind("\\");
        if (idx != std::string::npos)
        {
          std::string fileName = std::string(path.data).substr(idx + 1);
          textureList[i] = fileName;
        }
      }
    }
  }

  return textureList;
}

std::vector<Mesh*> MeshModel::LoadNode(VkPhysicalDevice newPhysicalDevice, VkDevice newDevice, VkQueue transferQueue,
                                       VkCommandPool transferCommandPool, aiNode* node, const aiScene* scene,
                                       const std::vector<int>& matToTex, VkAllocationCallbacks* callback)
{
  std::vector<Mesh*> meshList;
  for (size_t i = 0; i < node->mNumMeshes; ++i)
  {
    Mesh* mesh = LoadMesh(newPhysicalDevice, newDevice, transferQueue, transferCommandPool, scene->mMeshes[node->mMeshes[i]], scene, matToTex, callback);
    if (mesh)
    {
      meshList.push_back(mesh);
    }
  }

  for (size_t i = 0; i < node->mNumChildren; ++i)
  {
    std::vector<Mesh*> newList = LoadNode(newPhysicalDevice, newDevice, transferQueue, transferCommandPool, node->mChildren[i], scene, matToTex, callback);
    meshList.insert(meshList.end(), newList.begin(), newList.end());
  }

  return meshList;
}

Mesh* MeshModel::LoadMesh(VkPhysicalDevice newPhysicalDevice, VkDevice newDevice, VkQueue transferQueue,
                          VkCommandPool transferCommandPool, aiMesh* mesh, const aiScene* scene,
                          const std::vector<int>& matToTex, VkAllocationCallbacks* callback)
{
  std::vector<Vertex> vertices(mesh->mNumVertices);
  std::vector<uint32_t> indices;

  for (size_t i = 0; i < mesh->mNumVertices; ++i)
  {
    vertices[i].pos = { mesh->mVertices[i].x, mesh->mVertices[i].y, mesh->mVertices[i].z };
    if (mesh->mTextureCoords[0])
    {
      vertices[i].tex = { mesh->mTextureCoords[0][i].x, mesh->mTextureCoords[0][i].y };
    }
    else
    {
      vertices[i].tex = { 0.0f, 0.0f };
    }
    if (mesh->mColors[0])
    {
      vertices[i].col = { mesh->mColors[0][i].r, mesh->mColors[0][i].g, mesh->mColors[0][i].b };
    }
  }

  for (size_t i = 0; i < mesh->mNumFaces; ++i)
  {
    const aiFace& face = mesh->mFaces[i];
    for (size_t j = 0; j < face.mNumIndices; ++j)
    {
      indices.push_back(face.mIndices[j]);
    }
  }

  return new Mesh(newPhysicalDevice, newDevice, transferQueue, transferCommandPool, vertices, indices,
                  matToTex[mesh->mMaterialIndex], callback);
}

