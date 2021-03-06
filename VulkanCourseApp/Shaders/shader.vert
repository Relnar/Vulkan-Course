#version 450

layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 col;
layout(location = 2) in vec2 texCoords;

layout(set = 0, binding = 0) uniform UboViewProjection
{
	mat4 projection;
	mat4 view;
} uboViewProjection;

#ifndef USING_PUSH_CONSTANT
layout(set = 0, binding = 1) uniform UboModel
{
	mat4 model;
} uboModel;

#else
layout(push_constant) uniform PushModel
{
	mat4 model;
} pushModel;
#endif

layout(location = 0) out vec3 fragCol;
layout(location = 1) out vec2 fragTexCoords;

void main()
{
	gl_Position = uboViewProjection.projection * uboViewProjection.view *
#ifndef USING_PUSH_CONSTANT
				  uboModel.model *
#else
				  pushModel.model *
#endif
				  vec4(pos, 1.0);

	fragCol = col;
	fragTexCoords = texCoords;
}