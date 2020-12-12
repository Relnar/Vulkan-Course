#version 450

layout(location = 0) in vec3 fragCol;
layout(location = 1) in vec2 fragTexCoords;

layout(location = 0) out vec4 outColour;

layout(set = 1, binding = 0) uniform sampler2D Texture;

void main()
{
	vec4 texColor = texture(Texture, fragTexCoords);
	outColour = vec4(sqrt(mix(fragCol, texColor.rgb * texColor.rgb, 0.5)), texColor.a);
}