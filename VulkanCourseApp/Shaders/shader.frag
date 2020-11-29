#version 450

layout(location = 0) in vec3 fragCol;
//layout(location = 1) in vec2 fragTexCoords;

layout(location = 0) out vec4 outColour;

layout(binding = 0) uniform sampler2D Texture;

void main()
{
	vec4 texColor = vec4(1.0); // texture(Texture, fragTexCoords);
	outColour = vec4(fragCol * texColor.rgb, texColor.a);
}