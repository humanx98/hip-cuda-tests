#pragma once

#include <stb_image_write.h>
#include <string>
#include <vector>

void stbi_write_hdr_as_png(const std::string& filename, int w, int h, int comp, const std::vector<float>& data, size_t offset = 0);