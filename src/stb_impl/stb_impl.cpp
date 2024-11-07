#include "stb_impl.h"
#include <cstddef>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#include <cstdint>
#include <vector>

void stbi_write_hdr_as_png(const std::string& filename, int w, int h, int comp, const std::vector<float>& hdr_data, size_t offset) {
    std::vector<uint8_t> png_data;
    png_data.reserve(w * h * comp);
    for (size_t i = offset; i < hdr_data.size(); i++) {
        png_data.push_back(hdr_data[i] * 255);
    }
    stbi_write_png(filename.c_str(), w, h, comp, png_data.data(), w * comp);
}