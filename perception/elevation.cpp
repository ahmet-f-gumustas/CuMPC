#include "perception/elevation.h"

#include <algorithm>
#include <cmath>

namespace {
float clamp01(float x) { return std::min(std::max(x, 0.0f), 1.0f); }
}

std::vector<float> build_elevation_features(const std::vector<float>& height,
                                            int nx, int ny, float res,
                                            float slope_max, float rough_max)
{
    std::vector<float> out((size_t)nx * ny * 3, 0.0f);
    auto h = [&](int i, int j) {
        i = std::clamp(i, 0, nx - 1);
        j = std::clamp(j, 0, ny - 1);
        return height[(size_t)j * nx + i];
    };

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            // gradient (kenarda clamp'li komşu → tek taraflı farka düşer)
            const float gx = (h(i + 1, j) - h(i - 1, j)) / ((i > 0 && i < nx - 1) ? 2 * res : res);
            const float gy = (h(i, j + 1) - h(i, j - 1)) / ((j > 0 && j < ny - 1) ? 2 * res : res);
            const float slope = std::atan(std::hypot(gx, gy));

            // 3×3 lokal yükseklik std
            float mean = 0.0f;
            for (int dj = -1; dj <= 1; ++dj)
                for (int di = -1; di <= 1; ++di) mean += h(i + di, j + dj);
            mean /= 9.0f;
            float var = 0.0f;
            for (int dj = -1; dj <= 1; ++dj)
                for (int di = -1; di <= 1; ++di) {
                    const float d = h(i + di, j + dj) - mean;
                    var += d * d;
                }
            const float rough = std::sqrt(var / 9.0f);

            const float trav = clamp01(1.0f - slope / slope_max) *
                               clamp01(1.0f - rough / rough_max);

            const size_t base = ((size_t)j * nx + i) * 3;
            out[base + 0] = slope;
            out[base + 1] = rough;
            out[base + 2] = trav;
        }
    }
    return out;
}
