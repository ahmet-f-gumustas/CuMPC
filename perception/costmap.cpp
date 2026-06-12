#include "perception/costmap.h"

#include <algorithm>
#include <cmath>

namespace {

float sdf_circle(float px, float py, const Circle& c)
{
    return std::hypot(px - c.x, py - c.y) - c.r;
}

float sdf_aabox(float px, float py, const AABox& b)
{
    const float dx = std::fabs(px - b.x) - b.hx;
    const float dy = std::fabs(py - b.y) - b.hy;
    const float outside = std::hypot(std::max(dx, 0.0f), std::max(dy, 0.0f));
    const float inside  = std::min(std::max(dx, dy), 0.0f);
    return outside + inside;
}

}  // namespace

std::vector<float> build_esdf(const std::vector<Circle>& circles,
                              const std::vector<AABox>& boxes,
                              float origin_x, float origin_y, float res,
                              int nx, int ny)
{
    constexpr float kFar = 1.0e3f;  // engelsiz hücre: "çok uzak"
    std::vector<float> grid((size_t)nx * ny, kFar);
    for (int j = 0; j < ny; ++j) {
        const float wy = origin_y + j * res;
        for (int i = 0; i < nx; ++i) {
            const float wx = origin_x + i * res;
            float d = kFar;
            for (const auto& c : circles) d = std::min(d, sdf_circle(wx, wy, c));
            for (const auto& b : boxes)   d = std::min(d, sdf_aabox(wx, wy, b));
            grid[(size_t)j * nx + i] = d;
        }
    }
    return grid;
}
