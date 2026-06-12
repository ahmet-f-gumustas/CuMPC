#pragma once
#include <vector>

// Privileged MuJoCo geom'lerinden host ESDF build (spec §6.5).
// Build basit (brute-force O(nx·ny·obstacles)); device'taki QUERY production-identical kalır.
struct Circle { float x, y, r; };          // silindir engel (top-down)
struct AABox  { float x, y, hx, hy; };     // eksen-hizalı kutu engel (half-extents)

// data[j*nx + i] ↔ world (origin_x + i*res, origin_y + j*res); signed distance [m],
// engel içinde negatif. Engel yoksa büyük pozitif sabit.
std::vector<float> build_esdf(const std::vector<Circle>& circles,
                              const std::vector<AABox>& boxes,
                              float origin_x, float origin_y, float res,
                              int nx, int ny);
