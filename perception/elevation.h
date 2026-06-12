#pragma once
#include <vector>

// hfield yüksekliklerinden terrain feature build (spec §6.5):
//   slope         = atan(|∇h|)            (merkezi fark; kenarlarda tek taraflı)
//   roughness     = 3×3 pencere yükseklik std [m]
//   traversability= clamp(1 - slope/slope_max) · clamp(1 - rough/rough_max)
// Girdi: height[j*nx + i] ↔ world (origin + i·res, origin + j·res), elevation grid çözünürlüğünde.
// Çıktı: [ny*nx*3] interleaved (slope, roughness, traversability) — MapView::query_terrain layout'u.
std::vector<float> build_elevation_features(const std::vector<float>& height,
                                            int nx, int ny, float res,
                                            float slope_max, float rough_max);
