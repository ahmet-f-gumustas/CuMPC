#pragma once
#include <cuda_runtime.h>

// grid: world ↔ index dönüşüm metadatası
struct MapMeta {
    float origin_x, origin_y, res;
    int nx, ny;
};

// Bir haritanın hücrelerinin NE ANLAMA GELDİĞİ. ADR-014 Ç3 (2026-08-12) planner'ın normalize
// `[0,1]` maliyet uzayında planlayacağına karar verdi; K9(b) o kararın uygulaması. Ama bir cihaz
// işaretçisini teslim etmek anlamı değiştirmez: rollout maliyeti hücreyi METRE cinsinden bir signed
// distance olarak okuyorsa, `[0,1]` bir maliyet ona "her yerde bir metreden az açıklık" gibi
// görünür — ve her rollout'a uygulanan bir ceza softmax'ta sadeleştiği için bu uyuşmazlık ÇIKTIDA
// GÖRÜNMEZ. Bu yüzden anlam haritanın kendisiyle birlikte taşınır ve kernel ona göre dallanır;
// çağıranın doğru olanı beslediğine güvenilmez.
enum MapSemantics : int {
    MAP_SIGNED_DISTANCE_M = 0,  // hücre = engele signed distance [m] (v1'de BudgetRT beslemez)
    MAP_NORMALIZED_COST = 1,    // hücre = [0,1] maliyet (§13.2: C_i = max(G_i, clamp(S_i,0,1)))
};

struct TerrainCell {
    float slope;           // [rad]
    float roughness;       // lokal yükseklik std [m]
    float traversability;  // [0,1], 1 = tam geçilebilir
};

// Device-side read-only harita görünümü. Query PRODUCTION-IDENTICAL:
// build basitleşse/değişse de bu fonksiyonlar production cuMPC ile birebir aynıdır.
struct MapView {
    const float* data;  // ESDF/maliyet: [ny*nx]; elevation: [ny*nx*3] (slope,rough,trav)
    MapMeta m;
    // VARSAYILAN SDF'DİR ve bu bilinçli: bu alan eklenmeden önce yazılmış her çağrı yeri metre
    // cinsinden mesafe besliyordu, ve sessizce maliyet varsaymak onların hepsini bozardı. Yeni
    // anlamı isteyen onu AÇIKÇA söyler.
    MapSemantics semantics = MAP_SIGNED_DISTANCE_M;

// device query gövdeleri yalnız CUDA TU'larında derlenir; host-only TU'lar
// (pybind) sadece struct layout'unu görür — layout her iki tarafta aynıdır.
#ifdef __CUDACC__
    __device__ __forceinline__ bool valid() const { return data != nullptr; }

    // bilinear interpolated query, sınırda clamp (ESDF: engele signed distance [m])
    __device__ __forceinline__ float query(float wx, float wy) const {
        float gx = (wx - m.origin_x) / m.res;
        float gy = (wy - m.origin_y) / m.res;
        gx = fminf(fmaxf(gx, 0.0f), (float)(m.nx - 1));
        gy = fminf(fmaxf(gy, 0.0f), (float)(m.ny - 1));
        const int x0 = (int)gx, y0 = (int)gy;
        const int x1 = min(x0 + 1, m.nx - 1);
        const int y1 = min(y0 + 1, m.ny - 1);
        const float fx = gx - (float)x0, fy = gy - (float)y0;
        const float v00 = data[y0 * m.nx + x0], v01 = data[y0 * m.nx + x1];
        const float v10 = data[y1 * m.nx + x0], v11 = data[y1 * m.nx + x1];
        return (v00 * (1.0f - fx) + v01 * fx) * (1.0f - fy) +
               (v10 * (1.0f - fx) + v11 * fx) * fy;
    }

    // bilinear 3-kanal query → TerrainCell (data layout: [ny*nx*3] interleaved)
    __device__ __forceinline__ TerrainCell query_terrain(float wx, float wy) const {
        float gx = (wx - m.origin_x) / m.res;
        float gy = (wy - m.origin_y) / m.res;
        gx = fminf(fmaxf(gx, 0.0f), (float)(m.nx - 1));
        gy = fminf(fmaxf(gy, 0.0f), (float)(m.ny - 1));
        const int x0 = (int)gx, y0 = (int)gy;
        const int x1 = min(x0 + 1, m.nx - 1);
        const int y1 = min(y0 + 1, m.ny - 1);
        const float fx = gx - (float)x0, fy = gy - (float)y0;
        TerrainCell out;
        float* ch = &out.slope;  // {slope, roughness, traversability} sıralı
        for (int c = 0; c < 3; ++c) {
            const float v00 = data[(y0 * m.nx + x0) * 3 + c];
            const float v01 = data[(y0 * m.nx + x1) * 3 + c];
            const float v10 = data[(y1 * m.nx + x0) * 3 + c];
            const float v11 = data[(y1 * m.nx + x1) * 3 + c];
            ch[c] = (v00 * (1.0f - fx) + v01 * fx) * (1.0f - fy) +
                    (v10 * (1.0f - fx) + v11 * fx) * fy;
        }
        return out;
    }
#endif  // __CUDACC__
};
