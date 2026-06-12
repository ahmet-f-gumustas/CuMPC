#pragma once
#include <cuda_runtime.h>

#include "core/types.cuh"

// state=[px,py,theta], control=[v,omega] — EXACT ARC integration (Euler DEĞİL), slip-aware.
__device__ __forceinline__
void diff_drive_step(float& px, float& py, float& theta,
                     float v, float omega,
                     const SlipParams& slip, float dt)
{
    const float v_eff = slip.kappa_v * v;      // longitudinal slip
    const float w_eff = slip.kappa_w * omega;  // angular slip
    const float th0   = theta;

    if (fabsf(w_eff) < 1e-5f) {                // düz çizgi limiti
        px   += v_eff * dt * cosf(th0);
        py   += v_eff * dt * sinf(th0);
        theta = th0;
    } else {                                   // EXACT ARC (sabit-curvature yay)
        const float R   = v_eff / w_eff;
        const float th1 = th0 + w_eff * dt;
        px   += R * (sinf(th1) - sinf(th0));
        py   += -R * (cosf(th1) - cosf(th0));
        theta = th1;
    }
    // lateral drift (side-slip): heading'e dik, +v_y = sola
    px += -slip.v_y * dt * sinf(th0);
    py +=  slip.v_y * dt * cosf(th0);
}
