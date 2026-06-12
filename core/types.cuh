#pragma once
#include <cuda_runtime.h>

// === NAMING ZORUNLU (spec §5) — production cuMPC'ye birebir taşınır ===
// state   = [px, py, theta]
// control = [v, omega]
// U layout: [k][t*2+dim]  (Phase 0 kararı)

struct RobotParams {
    float wheel_radius;      // [m]
    float track_width;       // [m]
    float v_max, omega_max;  // [m/s], [rad/s]
    float a_max, alpha_max;  // [m/s^2], [rad/s^2]
    float robot_radius;      // [m] collision yarıçapı
};

struct SlipParams {
    float kappa_v;  // longitudinal slip faktörü (1.0 = slip yok)
    float kappa_w;  // angular slip faktörü
    float v_y;      // lateral drift [m/s] (side-slip)
};

struct CostWeights {
    float w_lat, w_head, w_prog, w_du;
    float w_coll_hard, w_coll_soft;
    float w_slope, w_rough, w_trav, w_rollover;
    // terminal cost (spec §6.4 "opsiyonel terminal cost"): rollout sonunda
    // lookahead'e (window sonu) kalan mesafenin karesi — horizon miyopluğunda
    // (ör. engel önünde durma deadlock'u) ilerleme baskısı sağlar.
    float w_term;
};

// Cost eşik/katsayıları (config: k_accel, safe_hard, safe_soft, rollover_slope)
struct CostParams {
    float k_accel;         // accel-limit soft penalty iç katsayısı
    float safe_hard;       // [m] hard barrier eşiği (clearance)
    float safe_soft;       // [m] soft margin başlangıcı
    float rollover_slope;  // [rad] rollover eşiği
    float term_slack;      // [m] terminal cost deadband: on-pace rollout'lar
                           // sıfır terminal baskısı görür (hız dengesini bozmaz),
                           // duran/yavaş rollout'lar tam baskıyı yer (anti-deadlock)
};

struct RefPoint {
    float x, y, heading;
};

// Device-side reference window görünümü: pts = [N*3] (x, y, heading)
struct RefWindow {
    const float* pts;
    int N;
};
