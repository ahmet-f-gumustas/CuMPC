#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <vector>

#include "core/debug.cuh"
#include "core/device_buffer.cuh"
#include "core/mppi.cuh"
#include "perception/costmap.h"
#include "perception/elevation.h"

namespace py = pybind11;
using farray = py::array_t<float, py::array::c_style | py::array::forcecast>;

namespace {

void require_shape(const farray& a, std::initializer_list<py::ssize_t> shape,
                   const char* name)
{
    if ((size_t)a.ndim() != shape.size())
        throw std::invalid_argument(std::string(name) + ": beklenmeyen ndim");
    int i = 0;
    for (py::ssize_t s : shape) {
        if (s >= 0 && a.shape(i) != s)
            throw std::invalid_argument(std::string(name) + ": beklenmeyen shape");
        ++i;
    }
}

MapMeta make_meta(float ox, float oy, float res, int nx, int ny)
{
    return {ox, oy, res, nx, ny};
}

}  // namespace

// Python-facing controller: NumPy I/O + H2D upload'lar; tüm kontrol matematiği CUDA'da.
class MPPIController {
public:
    explicit MPPIController(const MPPIConfig& cfg) : core_(cfg) {}

    // Two-slot immutable config, forwarded verbatim: the rules live in the core, and a binding that
    // re-implemented any of them would be a second, divergent answer to the same question.
    bool prepare(const MPPIConfig& cfg) { return core_.prepare(cfg); }
    ConfigEpoch commit() { return core_.commit(); }
    void abort_prepared() { core_.abort_prepared(); }
    bool has_prepared() const { return core_.has_prepared(); }
    ConfigEpoch active_epoch() const { return core_.active_epoch(); }
    ConfigEpoch observed_epoch() const { return core_.observed_epoch(); }

    // x: (3,) [px,py,theta]; ref: (N,3) [x,y,heading] → (2,) [v,omega]
    farray step(farray x, farray ref)
    {
        require_shape(x, {3}, "x");
        if (ref.ndim() != 2 || ref.shape(1) != 3)
            throw std::invalid_argument("ref: (N,3) bekleniyor");
        const float* xp = x.data();
        const float3 x0{xp[0], xp[1], xp[2]};
        const float2 u = core_.step(x0, ref.data(), (int)ref.shape(0));
        farray out(2);
        out.mutable_data()[0] = u.x;
        out.mutable_data()[1] = u.y;
        return out;
    }

    void set_esdf(farray grid, float ox, float oy, float res)
    {
        if (grid.ndim() != 2) throw std::invalid_argument("esdf grid: (ny,nx) bekleniyor");
        const int ny = (int)grid.shape(0), nx = (int)grid.shape(1);
        core_.set_esdf(grid.data(), make_meta(ox, oy, res, nx, ny));
    }

    void set_elevation(farray grid, float ox, float oy, float res)
    {
        if (grid.ndim() != 3 || grid.shape(2) != 3)
            throw std::invalid_argument("elevation grid: (ny,nx,3) bekleniyor");
        const int ny = (int)grid.shape(0), nx = (int)grid.shape(1);
        core_.set_elevation(grid.data(), make_meta(ox, oy, res, nx, ny));
    }

    void set_slip(float kappa_v, float kappa_w, float v_y)
    {
        core_.set_slip({kappa_v, kappa_w, v_y});
    }

    farray last_rollouts()
    {
        const auto& c = core_.config();
        farray out({(py::ssize_t)c.K, (py::ssize_t)c.H, (py::ssize_t)3});
        core_.last_rollouts(out.mutable_data());
        return out;
    }

    farray nominal()
    {
        const auto& c = core_.config();
        farray out({(py::ssize_t)c.H, (py::ssize_t)2});
        core_.nominal(out.mutable_data());
        return out;
    }

    void reset() { core_.reset(); }

    // test/debug: production-identical MapView::query yolu
    farray query_esdf(farray pts)
    {
        return run_query(pts, /*terrain=*/false);
    }
    farray query_terrain(farray pts)
    {
        return run_query(pts, /*terrain=*/true);
    }

private:
    farray run_query(farray pts, bool terrain)
    {
        if (pts.ndim() != 2 || pts.shape(1) != 2)
            throw std::invalid_argument("points: (M,2) bekleniyor");
        const int M = (int)pts.shape(0);
        const int out_dim = terrain ? 3 : 1;
        DeviceBuffer<float> d_pts((size_t)M * 2), d_out((size_t)M * out_dim);
        d_pts.upload(pts.data(), (size_t)M * 2);
        // THE MAP UPLOAD IS ASYNCHRONOUS AND LIVES ON THE CONTROLLER'S OWN STREAM. This query runs on
        // the default stream, so without this wait it can read the map BEFORE set_esdf()'s copy has
        // landed - which is exactly what three of this repository's own query tests caught the first
        // time the uploads became async. A debug/inspection path may block; the tick path may not,
        // and that asymmetry is the whole reason this wait is HERE rather than inside set_esdf().
        CUDA_CHECK(cudaStreamSynchronize(core_.stream()));
        if (terrain) {
            if (!core_.elev_view().data) throw std::runtime_error("elevation map set edilmedi");
            debug_query_terrain(core_.elev_view(), d_pts.get(), d_out.get(), M);
        } else {
            if (!core_.esdf_view().data) throw std::runtime_error("esdf map set edilmedi");
            debug_query(core_.esdf_view(), d_pts.get(), d_out.get(), M);
        }
        farray out = terrain ? farray({(py::ssize_t)M, (py::ssize_t)3})
                             : farray((py::ssize_t)M);
        d_out.download(out.mutable_data(), (size_t)M * out_dim);
        return out;
    }

    MPPI core_;
};

// test/debug: verilen control dizisini production diff_drive_step ile ilerlet
static farray simulate_controls(farray x, farray U,
                                float kappa_v, float kappa_w, float v_y, float dt)
{
    require_shape(x, {3}, "x");
    if (U.ndim() != 2 || U.shape(1) != 2)
        throw std::invalid_argument("U: (H,2) bekleniyor");
    const int H = (int)U.shape(0);
    const float* xp = x.data();
    DeviceBuffer<float> d_U((size_t)H * 2), d_states((size_t)H * 3);
    d_U.upload(U.data(), (size_t)H * 2);
    debug_simulate({xp[0], xp[1], xp[2]}, d_U.get(), d_states.get(),
                   {kappa_v, kappa_w, v_y}, dt, H);
    farray out({(py::ssize_t)H, (py::ssize_t)3});
    d_states.download(out.mutable_data(), (size_t)H * 3);
    return out;
}

static farray build_esdf_py(farray circles, farray boxes,
                            float ox, float oy, float res, int nx, int ny)
{
    std::vector<Circle> cs;
    std::vector<AABox> bs;
    if (circles.size() > 0) {
        if (circles.ndim() != 2 || circles.shape(1) != 3)
            throw std::invalid_argument("circles: (C,3) [x,y,r] bekleniyor");
        for (py::ssize_t i = 0; i < circles.shape(0); ++i) {
            const float* p = circles.data() + i * 3;
            cs.push_back({p[0], p[1], p[2]});
        }
    }
    if (boxes.size() > 0) {
        if (boxes.ndim() != 2 || boxes.shape(1) != 4)
            throw std::invalid_argument("boxes: (B,4) [x,y,hx,hy] bekleniyor");
        for (py::ssize_t i = 0; i < boxes.shape(0); ++i) {
            const float* p = boxes.data() + i * 4;
            bs.push_back({p[0], p[1], p[2], p[3]});
        }
    }
    const auto grid = build_esdf(cs, bs, ox, oy, res, nx, ny);
    farray out({(py::ssize_t)ny, (py::ssize_t)nx});
    std::copy(grid.begin(), grid.end(), out.mutable_data());
    return out;
}

static farray build_elevation_py(farray height, float res,
                                 float slope_max, float rough_max)
{
    if (height.ndim() != 2) throw std::invalid_argument("height: (ny,nx) bekleniyor");
    const int ny = (int)height.shape(0), nx = (int)height.shape(1);
    std::vector<float> h(height.data(), height.data() + (size_t)nx * ny);
    const auto feat = build_elevation_features(h, nx, ny, res, slope_max, rough_max);
    farray out({(py::ssize_t)ny, (py::ssize_t)nx, (py::ssize_t)3});
    std::copy(feat.begin(), feat.end(), out.mutable_data());
    return out;
}

PYBIND11_MODULE(cumpc_core, m)
{
    m.doc() = "cuMPC custom CUDA MPPI core (MuJoCo geliştirme kabuğu için pybind11 köprüsü)";

    py::class_<RobotParams>(m, "RobotParams")
        .def(py::init<>())
        .def_readwrite("wheel_radius", &RobotParams::wheel_radius)
        .def_readwrite("track_width", &RobotParams::track_width)
        .def_readwrite("v_max", &RobotParams::v_max)
        .def_readwrite("omega_max", &RobotParams::omega_max)
        .def_readwrite("a_max", &RobotParams::a_max)
        .def_readwrite("alpha_max", &RobotParams::alpha_max)
        .def_readwrite("robot_radius", &RobotParams::robot_radius);

    py::class_<SlipParams>(m, "SlipParams")
        .def(py::init<>())
        .def_readwrite("kappa_v", &SlipParams::kappa_v)
        .def_readwrite("kappa_w", &SlipParams::kappa_w)
        .def_readwrite("v_y", &SlipParams::v_y);

    py::class_<CostWeights>(m, "CostWeights")
        .def(py::init<>())
        .def_readwrite("w_lat", &CostWeights::w_lat)
        .def_readwrite("w_head", &CostWeights::w_head)
        .def_readwrite("w_prog", &CostWeights::w_prog)
        .def_readwrite("w_du", &CostWeights::w_du)
        .def_readwrite("w_coll_hard", &CostWeights::w_coll_hard)
        .def_readwrite("w_coll_soft", &CostWeights::w_coll_soft)
        .def_readwrite("w_slope", &CostWeights::w_slope)
        .def_readwrite("w_rough", &CostWeights::w_rough)
        .def_readwrite("w_trav", &CostWeights::w_trav)
        .def_readwrite("w_rollover", &CostWeights::w_rollover)
        .def_readwrite("w_term", &CostWeights::w_term);

    py::class_<CostParams>(m, "CostParams")
        .def(py::init<>())
        .def_readwrite("k_accel", &CostParams::k_accel)
        .def_readwrite("safe_hard", &CostParams::safe_hard)
        .def_readwrite("safe_soft", &CostParams::safe_soft)
        .def_readwrite("rollover_slope", &CostParams::rollover_slope)
        .def_readwrite("term_slack", &CostParams::term_slack);

    py::class_<MPPIConfig>(m, "MPPIConfig")
        .def(py::init<>())
        .def_readwrite("K", &MPPIConfig::K)
        .def_readwrite("H", &MPPIConfig::H)
        .def_readwrite("dt", &MPPIConfig::dt)
        // Python'da `lambda` keyword → attribute adı `lambda_` (YAML anahtarı `lambda` kalır)
        .def_readwrite("lambda_", &MPPIConfig::lambda)
        .def_readwrite("sigma_v", &MPPIConfig::sigma_v)
        .def_readwrite("sigma_omega", &MPPIConfig::sigma_omega)
        .def_readwrite("seed", &MPPIConfig::seed)
        .def_readwrite("robot", &MPPIConfig::robot)
        .def_readwrite("slip", &MPPIConfig::slip)
        .def_readwrite("weights", &MPPIConfig::weights)
        .def_readwrite("cost", &MPPIConfig::cost);

    py::class_<MPPIController>(m, "MPPIController")
        .def(py::init<const MPPIConfig&>())
        .def("step", &MPPIController::step, py::arg("x"), py::arg("ref"))
        .def("set_esdf", &MPPIController::set_esdf,
             py::arg("grid"), py::arg("origin_x"), py::arg("origin_y"), py::arg("res"))
        .def("set_elevation", &MPPIController::set_elevation,
             py::arg("grid"), py::arg("origin_x"), py::arg("origin_y"), py::arg("res"))
        .def("set_slip", &MPPIController::set_slip,
             py::arg("kappa_v"), py::arg("kappa_w"), py::arg("v_y"))
        .def("last_rollouts", &MPPIController::last_rollouts)
        .def("nominal", &MPPIController::nominal)
        .def("reset", &MPPIController::reset)
        // Two-slot immutable config. `prepare` stages and does the expensive work off the tick
        // path; `commit` swaps at a frame boundary and is refused while work is in flight.
        .def("prepare", &MPPIController::prepare, py::arg("cfg"))
        .def("commit", &MPPIController::commit)
        .def("abort_prepared", &MPPIController::abort_prepared)
        .def("has_prepared", &MPPIController::has_prepared)
        .def("active_epoch", &MPPIController::active_epoch)
        // The epoch the LAST step actually read. "published" and "applied" are two different
        // facts, and conflating them means an unapplied transition counts as applied.
        .def("observed_epoch", &MPPIController::observed_epoch)
        .def("query_esdf", &MPPIController::query_esdf, py::arg("points"))
        .def("query_terrain", &MPPIController::query_terrain, py::arg("points"));

    m.def("simulate_controls", &simulate_controls,
          py::arg("x"), py::arg("U"),
          py::arg("kappa_v") = 1.0f, py::arg("kappa_w") = 1.0f,
          py::arg("v_y") = 0.0f, py::arg("dt") = 0.05f,
          "Production diff_drive_step (device) ile deterministik state ilerletme");
    m.def("build_esdf", &build_esdf_py,
          py::arg("circles"), py::arg("boxes"),
          py::arg("origin_x"), py::arg("origin_y"), py::arg("res"),
          py::arg("nx"), py::arg("ny"));
    m.def("build_elevation_features", &build_elevation_py,
          py::arg("height"), py::arg("res"),
          py::arg("slope_max"), py::arg("rough_max"));
}
