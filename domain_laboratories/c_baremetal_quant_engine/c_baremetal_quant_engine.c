/**
 * @file c_baremetal_quant_engine.c
 * @brief Zero-Allocation, Cache-Optimized Bare-Metal Implementation of Quant Hybrids (OPFOR Hardened)
 */

#include "c_baremetal_quant_engine.h"
#include <math.h>
#include <string.h>

/* -------------------------------------------------------------------------
 * 1. ASTRODYNAMIC 3-BODY LOB ENGINE
 * ------------------------------------------------------------------------- */
void astrodynamic_lob_init(astrodynamic_lob_t* RESTRICT_PTR lob, float m1, float m2, float d, float omega) {
    lob->m1 = (m1 < 0.0f) ? 0.0f : m1;
    lob->m2 = (m2 < 0.0f) ? 0.0f : m2;
    lob->d = (d < 1e-4f) ? 1e-4f : d;
    lob->omega = omega;
    
    float m_total = lob->m1 + lob->m2;
    float mu = (m_total > 1e-6f) ? (lob->m2 / m_total) : 0.5f;
    
    /* Hill Sphere Radius r_h = d * cbrt(mu / 3) */
    lob->r_hill = lob->d * cbrtf(mu / 3.0f);
    float x_m2 = (1.0f - mu) * lob->d;
    lob->l1_x = x_m2 - lob->r_hill;
    lob->l2_x = x_m2 + lob->r_hill;
}

static inline float astrodynamic_effective_potential(const astrodynamic_lob_t* RESTRICT_PTR lob, float x, float y) {
    float m_total = lob->m1 + lob->m2;
    float mu = (m_total > 1e-6f) ? (lob->m2 / m_total) : 0.5f;
    float x_m1 = -mu * lob->d;
    float x_m2 = (1.0f - mu) * lob->d;

    float dx1 = x - x_m1;
    float dx2 = x - x_m2;
    float r1_sq = dx1 * dx1 + y * y + 1e-6f;
    float r2_sq = dx2 * dx2 + y * y + 1e-6f;

    /* Fast Carmack Inverse Square Root for 1/r */
    float inv_r1 = fast_inv_sqrt(r1_sq);
    float inv_r2 = fast_inv_sqrt(r2_sq);

    float centrifugal = 0.5f * (lob->omega * lob->omega) * (x * x + y * y);
    float gravitational = (lob->m1 * inv_r1) + (lob->m2 * inv_r2);
    return centrifugal + gravitational;
}

float astrodynamic_lob_jacobi_c(const astrodynamic_lob_t* RESTRICT_PTR lob, float x, float y, float vx, float vy) {
    float omega_val = astrodynamic_effective_potential(lob, x, y);
    float v_sq = vx * vx + vy * vy;
    return (2.0f * omega_val) - v_sq;
}

bool astrodynamic_lob_is_accessible(const astrodynamic_lob_t* RESTRICT_PTR lob, float x, float y, float current_c) {
    float omega_val = astrodynamic_effective_potential(lob, x, y);
    float v_sq = (2.0f * omega_val) - current_c;
    return (v_sq >= 0.0f);
}

/* -------------------------------------------------------------------------
 * 2. BIO-PLASMODIAL PHYSARUM ROUTING CORE
 * ------------------------------------------------------------------------- */
void physarum_router_init(physarum_router_t* RESTRICT_PTR router, uint32_t nodes, float gamma, float mu) {
    router->num_nodes = (nodes > MAX_PHYSARUM_NODES) ? MAX_PHYSARUM_NODES : nodes;
    router->gamma_decay = gamma;
    router->mu_viscosity = (mu < 1e-4f) ? 1e-4f : mu;
    memset(router->conductivity, 0, sizeof(router->conductivity));
    memset(router->length, 0, sizeof(router->length));
    memset(router->pressure, 0, sizeof(router->pressure));
    memset(router->flux, 0, sizeof(router->flux));

    for (uint32_t i = 0; i < router->num_nodes; i++) {
        for (uint32_t j = 0; j < router->num_nodes; j++) {
            router->conductivity[i][j] = (i == j) ? 0.0f : 0.1f;
            router->length[i][j] = (i == j) ? 0.0f : 1e6f;
        }
    }
}

void physarum_router_set_edge(physarum_router_t* RESTRICT_PTR router, uint32_t u, uint32_t v, float length, float conductance) {
    if (u < router->num_nodes && v < router->num_nodes) {
        router->length[u][v] = (length < 1e-4f) ? 1e-4f : length;
        router->length[v][u] = router->length[u][v];
        router->conductivity[u][v] = (conductance < 0.0f) ? 0.0f : conductance;
        router->conductivity[v][u] = router->conductivity[u][v];
    }
}

void physarum_router_step_adaptation(physarum_router_t* RESTRICT_PTR router, uint32_t source, uint32_t sink, float input_flux, float dt) {
    for (uint32_t iter = 0; iter < 32; iter++) {
        for (uint32_t i = 0; i < router->num_nodes; i++) {
            if (i == sink) {
                router->pressure[i] = 0.0f;
                continue;
            }
            float diag_sum = 0.0f;
            float off_diag_sum = 0.0f;
            for (uint32_t j = 0; j < router->num_nodes; j++) {
                if (i != j && router->length[i][j] < 1e5f) {
                    float k_ij = router->conductivity[i][j] / router->length[i][j];
                    diag_sum += k_ij;
                    off_diag_sum += k_ij * router->pressure[j];
                }
            }
            float s_i = (i == source) ? input_flux : 0.0f;
            if (diag_sum > 1e-6f) {
                router->pressure[i] = (off_diag_sum + s_i) / diag_sum;
            }
        }
    }

    for (uint32_t i = 0; i < router->num_nodes; i++) {
        for (uint32_t j = i + 1; j < router->num_nodes; j++) {
            if (router->length[i][j] < 1e5f) {
                float q_ij = (router->conductivity[i][j] / (router->mu_viscosity * router->length[i][j])) * (router->pressure[i] - router->pressure[j]);
                router->flux[i][j] = q_ij;
                router->flux[j][i] = -q_ij;
                float abs_q = fabsf(q_ij);
                float d_cond = (abs_q - router->gamma_decay * router->conductivity[i][j]) * dt;
                float new_cond = router->conductivity[i][j] + d_cond;
                if (new_cond < 0.001f) new_cond = 0.001f;
                router->conductivity[i][j] = new_cond;
                router->conductivity[j][i] = new_cond;
            }
        }
    }
}

/* -------------------------------------------------------------------------
 * 3. MHD & GRAVITATIONAL LENSING RECONSTRUCTOR
 * ------------------------------------------------------------------------- */
void mhd_lensing_init(mhd_lensing_t* RESTRICT_PTR mhd, float theta_e, float eta, float r_m_threshold) {
    mhd->theta_e = (theta_e < 0.0f) ? 0.0f : theta_e;
    mhd->theta_e_sq = mhd->theta_e * mhd->theta_e;
    mhd->eta = (eta < 1e-5f) ? 1e-5f : eta;
    mhd->r_m_threshold = r_m_threshold;
}

float mhd_lensing_reconstruct_source(const mhd_lensing_t* RESTRICT_PTR mhd, float observed_theta) {
    if (fabsf(observed_theta) < 1e-6f) return 0.0f;
    return observed_theta - (mhd->theta_e_sq / observed_theta);
}

float mhd_lensing_compute_rm(const mhd_lensing_t* RESTRICT_PTR mhd, float tick_velocity, float depth_scale) {
    return (fabsf(tick_velocity) * fabsf(depth_scale)) / mhd->eta;
}

bool mhd_lensing_is_turbulent(const mhd_lensing_t* RESTRICT_PTR mhd, float r_m) {
    return (r_m >= mhd->r_m_threshold);
}

/* -------------------------------------------------------------------------
 * 4. THALAMOCORTICAL PHASE-AMPLITUDE COUPLING
 * ------------------------------------------------------------------------- */
void neuro_pac_init(neuro_pac_t* RESTRICT_PTR pac) {
    memset(pac->bin_amplitudes, 0, sizeof(pac->bin_amplitudes));
    memset(pac->bin_counts, 0, sizeof(pac->bin_counts));
    pac->modulation_index = 0.0f;
    pac->phase_locking_value = 0.0f;
}

void neuro_pac_accumulate_sample(neuro_pac_t* RESTRICT_PTR pac, float theta_phase_rad, float gamma_amplitude) {
    float norm_phase = (theta_phase_rad + 3.14159265f) * (float)(NUM_PAC_BINS) / (2.0f * 3.14159265f);
    int32_t bin = (int32_t)norm_phase;
    if (bin < 0) bin = 0;
    if (bin >= NUM_PAC_BINS) bin = NUM_PAC_BINS - 1;

    pac->bin_amplitudes[bin] += (gamma_amplitude < 0.0f) ? 0.0f : gamma_amplitude;
    pac->bin_counts[bin]++;
}

float neuro_pac_compute_mi(neuro_pac_t* RESTRICT_PTR pac) {
    float total_amplitude = 0.0f;
    float mean_bins[NUM_PAC_BINS];

    for (uint32_t i = 0; i < NUM_PAC_BINS; i++) {
        mean_bins[i] = (pac->bin_counts[i] > 0) ? (pac->bin_amplitudes[i] / (float)pac->bin_counts[i]) : 1e-6f;
        total_amplitude += mean_bins[i];
    }

    if (total_amplitude <= 1e-6f) return 0.0f;

    float d_kl = 0.0f;
    float u = 1.0f / (float)NUM_PAC_BINS;
    for (uint32_t i = 0; i < NUM_PAC_BINS; i++) {
        float p_i = mean_bins[i] / total_amplitude;
        if (p_i > 1e-12f) {
            d_kl += p_i * logf(p_i / u);
        }
    }
    pac->modulation_index = d_kl / logf((float)NUM_PAC_BINS);
    return pac->modulation_index;
}

/* -------------------------------------------------------------------------
 * 5. BITWISE REGISTER INVARIANT ENGINE
 * ------------------------------------------------------------------------- */
void bitwise_invariant_init(bitwise_invariant_engine_t* RESTRICT_PTR engine) {
    engine->invariant_bitmask_required = 0;
    engine->invariant_bitmask_forbidden = 0;
}

void bitwise_invariant_set_rule(bitwise_invariant_engine_t* RESTRICT_PTR engine, uint64_t required_bits, uint64_t forbidden_bits) {
    engine->invariant_bitmask_required = required_bits;
    engine->invariant_bitmask_forbidden = forbidden_bits;
}

bool bitwise_invariant_verify(const bitwise_invariant_engine_t* RESTRICT_PTR engine, uint64_t market_state_bits) {
    bool has_required = ((market_state_bits & engine->invariant_bitmask_required) == engine->invariant_bitmask_required);
    bool has_no_forbidden = ((market_state_bits & engine->invariant_bitmask_forbidden) == 0ULL);
    return has_required && has_no_forbidden;
}
