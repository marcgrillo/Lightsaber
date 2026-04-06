import numpy as np
from numba import njit

@njit
def sosfilt_step(sos, x, zi):
    """Single-step SOS filter iteration."""
    n_sections = sos.shape[0]
    x_curr = x
    for i in range(n_sections):
        b0, b1, b2, a0, a1, a2 = sos[i, 0], sos[i, 1], sos[i, 2], sos[i, 3], sos[i, 4], sos[i, 5]
        # Direct Form II Transposed
        y_curr = b0 * x_curr + zi[i, 0]
        zi[i, 0] = b1 * x_curr - a1 * y_curr + zi[i, 1]
        zi[i, 1] = b2 * x_curr - a2 * y_curr
        x_curr = y_curr
    return x_curr

@njit
def run_fast_physics_kernel(
    n_samples,
    ti_start,
    # Noise inputs
    sus_noise_itm, sus_noise_etm,
    input_power_tt,
    sensor_noise_soft, sensor_noise_hard,
    # Mirrors / Plant parameters
    m1_R, m1_T, m2_R, m2_T,
    wavelength, BS_offset, 
    angle_to_bs, # (2,2)
    # SOS states
    high_pass_sos, high_pass_zi,
    rad_to_angle_sos_itm, rad_to_angle_zi_itm,
    rad_to_angle_sos_etm, rad_to_angle_zi_etm,
    act_to_angle_sos_itm, act_to_angle_zi_itm,
    act_to_angle_sos_etm, act_to_angle_zi_etm,
    # Controller SOS
    ctrl_sos_soft, ctrl_zi_soft,
    ctrl_sos_hard, ctrl_zi_hard,
    ctrl_matrix, # (2,2)
    # Sidles-Sigg / Beam properties
    local_to_eigen, # (2,2)
    # SS Compensation
    ss_sos_soft, ss_sos_hard,
    ss_zi, ss_eigen2local, prev_ss,
    dydth_soft, dydth_hard, kk_lp, P_const,
    # Standard Pointers
    P_av_ptr, # Array handle to maintain Pav visibility
    N_ptr,
    # Initial state
    last_pitch, # (2,)
    last_act    # (2,)
):
    """
    Advanced JIT-compiled mathematical simulation kernel.
    Processes n_samples in a continuous physics stream to prevent Python loop overhead. 
    State variables are explicitly passed in and out to ensure uninterrupted mathematical continuity across
    Python's batch boundaries, avoiding artificial step-functions or mechanical ringing.
    """
    readout_batch = np.zeros((n_samples, 2))
    actuation_batch = np.zeros((n_samples, 2))
    pitch_batch = np.zeros((n_samples, 2))
    
    ti = ti_start
    curr_pitch = last_pitch.copy()
    
    # ADVANCED FEATURE: Seamless physical integration
    # Without explicitly initializing from `last_act`, high-frequency filters experience a violent
    # impulse response at the start of every sequence. This preserves the analog-like behavior.
    curr_act_sus = last_act.copy()
    curr_act_mirror = np.zeros(2)
    # The running low-frequency drift tracker
    curr_ss = prev_ss.copy()
    
    C_LIGHT = 299792458.0
    SQRT_R1R2 = np.sqrt(m1_R * m2_R)
    
    Pav = P_av_ptr[0]
    N = N_ptr[0]
    
    for k in range(n_samples):
        # 1. Beam spots and length change
        bs0 = angle_to_bs[0,0]*curr_pitch[0] + angle_to_bs[0,1]*curr_pitch[1] + BS_offset[0]
        bs1 = angle_to_bs[1,0]*curr_pitch[0] + angle_to_bs[1,1]*curr_pitch[1] + BS_offset[1]
        
        dL = (bs0 * curr_pitch[0]) + (bs1 * curr_pitch[1])
        
        # 2. High-pass dL
        dL_hp = sosfilt_step(high_pass_sos, dL, high_pass_zi)
        
        # 3. Cavity Power
        # P = Pin * T1 / |1 - sqrt(R1R2)*exp(4j*pi*dL_hp/lambda)|**2
        # Simple scalar math
        dL_val_f = float(dL_hp)
        phase = (12.566370614359173 * dL_val_f) / wavelength
        
        denom_re = 1.0 - SQRT_R1R2 * np.cos(phase)
        denom_im = -SQRT_R1R2 * np.sin(phase)
        denom_sq = denom_re*denom_re + denom_im*denom_im
        
        P_cav = input_power_tt[ti] * m1_T / denom_sq
        
        # 4. Radiation Pressure Torques
        # torque = 2/c * (P*BS - Pav*BS_offset) [matches Lightsaber.py logic]
        # Pav update
        Pav = Pav + (1.0/N) * (P_cav - Pav)
        if N < 1000: N += 1.0
        
        torque_dc0 = (2.0 / C_LIGHT) * Pav * BS_offset[0]
        torque_dc1 = (2.0 / C_LIGHT) * Pav * BS_offset[1]
        
        tau0 = (2.0 / C_LIGHT) * P_cav * bs0 - torque_dc0
        tau1 = (2.0 / C_LIGHT) * P_cav * bs1 - torque_dc1
        
        # 5. Mirror Propagate (SOS filtering)
        # Pitch = noise + rad_to_angle(tau) + act_to_angle(act_s) - ss_compensation(drift)
        # ITM
        a_tau0 = sosfilt_step(rad_to_angle_sos_itm, tau0 + curr_act_mirror[0], rad_to_angle_zi_itm)
        a_sus0 = sosfilt_step(act_to_angle_sos_itm, curr_act_sus[0], act_to_angle_zi_itm)
        curr_pitch[0] = sus_noise_itm[ti] + a_tau0 + a_sus0 - curr_ss[0]
        
        # ETM
        a_tau1 = sosfilt_step(rad_to_angle_sos_etm, tau1 + curr_act_mirror[1], rad_to_angle_zi_etm)
        a_sus1 = sosfilt_step(act_to_angle_sos_etm, curr_act_sus[1], act_to_angle_zi_etm)
        curr_pitch[1] = sus_noise_etm[ti] + a_tau1 + a_sus1 - curr_ss[1]
        
        # Store pitch
        pitch_batch[k, 0] = curr_pitch[0]
        pitch_batch[k, 1] = curr_pitch[1]
        
        # 6. SENSING (Readout)
        # eigen = local_to_eigen @ local
        read_e0 = local_to_eigen[0,0]*curr_pitch[0] + local_to_eigen[0,1]*curr_pitch[1] + sensor_noise_soft[k]
        read_e1 = local_to_eigen[1,0]*curr_pitch[0] + local_to_eigen[1,1]*curr_pitch[1] + sensor_noise_hard[k]
        
        readout_batch[k, 0] = read_e0
        readout_batch[k, 1] = read_e1
        
        # 7. CONTROLLER
        # Soft / Hard SOS filtering using delay-lines attached directly to the overarching controller state
        out_s = sosfilt_step(ctrl_sos_soft, read_e0, ctrl_zi_soft)
        out_h = sosfilt_step(ctrl_sos_hard, read_e1, ctrl_zi_hard)
        
        # Actuation mapping (abstract eigenbasis -> applied physical localized forces)
        # Calculates negative feedback actuation array
        act0 = -(ctrl_matrix[0,0]*out_s + ctrl_matrix[0,1]*out_h)
        act1 = -(ctrl_matrix[1,0]*out_s + ctrl_matrix[1,1]*out_h)
        
        # Update physical memory latches for next micro-tick
        curr_act_sus[0] = act0
        curr_act_sus[1] = act1
        
        actuation_batch[k, 0] = act0
        actuation_batch[k, 1] = act1
        
        # 8. SS DRIFT COMPENSATION
        # Update dynamically varying gain correctly on a per-sample basis
        if N < 1000:
            eff_Pav = Pav
        else:
            eff_Pav = max(1000.0, Pav)
        
        # Soft Gain calc
        F_soft = -1.0
        r_s_dyn = F_soft * 2.0 * eff_Pav / 299792458.0 * dydth_soft
        Gain_soft = 2.567652 * r_s_dyn
        G_soft = Gain_soft * kk_lp
        
        # Hard Gain calc
        F_hard = -(1.0 - P_const / eff_Pav)
        r_h_dyn = F_hard * 2.0 * eff_Pav / 299792458.0 * dydth_hard
        Gain_hard = 2.567652 * r_h_dyn
        G_hard = Gain_hard * kk_lp
        
        # Filter readouts dynamically through the unit-gain stabilizing tensor multiplied by live continuous gain scalars
        ss0 = sosfilt_step(ss_sos_soft, read_e0 * G_soft, ss_zi[0])
        ss1 = sosfilt_step(ss_sos_hard, read_e1 * G_hard, ss_zi[1])
        
        curr_ss[0] = ss_eigen2local[0,0]*ss0 + ss_eigen2local[0,1]*ss1
        curr_ss[1] = ss_eigen2local[1,0]*ss0 + ss_eigen2local[1,1]*ss1
        
        ti += 1

    # Apply latches to external pointers
    prev_ss[0] = curr_ss[0]
    prev_ss[1] = curr_ss[1]
    P_av_ptr[0] = Pav
    N_ptr[0] = N
    
    return readout_batch, actuation_batch, pitch_batch, curr_pitch, curr_act_sus
