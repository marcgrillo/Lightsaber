import numpy as np
from scipy import signal

class SS_compensation:
    def __init__(self, L, R_ITM, R_ETM, fs):
        self.fs = fs
        self.L = L
        self.R_ITM = R_ITM
        self.R_ETM = R_ETM
        self.P_const = 56700.0
        self.Pav = 0.0
        self.N = 1.0

        # Authentic TM P to P transfer function zeroes and poles from 6th_clean
        self.zz = np.array([-1.772565e-01 + 2.866176e+00j, -1.772565e-01 - 2.866176e+00j, -1.755293e-01 + 7.064508e+00j, -1.755293e-01 - 7.064508e+00j])
        self.pp = np.array([-1.393094e-01 + 2.737083e+00j, -1.393094e-01 - 2.737083e+00j, -8.749749e-02 + 3.493148e+00j, -8.749749e-02 - 3.493148e+00j, -3.185553e-01 + 9.347665e+00j, -3.185553e-01 - 9.347665e+00j])

        self.initialize_parameters()

    def initialize_parameters(self):
        self.g1 = 1 - self.L / self.R_ITM
        self.g2 = 1 - self.L / self.R_ETM
        self.r = 0.5 * (self.g1 - self.g2 + np.sqrt((self.g1 - self.g2) ** 2 + 4))
        self.eigen2local = np.array([[1, -self.r], [self.r, 1]])

        self.dydth_soft = (self.L / 2) * ((self.g2 + self.g1) + np.sqrt((self.g2 - self.g1) ** 2 + 4)) / (self.g1*self.g2 - 1)
        self.dydth_hard = (self.L / 2) * ((self.g2 + self.g1) - np.sqrt((self.g2 - self.g1) ** 2 + 4)) / (self.g1*self.g2 - 1)

        self.zz_lp, self.pp_lp, self.kk_lp = signal.ellip(2, 1., 40., 2.*np.pi*17., analog=True, output='zpk')
        self.kk_lp *= 10.**(1./20.)

        self.z_ss = np.hstack([self.zz, self.zz_lp])
        self.p_ss = np.hstack([self.pp, self.pp_lp])

        self.global_sos_state = np.zeros((2, 4, 2), dtype=np.float64)

    def get_ss_sos(self, Pav):
        F = (-1.0) * np.array([1.0, 1.0 - self.P_const / Pav]) 

        r_s = F[0] * 2.0 * Pav / 299792458.0 * self.dydth_soft
        k_s = 2.567652 * r_s
        r_h = F[1] * 2.0 * Pav / 299792458.0 * self.dydth_hard
        k_h = 2.567652 * r_h

        k_ss_soft = k_s * self.kk_lp
        k_ss_hard = k_h * self.kk_lp

        zpk_soft = signal.bilinear_zpk(self.z_ss, self.p_ss, k_ss_soft, self.fs)
        ss_soft_sos = signal.zpk2sos(*zpk_soft)

        zpk_hard = signal.bilinear_zpk(self.z_ss, self.p_ss, k_ss_hard, self.fs)
        ss_hard_sos = signal.zpk2sos(*zpk_hard)

        return ss_soft_sos, ss_hard_sos

    def get_ss_sos_unit(self):
        """Returns the compensation SOS matrices normalized with k=1.0 for dynamic inner-loop gain scaling."""
        zpk_soft = signal.bilinear_zpk(self.z_ss, self.p_ss, 1.0, self.fs)
        ss_soft_sos = signal.zpk2sos(*zpk_soft)

        zpk_hard = signal.bilinear_zpk(self.z_ss, self.p_ss, 1.0, self.fs)
        ss_hard_sos = signal.zpk2sos(*zpk_hard)

        return ss_soft_sos, ss_hard_sos
