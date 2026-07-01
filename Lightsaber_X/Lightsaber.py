"""Lightsaber is an ASC time-domain simulator to test novel feedback-filter designs.

Produced by Tomislav Andric and Jan Harms

Collaborators Rana Adhikari and Hang Yu from Caltech provided all the insight and data for the ASC modeling.

version 11.0 (18-March-2022) time dependent plant with Sidles-Sigg compensation.
Lightsaber implements pitch dynamics with noise inputs from ISI-L and TOP NL/NP from damping OSEMs. The dynamics
include a power-dependent Sidles-Sigg torque feedback. Lightsaber simulates the test-mass pitch soft-hard mode readout.
In lack of a state-space/SOS model for the ISI/TOP input noises, they are produced by Fourier methods in fixed-size
batches.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as signal
from tqdm import tqdm
from numpy import genfromtxt

#np.seterr(all='raise')

def faster_sosfilt(sos, x, zi):
    """Copy inputs and go directly to the cython implementation."""
    x_shape = x.shape
    zi_shape = zi.shape
    x_dtype = x.dtype
    x = np.array([x], order='C', dtype=np.float64)
    zi = np.array([zi], order='C')
    signal._sosfilt._sosfilt(sos, x, zi)  # modifies inputs in place
    x.shape = x_shape
    zi.shape = zi_shape
    return (x.astype(x_dtype), zi)

def plot_psd(timeseries, T_fft, fs, ylabel='Spectrum [Hz$^{-1/2}$]', filename=None):
    n_fft = T_fft*fs
    window = signal.kaiser(n_fft, beta=35)  # note that beta>35 does not give you more sidelobe suppression
    ff, psd = signal.welch(timeseries, fs=fs, window=window, nperseg=n_fft, noverlap=n_fft//2)

    rms = np.sqrt(1./T_fft*np.sum(psd))

    plt.figure()
    plt.loglog(ff, np.sqrt(psd), label='rms = {:5.2e}'.format(rms))
    plt.xlim(0.1, 100)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, which='both')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_hoft(h_noise, T_fft, fs, reference_data_file, label, filename=None):
    
    n_fft = T_fft*fs
    window = signal.kaiser(n_fft, beta=35)  # note that beta>35 does not give you more sidelobe suppression

    dn = pd.read_csv(reference_data_file,
                     names=['ff', 'susT', 'coatT', 'quantum', 'aplus'], delimiter=' ', skipinitialspace=True)
    ff = np.array(dn[['ff']].values.flatten())
    aplus = np.array(dn[['aplus']].values.flatten())

    plt.figure()
    for i in range(len(h_noise[0, :])):
        ff_data, psd = signal.welch(h_noise[:, i], fs=fs, window=window, nperseg=n_fft, noverlap=n_fft//2)
        plt.loglog(ff_data, np.sqrt(psd), label=label[i])
    
    
    plt.loglog(ff, aplus, label='AdV LIGO +')
    plt.xlim(1, 100)
    plt.ylim(1e-25, 3e-22)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Strain noise [Hz$^{-1/2}$]')
    plt.legend()
    #txt = " / ".join(list(label) + ["AdV LIGO +"])
    #plt.text(0.02, 0.02, txt, transform=plt.gca().transAxes, fontsize=8)
    plt.grid(True, which='both')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_diff_disp_noise(deltaL, T_fft, fs, label, filename=None):

    n_fft = T_fft*fs
    window = signal.kaiser(n_fft, beta=35)  # note that beta>35 does not give you more sidelobe suppression

    plt.figure()
    for i in range(len(deltaL[0, :])):
        ff_data, psd = signal.welch(deltaL[:, i], fs=fs, window=window, nperseg=n_fft, noverlap=n_fft//2)
        plt.loglog(ff_data, np.sqrt(psd), label=label[i])

    plt.xlim(10, 100)
    plt.ylim(1e-21, 4e-17)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Differential displacement noise [mHz$^{-1/2}$]')
    plt.legend()
    plt.grid(True, which='both')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def sos_freq_resp(sos_sys, fs, filename=None):
    w, h = signal.sosfreqz(sos_sys, worN = 100000, fs = fs)
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.semilogx(w, 20*np.log10(np.abs(h)), alpha=0.8)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('tf, mag [dB]')
    plt.xlim(0.1, 100)
    plt.grid(True, which='both')
    plt.subplot(2, 1, 2)
    plt.semilogx(w[1:46000], np.unwrap(np.angle(h[1:46000], deg=True), discont=179), alpha=0.8)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('tf, phase [deg]')
    plt.xlim(0.1, 100)
    plt.grid(True, which='both')
    plt.tight_layout()
    #plt.subplots_adjust(hspace=0)
    #plt.show()
    plt.savefig(filename, dpi=300)
    plt.close()

def transfer_function(sos_sys, T, fs, T_fft=64, ylabel='Transfer function', filename=None):

    # Fourier amplitudes of white noise
    re = np.random.normal(0, 1, T*fs//2+1)
    im = np.random.normal(0, 1, T*fs//2+1)
    wtilde = re + 1j*im
    wtilde[0] = 0

    input_signal = np.fft.irfft(wtilde)*fs

    tt = np.linspace(0, T, len(input_signal)+1)
    tt = tt[0:-1]

    state = signal.sosfilt_zi(sos_sys)
    output, zf = signal.sosfilt(sos_sys, input_signal, zi=state)

    n_fft = T_fft * fs
    window = signal.hann(n_fft)  # note that beta>35 does not give you more sidelobe suppression
    ff, pxy = signal.csd(input_signal, output, fs=fs, window=window, nperseg=n_fft, noverlap=n_fft//2)
    ff, pxx = signal.welch(input_signal, fs=fs, window=window, nperseg=n_fft, noverlap=n_fft//2)

    tf = pxy/pxx

    fi = np.logical_and(ff>0.1, ff<100)     # constrain plotted values since this leads to better automatic y-range in the plot
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.semilogx(ff[fi], 20*np.log10(np.abs(tf[fi])))  # Bode magnitude plot
    plt.xlabel('Frequency [Hz]')
    plt.ylabel(ylabel +', mag [dB]')
    plt.xlim(0.1, 100)
    plt.grid(True, which='both')
    plt.subplot(2, 1, 2)
    plt.semilogx(ff[fi], np.unwrap(np.angle(tf[fi])*180./np.pi, discont=179))  # Bode phase plot
    plt.xlabel('Frequency [Hz]')
    plt.ylabel(ylabel + ', phase [deg]')
    plt.xlim(0.1, 100)
    plt.grid(True, which='both')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()




class Plant:

    def __init__(self, physics, data, parameters, plot_dir, noise_files, reference_data_file, transfer_files, seed=None):

        self.fs = data['sampling_frequency']
        self.T_batch = data['duration_batch']
        self.T_fft = data['duration_fft']

        self.ns = []                                      # noise models read from files as PSDs
        self.tfs = []                                     # transfer functions read from files as complex amplitudes
        self.tst_noise_t = np.array([])                   # time series of test-mass noise from ISI stage 2, and damping OSEMs at top-mass
        self.input_power = []                             # time series of input power to the arm cavity
        self.cavity_power = 0.                            # inside cavity power
        self.test_mass_angular_local = np.array([0., 0.]) # test mass angles in local basis
        self.beam_spots = np.array([0., 0.])              # beam spots
        self.deltaL = 0.                                  # arm length change
        self.deltaL_hp = 0.                               # high-passed arm length change
        self.sensitivity = []                             # other noises (without ASC noise)
        self.dc_offset = np.array([3e-3, -2.6e-3])        # 3 mm DC offset in hard mode
        self.P_dc = 200000.

        self._rng_state = np.random.RandomState(seed=seed)

        self.ti = 0                                        # index running through input noise batch
        self.pumP_2_tstP_sos_state = np.zeros((2, 3, 2))
        self.tstP2P_sos_state = np.zeros((2, 3, 2))
        self.high_pass_sos_state = np.zeros((1, 2))

        self.P = physics['P']
        self.L = physics['L']
        self.R_ITM = physics['R_ITM']
        self.R_ETM = physics['R_ETM']
        self.t_ITM = physics['t_ITM']
        self.lambda0 = 1064*1e-9

        self.scale_ITM_ISI_L = float(parameters['scale_ITM_ISI_L'])
        self.scale_ETM_ISI_L = float(parameters['scale_ETM_ISI_L'])
        self.scale_OSEM_L = float(parameters['scale_OSEM_L'])
        self.scale_OSEM_P = float(parameters['scale_OSEM_P'])
        self.scale_RIN = float(parameters['scale_RIN'])

        self.set_models(plot_dir=plot_dir)                          # definition of state-space models
        self.read_noise_from_top(noise_files, plot_dir)             # read models for test-mass pitch noise from ISI/TOP OSEMs
        self.read_sus_transfer_functions(transfer_files, plot_dir)  # read transfer functions ISI/TOP -> TST
        self.create_tst_noise_from_top(plot_dir)                    # create batch of test-mass pitch noise from ISI/TOP OSEMs
        self.read_optical_noise(noise_files, reference_data_file, plot_dir)
        self.calculate_torque()
        self.initialize_parameters()
        self.high_pass()
        self.calculate_cavity_power()

    def initialize_parameters(self):

        self.rho_ITM = np.sqrt(1 - self.t_ITM)
        self.g1 = 1 - self.L / self.R_ITM
        self.g2 = 1 - self.L / self.R_ETM
        self.r = 0.5 * (self.g1 - self.g2 + np.sqrt((self.g1 - self.g2) ** 2 + 4))

        coeffic = self.L / (1 - self.g1 * self.g2)
        self.bs_matrix = coeffic * np.array([[self.g2, 1], [1, self.g1]])  # matrix that connects beam spots with local angles

        self.dL_2_strain = np.sqrt(2.)/self.L  # differential displacement to strain noise coefficient


    def reset_counters(self):
        self.ti = 0

    def read_optical_noise(self, files, reference_data_file, plot_dir):
        power_psd = genfromtxt(files[4], delimiter=',')

        frequencies = np.linspace(0, self.fs // 2, self.T_batch * self.fs // 2 + 1)

        delta_freq = 1. / self.T_batch
        norm = 0.5 * (1. / delta_freq) ** 0.5

        # Fourier amplitudes of white noise
        re = self._rng_state.normal(0, norm, len(frequencies))
        im = self._rng_state.normal(0, norm, len(frequencies))
        wtilde = re + 1j * im

        rpsd = np.interp(frequencies, power_psd[:, 0], self.scale_RIN*power_psd[:, 1], left=0, right=0)
        ctilde = wtilde * rpsd

        # set DC = 0
        ctilde[0] = 0

        self.input_power = np.fft.irfft(ctilde) * self.fs
        self.input_power = (1 + np.array(self.input_power)) * self.P

        if plot_dir:
            plot_psd(self.input_power / self.P, self.T_fft, self.fs, ylabel='Relative input power fluctuations [Hz$^{-1/2}$]',
                     filename=os.path.join(plot_dir, 'psd_of_input_power.png'))

        
        # sensitivity of AdV LIGO +
        dn = pd.read_csv(reference_data_file,
                     names=['ff', 'susT', 'coatT', 'quantum', 'aplus'], delimiter=' ', skipinitialspace=True)
        ff = np.array(dn[['ff']].values.flatten())
        aplus = np.array(dn[['aplus']].values.flatten())

        # Fourier amplitudes of white noise
        re = self._rng_state.normal(0, norm, len(frequencies))
        im = self._rng_state.normal(0, norm, len(frequencies))
        wtilde = re + 1j * im

        rpsd_sens = np.interp(frequencies, ff, aplus, left=0, right=0)
        ctilde_sens = wtilde * rpsd_sens

        # set DC = 0
        ctilde_sens[0] = 0

        self.sensitivity = np.fft.irfft(ctilde_sens) * self.fs


    def read_noise_from_top(self, files, plot_dir):
        units = ['m', 'm', 'm', 'rad']

        self.ns = []
        for k in np.arange(4):
            dn = pd.read_csv(files[k], names=['ff', 'rPSD'], delimiter=' ', skipinitialspace=True)
            ff = np.array(dn[['ff']].values.flatten())
            rpsd = np.array(dn[['rPSD']].values.flatten())

            # Use only the basename (no "noise_inputs/" prefix) for plotting
            base = os.path.basename(files[k])            # e.g. "ASD_R0_nominal.csv"
            name, _ = os.path.splitext(base)             # -> "ASD_R0_nominal"

            self.ns.append({'name': name, 'ff': ff, 'rPSD': rpsd, 'unit': units[k]})

            if plot_dir:
                plt.figure()
                plt.loglog(ff, rpsd)
                plt.xlim(0.1, 100)
                plt.xlabel('Frequency [Hz]')
                plt.ylabel('Model spectrum, {0} [{1}]'.format(name, units[k]+'/$\sqrt{\\rm Hz}$'))
                plt.grid(True, which='both')
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, name + '.png'), dpi=300)
                plt.close()



    def read_sus_transfer_functions(self, files, plot_dir):
        units = [['rad', 'm'], ['rad', 'm'], ['rad', 'm'], ['rad', 'rad']]

        self.tfs = []
        for k in range(len(files)):
            dtf = pd.read_csv(files[k], names=['ff', 'transfer'], delimiter=' ', skipinitialspace=True)
            ff = np.array(dtf[['ff']].values.flatten())
            tf = np.array(list(map(complex, dtf[['transfer']].values.flatten())))

            base = os.path.basename(files[k])     # e.g. "tf_topL_2_tstP.txt"
            name, _ = os.path.splitext(base)      # -> "tf_topL_2_tstP"

            self.tfs.append({'name': name, 'ff': ff, 'tf': tf, 'unit': units[k]})

            if plot_dir:
                plt.figure()
                plt.loglog(ff, np.abs(tf))
                plt.xlim(0.1, 100)
                plt.xlabel('Frequency [Hz]')
                plt.ylabel(name+' ['+units[k][0]+'/'+units[k][1]+']')
                plt.grid(True, which='both')
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, name + '.png'), dpi=300)
                plt.close()




    def create_tst_noise_from_top(self, plot_dir):

        frequencies = np.linspace(0, self.fs//2, self.T_batch*self.fs//2+1)
        
        delta_freq = 1./self.T_batch
        norm = 0.5 * (1. / delta_freq)**0.5

        seismic_and_damping_scaling_parameters = np.array([self.scale_ITM_ISI_L, self.scale_ETM_ISI_L, self.scale_OSEM_L, self.scale_OSEM_P])

        noises_t = np.zeros((2*(len(frequencies)-1), 6))
        ii = np.array([0, 2, 3, 1, 2, 3])
        for k in np.arange(6):
            # Fourier amplitudes of white noise
            re = self._rng_state.normal(0, norm, len(frequencies))
            im = self._rng_state.normal(0, norm, len(frequencies))
            wtilde = re + 1j * im

            # convolve with noise root PSD (note that ss or [b,a] models lead to divergence)
            rpsd = np.interp(frequencies, self.ns[ii[k]]['ff'], self.ns[ii[k]]['rPSD'], left=0, right=0)
            tf = np.interp(frequencies, self.tfs[ii[k]]['ff'], self.tfs[ii[k]]['tf'], left=0, right=0)
            ctilde = wtilde * rpsd * tf

            # set DC = 0
            ctilde[0] = 0

            n_t = seismic_and_damping_scaling_parameters[ii[k]] * np.fft.irfft(ctilde) * self.fs
            noises_t[:, k] = n_t

            if plot_dir:
                name = self.tfs[ii[k]]['name']+'x'+self.ns[ii[k]]['name']
                unit = self.tfs[ii[k]]['unit'][0]+'/$\sqrt{\\rm Hz}$'

                plot_psd(n_t, self.T_fft, self.fs,
                        ylabel='Spectrum, {0} [{1}]'.format(name,unit), filename=os.path.join(plot_dir, name+'_S.png'))

        self.tst_noise_t = np.array([np.sum(noises_t[:, :3], 1), np.sum(noises_t[:, 3:], 1)])

        if plot_dir:
            for col_idx, plot_name in [(0,"ITM"), (1, "ETM")]:
                plot_psd(self.tst_noise_t[col_idx, :], self.T_fft, self.fs,
                        ylabel='Test-mass pitch noise from ISI and TOP, ' + plot_name + ' [rad/$\sqrt{\\rm Hz}$]',
                        filename=os.path.join(plot_dir, f'n_tstP_from_isi_top_S_{plot_name}.png'))


    def set_models(self, plot_dir):
        """
        The following model is based on the zpk models from
        https://alog.ligo-la.caltech.edu/aLOG/index.php?callRep=41815

        The system defined here has its input at TM P (angle), which makes it possible to easily inject the
        ISI / TOP OSEM noise in the feedback model.
        """
        # PUM P to TM P transfer function
        zz = np.array([-2.107342e-01 + 2.871199e+00j, -2.107342e-01 - 2.871199e+00j])
        pp = np.array([-1.543716e-01 + 2.727201e+00j, -1.543716e-01 - 2.727201e+00j, -8.732026e-02 + 3.492316e+00j,
                       -8.732026e-02 - 3.492316e+00j, -3.149511e-01 + 9.411627e+00j, -3.149511e-01 - 9.411627e+00j])
        k = 9.352955e+01

        zpk = signal.bilinear_zpk(zz, pp, k, self.fs)
        self.pumP_2_tstP_sos = signal.zpk2sos(*zpk)

        # TM P to P transfer function
        self.zz = np.array([-1.772565e-01 + 2.866176e+00j, -1.772565e-01 - 2.866176e+00j, -1.755293e-01 + 7.064508e+00j,
                       -1.755293e-01 - 7.064508e+00j])
        self.pp = np.array([-1.393094e-01 + 2.737083e+00j, -1.393094e-01 - 2.737083e+00j, -8.749749e-02 + 3.493148e+00j,
                       -8.749749e-02 - 3.493148e+00j, -3.185553e-01 + 9.347665e+00j, -3.185553e-01 - 9.347665e+00j])

        zpk = signal.bilinear_zpk(self.zz, self.pp, 2.567652, self.fs)
        self.tstP2P_sos = signal.zpk2sos(*zpk)

        
        if plot_dir:
            sos_freq_resp(self.pumP_2_tstP_sos, self.fs, os.path.join(plot_dir, f'bode_pumP_2_tstP.png'))
            sos_freq_resp(self.tstP2P_sos, self.fs, os.path.join(plot_dir, f'bode_tstP2P.png'))

        return self.zz, self.pp

    def high_pass(self):
        z, p, k = signal.ellip(2, 1., 140., 2.*np.pi*50., btype='high', analog=True, output='zpk')
        k*=10.**(1./20.)

        zpk = signal.bilinear_zpk(z, p, k, self.fs)
        self.high_pass_sos = signal.zpk2sos(*zpk)


    def calculate_cavity_power(self):
        
        # in case of NONLINEAR noise coupling
        self.beam_spots = self.bs_matrix @ self.test_mass_angular_local  # arm force calculation, in local basis

        # in case of LINEAR noise coupling
        #self.beam_spots = self.bs_matrix @ self.test_mass_angular_local + self.dc_offset  # arm force calculation, in local basis

        self.deltaL = np.sum(self.beam_spots*self.test_mass_angular_local, axis=-1)  # length change of arm cavity

        # applying high-pass filter to length change for arm cavity power calculation
        output, zf = faster_sosfilt(self.high_pass_sos, np.array([self.deltaL]), zi=self.high_pass_sos_state)
        self.high_pass_sos_state = zf

        self.deltaL_hp = output[0]

        self.cavity_power = self.input_power[self.ti] * self.t_ITM / np.abs(
            1 - self.rho_ITM * np.exp(4j * np.pi * self.deltaL_hp / self.lambda0)) ** 2  # final power in the arm cavity

        return self.cavity_power


    def calculate_torque(self):

        # calculation of radiation-pressure torques on ITM and ETM
        # in case of NONLINEAR noise coupling:
        torque = 2 / 299792458.0 * (self.cavity_power*self.beam_spots + self.dc_offset*(self.cavity_power - self.P_dc))

        # in case of LINEAR noise coupling:
        # torque = 2 / 299792458.0 * (self.cavity_power * self.beam_spots - self.dc_offset*self.P_dc)

        torque_noise = np.zeros((2, ))
        for i in range(2):
            
            output, zf = faster_sosfilt(self.tstP2P_sos, np.array([torque[i]]), zi=self.tstP2P_sos_state[i])
            self.tstP2P_sos_state[i] = zf

            torque_noise[i] = output[0]
            
        return torque_noise


    def propagate(self, pum_input_signal=None, SS_comp=None):
        self.calculate_cavity_power()
        torque_noise = self.calculate_torque()

        #ITM, ETM noises (including the optical torque noise)
        pitch_ITM_ETM = self.tst_noise_t[:, self.ti] + torque_noise
        self.ti += 1

        if pum_input_signal is not None:
            for i in range(2):

                output, zf = faster_sosfilt(self.pumP_2_tstP_sos, np.array([pum_input_signal[i]]), zi=self.pumP_2_tstP_sos_state[i])
                self.pumP_2_tstP_sos_state[i] = zf

                # add signal from PUM P input torque
                pitch_ITM_ETM[i] += output[0]
        
        # subtract signal from SS compensation path
        self.test_mass_angular_local = pitch_ITM_ETM - SS_comp

        # strain noise
        asc_noise = self.dL_2_strain*self.deltaL                    # ASC strain noise
        strain_noise = asc_noise + self.sensitivity[self.ti]        # total strain noise

        return self.test_mass_angular_local, self.beam_spots, self.cavity_power, asc_noise, strain_noise, torque_noise


class Sensors:
    def __init__(self, sensing, physics, data, seed=None):

        self.fs = data['sampling_frequency']
        self.n_soft = sensing['noise_soft_mode']
        self.n_hard = sensing['noise_hard_mode']

        self.L = physics['L']
        self.R_ITM = physics['R_ITM']
        self.R_ETM = physics['R_ETM']

        self._rng_state = np.random.RandomState(seed=seed)

        self.initialize_parameters()

    def initialize_parameters(self):

        self.g1 = 1 - self.L / self.R_ITM
        self.g2 = 1 - self.L / self.R_ETM
        self.r = 0.5 * (self.g1 - self.g2 + np.sqrt((self.g1 - self.g2) ** 2 + 4))

        self.local2eigen = np.array([[1, self.r], [-self.r, 1]]) / (1 + self.r ** 2)


    def sample_readout(self, input_signal_s=np.array([0., 0.])):

        read_out = self.local2eigen @ input_signal_s
        read_out[0] += self._rng_state.normal(0, (self.fs/2.)**0.5 * self.n_soft)
        read_out[1] += self._rng_state.normal(0, (self.fs/2.)**0.5 * self.n_hard)

        return read_out


class SS_compensation:
    
    def __init__(self, data, physics, asc_plant, plot_dir):

        self.fs = data['sampling_frequency']
        self.L = physics['L']
        self.R_ITM = physics['R_ITM']
        self.R_ETM = physics['R_ETM']
        self.P_const = 56700
        self.Pav = 0
        self.N = 1
        self.zz, self.pp = asc_plant.set_models(plot_dir)
        self.initialize_parameters()

    def initialize_parameters(self):

        self.g1 = 1 - self.L / self.R_ITM
        self.g2 = 1 - self.L / self.R_ETM
        self.r = 0.5 * (self.g1 - self.g2 + np.sqrt((self.g1 - self.g2) ** 2 + 4))
        self.eigen2local = np.array([[1, -self.r], [self.r, 1]])

        self.dydth_soft = (self.L / 2) * ((self.g2 + self.g1) + np.sqrt((self.g2 - self.g1) ** 2 + 4)) / (self.g1*self.g2 - 1)
        self.dydth_hard = (self.L / 2) * ((self.g2 + self.g1) - np.sqrt((self.g2 - self.g1) ** 2 + 4)) / (self.g1*self.g2 - 1)

        self.zz_lp, self.pp_lp, self.kk_lp=signal.ellip(2, 1., 40., 2.*np.pi*17., analog=True, output='zpk')
        self.kk_lp*=10.**(1./20.)

        self.z_ss = np.hstack([self.zz, self.zz_lp])
        self.p_ss = np.hstack([self.pp, self.pp_lp])

        self.global_sos_state = np.zeros((2, 4, 2))

    def sample_compensation(self, cavity_power = 0., input_signal=np.array([0., 0.])):

        self.Pav = self.Pav+1/self.N*(cavity_power[0]-self.Pav)
        if self.N<1000:
            self.N += 1

        F = (-1)*np.array([1, 1 - self.P_const/self.Pav])     # gain-adjustment factor

        r_s = F[0] * 2 * self.Pav / 299792458. * self.dydth_soft
        k_s = 2.567652*r_s
        r_h = F[1] * 2 * self.Pav / 299792458. * self.dydth_hard
        k_h = 2.567652*r_h

        k_ss_soft = k_s*self.kk_lp
        k_ss_hard = k_h*self.kk_lp

        zpk_soft = signal.bilinear_zpk(self.z_ss, self.p_ss, k_ss_soft, self.fs)
        ss_soft_sos = signal.zpk2sos(*zpk_soft)

        zpk_hard = signal.bilinear_zpk(self.z_ss, self.p_ss, k_ss_hard, self.fs)
        ss_hard_sos = signal.zpk2sos(*zpk_hard)

        ss_global_sos = [ss_soft_sos, ss_hard_sos]

        torque_noise_SS = np.zeros((2, ))

        for i in range(2):

            output, zf = faster_sosfilt(ss_global_sos[i], np.array([input_signal[i]]), zi=self.global_sos_state[i])
            self.global_sos_state[i] = zf

            torque_noise_SS[i] = output[0]

        return self.eigen2local @ torque_noise_SS

    """def sample_compensation(self, asc_plant, input_signal=np.array([0., 0.])):
        return np.array([0., 0.])"""


class Controller:

    def __init__(self, data, physics, plot_dir, controller_name="C0_nominal"):

        self.fs = data['sampling_frequency']

        self.L = physics['L']
        self.R_ITM = physics['R_ITM']
        self.R_ETM = physics['R_ETM']

        # which hard-mode controller variant we are using
        self.controller_name = controller_name

        self.soft_control_sos_state = np.zeros((7, 2))
        self.hard_control_sos_state = np.zeros((10, 2))

        # bumpless-handover state (populated by switch_hard); _ramp_left==0 => no ramp in progress
        self._ramp_left = 0
        self._ramp_total = 1
        self._prev_hard_sos = None
        self._prev_hard_state = None

        self.initialize_parameters()
        self.set_feedback_filter_soft(plot_dir)
        self.set_feedback_filter_hard(plot_dir)

    def initialize_parameters(self):

        self.g1 = 1 - self.L / self.R_ITM
        self.g2 = 1 - self.L / self.R_ETM
        self.r = 0.5 * (self.g1 - self.g2 + np.sqrt((self.g1 - self.g2) ** 2 + 4))

        self.eigen2local = np.array([[1, -self.r], [self.r, 1]])

    def reset_counters(self):
        pass

    def set_feedback_filter_soft(self, plot_dir):
        # Example: ASC feedback filter used 2019(?) at LIGO for soft mode

        dc_gain = 20.0

        ## optical response in [ct/rad]
        K_opt = 10419.

        l2_ct2tau = 7.629e-5 * 0.268e-3 * 0.0309

        factor = dc_gain * K_opt * l2_ct2tau

        ## ctrl
        zz_ctrl = np.array([-0.88 + 8.75j, -0.88 - 8.75j, -1.885])
        pp_ctrl = np.array([-46 + 100j, -46 - 100j, -39.2 + 111j, -39.2 - 111j])
        k_ctrl = 26305469482

        z_lp, p_lp, k_lp = signal.ellip(4, 1, 30, 2. * np.pi * 14, analog=True, output='zpk')
        k_lp *= 10. ** (1. / 20.)

        # first set of zpk's
        zz_ctrl_fin = np.hstack([zz_ctrl, z_lp])
        pp_ctrl_fin = np.hstack([pp_ctrl, p_lp])
        k_ctrl_fin = k_ctrl*k_lp

        ## low-pass
        zz, pp, k = signal.ellip(2, 1, 10, 2. * np.pi * 8, analog=True, output='zpk')

        zzlp = np.array([-4.45 + 8.31j, -4.45 - 8.31j])
        pplp = np.array([-7.06 + 6.245j, -7.06 - 6.245j])
        kklp = 1

        # second set of zpk's
        zz_lp_fin = np.hstack([zz, zzlp])
        pp_lp_fin = np.hstack([pp, pplp])
        k_fin = k * 10. ** (1. / 20.)

        ## boost
        zz_boost = np.array([-1.07 + 2.75j, -1.07 - 2.75j])
        pp_boost = np.array([-0.27 + 2.94j, -0.27 - 2.94j])
        k_boost = factor

        zz_final = np.hstack([zz_ctrl_fin, zz_lp_fin, zz_boost])
        pp_final = np.hstack([pp_ctrl_fin, pp_lp_fin, pp_boost])
        gain_soft = k_ctrl_fin*k_fin*k_boost

        zpk = signal.bilinear_zpk(zz_final, pp_final, gain_soft, self.fs)
        self.soft_control_sos = signal.zpk2sos(*zpk)

        if plot_dir:
            sos_freq_resp(self.soft_control_sos, self.fs, os.path.join(plot_dir, 'bode_feedback_soft.png'))


    def set_feedback_filter_hard(self, plot_dir):
        # Example: ASC feedback filter used 2019(?) at LIGO for hard mode

        # --- choose parameters based on controller_name ---
        name = self.controller_name

        if name == "C0_nominal":
            # original nominal controller
            dc_gain = 30.0
            K_opt = 4.44e10
            l2_ct2tau = 7.629e-5 * 0.268e-3 * 0.0309

            zz_ctrl = np.array([-0.3436+4.11j, -0.3436-4.11j,
                                -0.7854+9.392j, -0.7854-9.392j])
            pp_ctrl = np.array([-78.77+171.25j, -78.77-171.25j,
                                -0.062832, -628.32])
            k_ctrl = 5797.86

            use_boost = True

        elif name == "C1_high_micro":
            dc_gain = 44.20
            K_opt = 4.44e10
            l2_ct2tau = 7.629e-5 * 0.268e-3 * 0.0309

            zz_ctrl = np.array([-0.3436+4.11j, -0.3436-4.11j,
                                -0.7854+9.392j, -0.7854-9.392j])
            pp_ctrl = np.array([-78.77+171.25j, -78.77-171.25j,
                                -0.062832, -628.32])
            k_ctrl = 5797.86

            use_boost = True

        elif name == "C2_high_micro2":
            dc_gain = 50.0
            K_opt = 4.44e10
            l2_ct2tau = 7.629e-5 * 0.268e-3 * 0.0309

            zz_ctrl = np.array([-0.3436+4.11j, -0.3436-4.11j,
                                -0.7854+9.392j, -0.7854-9.392j])
            pp_ctrl = np.array([-78.77+171.25j, -78.77-171.25j,
                                -0.062832, -628.32])
            k_ctrl = 5797.86

            use_boost = True

        else:
            # fallback to nominal if unknown name
            dc_gain = 30.0
            K_opt = 4.44e10
            l2_ct2tau = 7.629e-5 * 0.268e-3 * 0.0309

            zz_ctrl = np.array([-0.3436+4.11j, -0.3436-4.11j,
                                -0.7854+9.392j, -0.7854-9.392j])
            pp_ctrl = np.array([-78.77+171.25j, -78.77-171.25j,
                                -0.062832, -628.32])
            k_ctrl = 5797.86

            use_boost = True

        # --- rest is exactly our old code, using dc_gain, K_opt, etc. ---

        factor = dc_gain * K_opt * l2_ct2tau

        ## low-pass
        #zz_lp1, pp_lp1, k_lp1 = signal.ellip(2, 1., 40., 2. * np.pi * 10., analog=True, output='zpk')
        #zz_lp2, pp_lp2, k_lp2 = signal.ellip(4, 1., 10., 2. * np.pi * 20., analog=True, output='zpk')

        # --- per-controller shaping knobs ---
        if name == "C0_nominal":
            f_lp1 = 8.0
            f_lp2 = 16.0
            f_leak = 0.03   # Hz  (integrator pole at -2π f_leak)
        elif name == "C1_high_micro":
            f_lp1 = 10.0
            f_lp2 = 20.0
            f_leak = 0.01
        elif name == "C2_high_micro2":
            f_lp1 = 12.0
            f_lp2 = 26.0
            f_leak = 0.0    # keep near ideal integrator (or 0.003)
        else:
            f_lp1, f_lp2, f_leak = 10.0, 20.0, 0.01

        # low-pass sections
        zz_lp1, pp_lp1, k_lp1 = signal.ellip(2, 1., 40., 2.*np.pi*f_lp1, analog=True, output='zpk')
        zz_lp2, pp_lp2, k_lp2 = signal.ellip(4, 1., 10., 2.*np.pi*f_lp2, analog=True, output='zpk')

        # leaky integrator pole location
        p_int = 0.0 if f_leak <= 0 else (-2.0*np.pi*f_leak)
        z_int = -2.0*np.pi*0.1  # keep your zero if you want the same high-pass-ish shaping


        if use_boost:
            ## boost
            zz_boost = np.array([-0.322 + 0.299j, -0.322 - 0.299j,
                                 -0.786 + 0.981j, -0.786 - 0.981j,
                                 -1.068 + 2.753j, -1.068 - 2.753j,
                                 -1.53 + 4.13j, -1.53 - 4.13j])
            pp_boost = np.array([-0.161 + 0.409j, -0.161 - 0.409j,
                                 -0.313 + 1.217j, -0.313 - 1.217j,
                                 -0.268 + 2.941j, -0.268 - 2.941j,
                                 -0.24 + 4.39j, -0.24 - 4.39j])
            k_boost = factor
        else:
            zz_boost = np.array([])
            pp_boost = np.array([])
            k_boost = 1.0  # neutral multiplier

        zz_final = np.hstack([zz_ctrl, [z_int], zz_lp1, zz_lp2, zz_boost])
        pp_final = np.hstack([pp_ctrl, [p_int], pp_lp1, pp_lp2, pp_boost])

        gain_hard = k_ctrl * k_lp1 * k_lp2 * k_boost

        """zz_final = np.hstack([zz_ctrl, [-2. * np.pi * 0.1], zz_lp1, zz_lp2, zz_boost])
        pp_final = np.hstack([pp_ctrl, [0], pp_lp1, pp_lp2, pp_boost])
        gain_hard = k_ctrl*k_lp1*k_lp2*k_boost"""




        zpk = signal.bilinear_zpk(zz_final, pp_final, gain_hard, self.fs)
        self.hard_control_sos = signal.zpk2sos(*zpk)

        self.global_control_sos = [self.soft_control_sos, self.hard_control_sos]
        self.global_control_sos_state = [self.soft_control_sos_state, self.hard_control_sos_state]

        if plot_dir:
            sos_freq_resp(self.hard_control_sos, self.fs,
                          os.path.join(plot_dir, f'bode_feedback_hard_{name}.png'))


    def sample_feedback(self, input_signal=np.array([0., 0.])):

        out_soft, self.global_control_sos_state[0] = faster_sosfilt(
            self.global_control_sos[0], np.array([input_signal[0]]), zi=self.global_control_sos_state[0])
        out_hard, self.global_control_sos_state[1] = faster_sosfilt(
            self.global_control_sos[1], np.array([input_signal[1]]), zi=self.global_control_sos_state[1])

        u_soft = out_soft[0]
        u_hard = out_hard[0]

        # bumpless handover: after switch_hard(), cross-fade the applied hard actuation from the
        # previous controller (kept running) to the new one over the ramp window. At the switch
        # instant the output equals the previous controller's (no discontinuity); it slews to the
        # new controller as the ramp completes.
        if self._ramp_left > 0:
            out_prev, self._prev_hard_state = faster_sosfilt(
                self._prev_hard_sos, np.array([input_signal[1]]), zi=self._prev_hard_state)
            alpha = 1.0 - self._ramp_left/self._ramp_total
            u_hard = (1.0 - alpha)*out_prev[0] + alpha*u_hard
            self._ramp_left -= 1

        return self.eigen2local @ np.array([u_soft, u_hard])

    def switch_hard(self, new_controller_name, ramp_s=2.0):
        """Bumpless switch of the hard-mode controller. The outgoing filter is kept running and
        its output cross-faded into the incoming one over ramp_s seconds, so the actuation is
        continuous (no impulse). This is the realistic handover; calling set_feedback_filter_hard
        directly instead would reset the loop state (an actuator kick). The soft loop is unchanged."""
        if new_controller_name == self.controller_name and self._ramp_left == 0:
            return
        live_soft = self.global_control_sos_state[0].copy()   # preserve the (unchanged) soft loop
        prev_hard_sos = self.global_control_sos[1]
        prev_hard_state = self.global_control_sos_state[1].copy()
        self.controller_name = new_controller_name
        self.set_feedback_filter_hard(plot_dir=None)          # rebuilds hard SOS; resets global state
        self.global_control_sos_state[0] = live_soft          # restore live soft state
        self.global_control_sos_state[1] = np.zeros((self.hard_control_sos.shape[0], 2))  # new hard: cold, warms during ramp
        self._prev_hard_sos = prev_hard_sos
        self._prev_hard_state = prev_hard_state
        self._ramp_total = max(1, int(round(ramp_s*self.fs)))
        self._ramp_left = self._ramp_total


class FilterLP:

    def __init__(self, data, low_pass):
        self.fs = data['sampling_frequency']

        f_pass = low_pass['pass-band_edge']
        f_stop = low_pass['stop-band_edge']
        min_att = low_pass['minimum_attenuation']

        ## low-pass
        [n, fn] = signal.ellipord(f_pass, f_stop, 1, min_att, fs=self.fs)
        print('Filter order:',n)

        zz, pp, k = signal.ellip(n, 1., min_att, 2*np.pi*fn, analog=True, output='zpk')
        zpk = signal.bilinear_zpk(zz, pp, k, self.fs)
        self.low_pass_sos = signal.zpk2sos(*zpk)
        self.low_pass_sos_state = np.zeros((2, 1, 2))

    def sample(self, input_signal=np.array([0., 0.])):

        fin_out_lp = []

        for i in range(2):
            output, zf = faster_sosfilt(self.low_pass_sos, np.array([input_signal[i]]), zi=self.low_pass_sos_state[i])
            self.low_pass_sos_state[i] = zf
            
            fin_out_lp.append(output[0])

        return np.array(fin_out_lp)


class Postprocessing:
    """Strain noise filtering. Since the frequencies below 10 Hz and above 40 Hz would dominate during the 
    reward process, and controls noise is relevant between 10 Hz and 25 Hz we need to whiten the strain noise
    in order that rewards are dominated by the noise in this frequency band."""

    def __init__(self, data):

        self.fs = data['sampling_frequency']
        self.band_pass()
        self.band_pass_sos_state = np.zeros((12, 2))

    def band_pass(self):
        z1 = np.array([1 + 0j, 1 - 0j, 0.99 + 0j, 0.99 - 0j, 1.01 + 0j, 1.01 - 0j, 1 + 0j, 1 - 0j, 1 + 0j, 1 - 0j, 1 + 0j, 1 - 0j])
        p1 = np.array([-2*np.pi*5*1 + 2*np.pi*5*1j, -2*np.pi*5*1 - 2*np.pi*5*1j, -1.99*np.pi*5*1 + 1.99*np.pi*5*1j, -1.99*np.pi*5*1 - 1.99*np.pi*5*1j,
                    -2*np.pi*5*1 + 2*np.pi*5*1j, -2*np.pi*5*1 - 2*np.pi*5*1j, -2*np.pi*5*1 + 2*np.pi*5*1j, -2*np.pi*5*1 - 2*np.pi*5*1j, -2*np.pi*5*1 + 2*np.pi*5*1j, -2*np.pi*5*1 - 2*np.pi*5*1j,
                    -2*np.pi*40*1 + 2*np.pi*40*1j, -2*np.pi*40*1 - 2*np.pi*40*1j, -2*np.pi*40*1 + 2*np.pi*40*1j, -2*np.pi*40*1 - 2*np.pi*40*1j,
                    -2*np.pi*40*1 + 2*np.pi*40*1j, -2*np.pi*40*1 - 2*np.pi*40*1j, -1.99*np.pi*40*1 + 1.99*np.pi*40*1j, -1.99*np.pi*40*1 - 1.99*np.pi*40*1j,
                    -2*np.pi*40*1 + 2*np.pi*40*1j, -2*np.pi*40*1 - 2*np.pi*40*1j, -2*np.pi*40*1 + 2*np.pi*40*1j, -2*np.pi*40*1 - 2*np.pi*40*1j, -2*np.pi*40*1 + 2*np.pi*40*1j, -2*np.pi*40*1 - 2*np.pi*40*1j])
        k1 = 1.2e31

        zpk = signal.bilinear_zpk(z1, p1, k1, self.fs)
        self.band_pass_sos = signal.zpk2sos(*zpk)

    def sample(self, strain_noise=None):

        if strain_noise is not None:
            output, zf = faster_sosfilt(self.band_pass_sos, strain_noise, zi=self.band_pass_sos_state)
            self.band_pass_sos_state = zf

        return output[0]


def open_loop_run(asc_plant, asc_sensing, asc_SS_compensation, asc_controller, low_pass_filter, data, plot_dir):
    asc_plant.reset_counters()

    N = data['duration_batch']*data['sampling_frequency']

    tstP_t = np.zeros((N, 2))
    tstBS_t = np.zeros((N, 2))
    readout_t = np.zeros((N, 2))
    SS_compensation_t = np.zeros((N, 2))    # Sidles-Sigg compensation signal fed back onto test mass
    control_t = np.zeros((N, 2))
    low_passed_t = np.zeros((N, 2))
    cavity_power_t = np.zeros((N, 1))
    asc_noise_t = np.zeros((N, 1))
    strain_noise_t = np.zeros((N, 1))
    for k in tqdm(range(N-1)):
        tstP_t[k+1, :], tstBS_t[k+1, :], cavity_power_t[k+1, :], asc_noise_t[k+1, :], strain_noise_t[k+1, :], _ = asc_plant.propagate(SS_comp = SS_compensation_t[k, :])
        readout_t[k+1, :] = asc_sensing.sample_readout(input_signal_s=tstP_t[k+1, :])
        SS_compensation_t[k+1, :] = asc_SS_compensation.sample_compensation(cavity_power = cavity_power_t[k+1, :], input_signal=readout_t[k+1, :])
        control_t[k+1, :] = asc_controller.sample_feedback(input_signal=readout_t[k+1, :])
        low_passed_t[k+1, :] = low_pass_filter.sample(control_t[k+1, :])
        if np.all(np.abs(tstP_t[k+1, :]) > 1):
            print('Diverging time series at', np.round(100.*k/N),'%')
            sys.exit(0)

    # -----------------------  Everything following here is just plotting and saving of data ----------------------
    # plot of TST pitch motion and output of linear controller
    for col_idx, plot_name in [(0,"ITM"), (1, "ETM")]:
        plot_psd(tstP_t[:, col_idx], data['duration_fft'], data['sampling_frequency'],
                ylabel='TST P [rad/$\sqrt{\\rm Hz}$]', filename=os.path.join(plot_dir, f'tstP_open_loop_{plot_name}.png'))
        plot_psd(control_t[:, col_idx], data['duration_fft'], data['sampling_frequency'],
                ylabel='Control output [Nm/$\sqrt{\\rm Hz}$]', filename=os.path.join(plot_dir, f'control_output_open_loop_{plot_name}.png'))

    # plot of pitch motion as seen by the soft-mode and hard-mode sensors
    for col_idx, plot_name in [(0,"soft"), (1, "hard")]:
        plot_psd(readout_t[:, col_idx], data['duration_fft'], data['sampling_frequency'],
                ylabel='Control input [rad/$\sqrt{\\rm Hz}$]', filename=os.path.join(plot_dir, f'control_input_open_loop_{plot_name}.png'))
        
    return tstP_t, readout_t, control_t, strain_noise_t


def closed_loop_run(asc_plant, asc_sensing, asc_SS_compensation, asc_controller, postprocessing, data, plot_dir, reference_data_file):
    asc_plant.reset_counters()

    N = data['duration_batch']*data['sampling_frequency']

    # SIGNALS THAT CAN BE OBSERVED AND USED IN A COST FUNCTION
    strain_noise_t = np.zeros((N, 1))       # Strain noise of the GW detector
    readout_t = np.zeros((N, 2))            # readout signals of soft- and hard-mode sensors
    SS_compensation_t = np.zeros((N, 2))    # Sidles-Sigg compensation signal fed back onto test mass
    control_t = np.zeros((N, 2))            # angular-control signals fed back onto the two quad suspensions
    cavity_power_t = np.zeros((N, 1))       # light power inside the arm cavity

    # AUXILIARY SIGNALS THAT CANNOT BE OBSERVED
    tstP_t = np.zeros((N, 2))               # pitch motion of two test masses
    tstBS_t = np.zeros((N, 2))              # beam-spot motion on two test masses
    asc_noise_t = np.zeros((N, 1))          # only the ASC contribution to strain noise
    whitened_strain_t = np.zeros((N, 1))    # Whitened strain noise of the GW detector, for RL reward process
    torque_noise_t = np.zeros((N, 2))
    for k in tqdm(range(N-1)):
        tstP_t[k+1, :], tstBS_t[k+1, :], cavity_power_t[k+1, :], asc_noise_t[k+1, :], strain_noise_t[k+1, :], torque_noise_t[k+1, :] = asc_plant.propagate(pum_input_signal=-control_t[k, :], SS_comp = SS_compensation_t[k, :])

                # --- HARD FAIL on NaN/Inf to avoid SciPy C-level crashes ---
        if (not np.isfinite(tstP_t[k+1, :]).all() or
            not np.isfinite(cavity_power_t[k+1, :]).all() or
            not np.isfinite(control_t[k, :]).all()):
            print("NaN/Inf detected at step", k+1)
            print("tstP:", tstP_t[k+1, :])
            print("cavity_power:", cavity_power_t[k+1, :])
            print("control(prev):", control_t[k, :])
            sys.exit(0)


        readout_t[k+1, :] = asc_sensing.sample_readout(input_signal_s=tstP_t[k+1, :])
        SS_compensation_t[k+1, :] = asc_SS_compensation.sample_compensation(cavity_power = cavity_power_t[k+1, :], input_signal=readout_t[k+1, :])
        control_t[k+1, :] = asc_controller.sample_feedback(input_signal=readout_t[k+1, :])
        whitened_strain_t[k+1, :] = postprocessing.sample(strain_noise = strain_noise_t[k+1, :])
        if np.any(np.abs(tstP_t[k+1, :]) > 1) or np.any(~np.isfinite(tstP_t[k+1, :])):
            print('Diverging time series at', np.round(100.*k/N),'%')
            sys.exit(0)


    # -----------------------  Everything following here is just plotting and saving of data ----------------------
    L = 3994.5
    R_ITM = 1934
    R_ETM = 2245
    dL_2_strain = np.sqrt(2.)/L
    g1 = 1 - L / R_ITM
    g2 = 1 - L / R_ETM
    r = 0.5 * (g1 - g2 + np.sqrt((g1 - g2) ** 2 + 4))
    local2eigen = np.array([[1, r], [-r, 1]]) / (1 + r ** 2)
    
    #N = 2*262144
    N = 262144
    angles_eigen = np.zeros((N, 2))
    for k in range(N):
        angles_eigen[k, :] = local2eigen @ tstP_t[k, :]

    labels1 = ['ASC noise', 'total noise', 'total whitened noise']
    strain_noises = np.hstack((asc_noise_t, strain_noise_t, whitened_strain_t))

    # plot detector data as GW strain noise
    plot_hoft(strain_noises, data['duration_fft'], data['sampling_frequency'],
                reference_data_file=reference_data_file, label = labels1,
                filename=os.path.join(plot_dir, f'StrainNoise.png'))

    # plot detector data as displacement noise
    plot_diff_disp_noise(strain_noises/dL_2_strain, data['duration_fft'], data['sampling_frequency'], label = labels1,
                filename=os.path.join(plot_dir, f'Diff_disp_noise.png'))
    

    control_eigen = np.zeros((N, 2))
    for k in range(N):
        control_eigen[k, :] = local2eigen @ control_t[k, :]


    # plot of TST pitch motion and output of linear controller
    for col_idx, plot_name in [(0,"ITM"), (1, "ETM")]:
        plot_psd(control_t[:, col_idx], data['duration_fft'], data['sampling_frequency'],
                ylabel='Control output [Nm/$\sqrt{\\rm Hz}$]', filename=os.path.join(plot_dir, f'control_output_closed_loop_{plot_name}.png'))

    # plot of pitch motion as seen by the soft-mode and hard-mode sensors
    for col_idx, plot_name in [(0,"soft"), (1, "hard")]:
        plot_psd(angles_eigen[:, col_idx], data['duration_fft'], data['sampling_frequency'],
                ylabel='TST P [rad/$\sqrt{\\rm Hz}$]', filename=os.path.join(plot_dir, f'tstP_closed_loop_{plot_name}.png'))   
        plot_psd(readout_t[:, col_idx], data['duration_fft'], data['sampling_frequency'],
                ylabel='Control input [rad/$\sqrt{\\rm Hz}$]', filename=os.path.join(plot_dir, f'control_input_closed_loop_{plot_name}.png'))
        plot_psd(control_eigen[:, col_idx], data['duration_fft'], data['sampling_frequency'],
                ylabel='Control output [Nm/$\sqrt{\\rm Hz}$]', filename=os.path.join(plot_dir, f'control_output_closed_loop_{plot_name}.png'))
    

    # Save all data inside the run's plot directory
    def save_csv(name, arr):
        np.savetxt(os.path.join(plot_dir, name), arr, delimiter=" ")

    save_csv("torque_noise_t.csv", torque_noise_t)
    save_csv("asc_noise.csv", asc_noise_t)
    save_csv("strain_noise.csv", strain_noise_t)
    save_csv("whitened_strain_noise.csv", whitened_strain_t)
    save_csv("cavity_power.csv", cavity_power_t)
    save_csv("control_ITM_ETM.csv", control_t)
    save_csv("SS_compensation.csv", SS_compensation_t)
    save_csv("BSM_ITM_ETM.csv", tstBS_t)
    save_csv("ITM_ETM_angle.csv", tstP_t)
    save_csv("angles_eigen.csv", angles_eigen)
    save_csv("control_input.csv", readout_t)

    return tstP_t, readout_t, control_t, strain_noise_t

