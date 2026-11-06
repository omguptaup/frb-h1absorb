import numpy as np
from absorption import AbsorptionLine

class ScintIndividual:
    def __init__(self, I_nu_input, freq_arr, central_freq, dnu, noise_flag=False, fluxes=1.0, SEFD=1.36):
        for i in range(I_nu_input.shape[0]):
            # print(np.max(I_nu[i]))
            I_nu_input[i] /= np.mean(I_nu_input[i])
        self.I_nu_input = I_nu_input
        self.central_freq = central_freq
        self.dnu = dnu
        self.nu = np.array(freq_arr)
        self.noise_flag = noise_flag
        self.fluxes = fluxes
        if noise_flag:
            self.SEFD = SEFD

    def noise(self, n, sefd, delt, delf, npol=2):
        """
        Generate noise for a given number of samples.
        
        Parameters:
        n (int): Number of FRB instances
        sefd (float, list, numpy.ndarray): System Equivalent Flux Density array in Jy
        delt (float): Time interval in seconds
        delf (float): Frequency interval in Hz
        Tsys (float): System temperature in K
        gain (float): Gain in K/Jy
        npol (int): Number of polarizations
        
        Returns:
        numpy.ndarray: Generated noise
        """
        if np.ndim(sefd) == 0:
            sefd = np.array([sefd])
        elif isinstance(sefd, list):
            sefd = np.array(sefd)
        sigma = sefd/np.sqrt(npol*delt*delf)
        # print(sigma.shape)
        if np.ndim(sefd) == 2:
            noise = np.random.normal(0, sigma, (n, sefd.shape[1]))
        else:
            noise = np.random.normal(0, sigma, (n, len(sefd)))
            # print(noise.shape)
        if len(sefd) == 1:
            noise = noise.flatten()
        return noise
    
    def handler(self, D=17.969, NHI=8.492e20, Ts=50, line_offset=0, tol=0.001):
        
        if np.ndim(self.nu) == 1:
            return None
        elif np.ndim(self.nu) == 2:
            tau = np.zeros_like(self.nu)
            for i in range(self.nu.shape[0]):
                absline = AbsorptionLine(NHI, Ts, D, self.central_freq[i], line_offset)
                tau[i] = absline.taunu(self.nu[i])
        I_norm = np.exp(-tau)
        # print(I_norm.shape)
        I_nu = self.I_nu_input * I_norm
        # print(I_nu.shape)
        I_nu = I_nu * self.fluxes

        flux_nu = np.zeros_like(I_nu)
        if self.noise_flag:
            for i in range(len(I_nu)):
                # SEFD = self.Tsys / self.gain  # Add SEFD = Tsys / gain
                # print(SEFD.shape, I_nu[i].shape, self.dnu[i])
                flux_nu[i] = I_nu[i] + self.noise(len(I_nu[i]), self.SEFD, 0.001, self.dnu[i])
        else:
            flux_nu = I_nu

        return flux_nu
    
    def paramsearch_fluxNHI(self, idx, NHI_arr, D=17.969, NHI=8.492e20, Ts=50, line_offset=0, tol=0.001):
        tau = np.zeros((len(NHI_arr), self.nu.shape[0]))
        for i in range(len(NHI_arr)):
            absline = AbsorptionLine(NHI_arr[i], Ts, D, self.central_freq, line_offset)
            tau[i] = absline.taunu(self.nu)
        I_norm = np.exp(-tau)

        I_nu = self.I_nu_input * I_norm

        I_nu = I_nu * self.fluxes

        flux_nu = np.zeros_like(I_nu)
        if self.noise_flag:
            for i in range(len(I_nu)):
                # SEFD = self.Tsys / self.gain  # Add SEFD = Tsys / gain
                # print(SEFD.shape, I_nu[i].shape, self.dnu[i])
                flux_nu[i] = I_nu[i] + self.noise(len(I_nu[i]), self.SEFD, 0.001, self.dnu)
        else:
            flux_nu = I_nu
        return flux_nu