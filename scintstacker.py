import numpy as np
from absorption import AbsorptionLine

class ScintStacker:
    def __init__(self, I_nu_input, freq_arr, central_freq, dfreq, dnu, scint_flag=True, noise_flag=False, fluxdist_flag=False, fluxes=1.0, Tsys=22.0, gain=16.1):
        for i in range(I_nu_input.shape[0]):
            # print(np.max(I_nu[i]))
            I_nu_input[i] /= np.mean(I_nu_input[i])
        Inu = []
        for j in range(I_nu_input.shape[0]):
            Inu.append(np.array([np.sum(I_nu_input[j,i:i+int(dnu/dfreq)]) for i in range(0, freq_arr.shape[0], int(dnu/dfreq))]) / (dnu/dfreq))
        I_nu_input = np.array(Inu)
        del Inu
        # self.nu, I_nu_input = self.binning(freq_arr, I_nu_input, dnu)
        self.I_nu_input = I_nu_input
        self.freq_arr = freq_arr
        self.central_freq = central_freq
        self.dfreq = dfreq
        self.dnu = dnu
        self.nu = np.array([self.freq_arr[i] for i in range(0, self.freq_arr.shape[0], int(self.dnu/self.dfreq))])
        self.scint_flag = scint_flag
        self.noise_flag = noise_flag
        self.fluxdist_flag = fluxdist_flag
        self.fluxes = fluxes
        if noise_flag:
            self.Tsys = Tsys
            self.gain = gain

    def binning(self, x, y, bin_size):
        dx = x[1] - x[0]
        binsize_norm = int(bin_size // dx)
        nbins = int(len(x) // binsize_norm)
        x_binned = np.zeros(nbins)
        y_binned = np.zeros(nbins)
        for i in range(nbins):
            x_binned[i] = np.mean(x[i*binsize_norm:(i+1)*binsize_norm])
            y_binned[i] = np.mean(y[i*binsize_norm:(i+1)*binsize_norm])
        return x_binned, y_binned

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
        if np.ndim(sefd) == 2:
            noise = np.random.normal(0, sigma, (n, sefd.shape[1]))
        else:
            noise = np.random.normal(0, sigma, (n, len(sefd)))
        if len(sefd) == 1:
            noise = noise.flatten()
        return noise
    
    def flux_distribution(self, n, alpha, fmin=0.001, fmax=5000, seed=None, broken=0, breaks=[3.2, 7.8]):
        """
        Generate `n` samples whose CDF scales as F(f) ∝ f^{alpha} between fmin and fmax,
        using inverse transform sampling.

        Parameters
        ----------
        n : int
            Number of samples to generate.
        alpha : float
            Slope of the CDF (F(f) ∝ f**alpha). If alpha -> 0 this tends to the
            log-uniform (scale) distribution.
        fmin : float, optional
            Minimum flux (must be > 0). Default 1.0.
        fmax : float, optional
            Maximum flux (must be > fmin). Default 1e6.
        seed : int or None, optional
            Random seed for reproducibility.

        Returns
        -------
        None
        """
        if not self.fluxdist_flag:
            raise ValueError("Flux distribution flag is not set. Set fluxdist_flag=True to use this method.")

        if n <= 0:
            return np.empty(0)
        if fmin <= 0 or fmax <= fmin:
            raise ValueError("Require fmin>0 and fmax>fmin.")
        
        if broken == 0:
            pass
        elif broken == 1:
            self.fluxes = self.break1_power_law(n, fmin, breaks, fmax, -alpha[0], -alpha[1])
            return None
        elif broken == 2:
            self.fluxes = self.break2_power_law(n, fmin, breaks[0], breaks[1], fmax, -alpha[0], -alpha[1], -alpha[2])
            return None
        else:
            raise ValueError("Invalid value for 'broken'. Must be 0, 1, or 2.")

        rng = np.random.default_rng(seed)
        u = rng.random(n)

        # alpha == 0 -> log-uniform (limit of f = fmin * (fmax/fmin)**u)
        if np.isclose(alpha, 0.0):
            return fmin * (fmax / fmin) ** u

        low = fmin ** alpha
        high = fmax ** alpha
        self.fluxes = (u * (high - low) + low) ** (1.0 / alpha)
        return None
    
    def break1_power_law(self, n_samples, x_min, x1, x_max, y1, y2):
        """
        Generates random samples from a single-break broken power law distribution 
        using inverse transform sampling.
        
        The probability density function (PDF) is proportional to:
        f(x) \propto x^{-y1} for x_min <= x < x1
        f(x) \propto c1 * x^{-y2} for x1 <= x <= x_max
        
        where c1 is a constant ensuring continuity at x1.
        
        Parameters:
        n_samples (int): Number of samples to generate.
        x_min (float): Minimum bound of the domain.
        x1 (float): The break point.
        x_max (float): Maximum bound of the domain.
        y1 (float): Power law exponent before x1.
        y2 (float): Power law exponent after x1.
        
        Returns:
        numpy.ndarray: Array of generated samples.
        """
        if not (0 < x_min < x1 < x_max):
            raise ValueError("Domain bounds must satisfy 0 < x_min < x1 < x_max")

        # Ensure continuity at the break point:
        # f(x1-) = f(x1+) => x1**(-y1) = c1 * x1**(-y2) => c1 = x1**(y2 - y1)
        c1 = x1**(y2 - y1)

        def integral(x, y):
            """Helper to calculate indefinite integral of x^{-y}."""
            if y == 1.0:
                return np.log(x)
            else:
                return (x**(1 - y)) / (1 - y)

        # Calculate areas under each segment for normalization
        # Segment 1: x_min to x1
        area1 = integral(x1, y1) - integral(x_min, y1)
        
        # Segment 2: x1 to x_max
        area2 = c1 * (integral(x_max, y2) - integral(x1, y2))
        
        # Total area for normalization
        total_area = area1 + area2

        # Calculate probability masses for each segment
        p1 = area1 / total_area
        p2 = area2 / total_area
        
        # Cumulative probability at the break point
        cdf_x1 = p1

        # Generate uniform random variables
        u = np.random.uniform(0, 1, n_samples)
        samples = np.zeros(n_samples)

        # Invert CDF for segment 1 (x_min to x1)
        mask1 = u <= cdf_x1
        u1 = u[mask1]
        # For u in segment 1:
        # u = (integral(x, y1) - integral(x_min, y1)) / total_area
        # u * total_area + integral(x_min, y1) = integral(x, y1)
        target1 = u1 * total_area + integral(x_min, y1)
        if y1 == 1.0:
            samples[mask1] = np.exp(target1)
        else:
            samples[mask1] = (target1 * (1 - y1))**(1 / (1 - y1))

        # Invert CDF for segment 2 (x1 to x_max)
        mask2 = u > cdf_x1
        u2 = u[mask2]
        # For u in segment 2:
        # u = cdf_x1 + c1 * (integral(x, y2) - integral(x1, y2)) / total_area
        # (u - cdf_x1) * total_area / c1 + integral(x1, y2) = integral(x, y2)
        target2 = (u2 - cdf_x1) * total_area / c1 + integral(x1, y2)
        if y2 == 1.0:
            samples[mask2] = np.exp(target2)
        else:
            samples[mask2] = (target2 * (1 - y2))**(1 / (1 - y2))

        return samples
    
    def break2_power_law(self, n_samples, x_min, x1, x2, x_max, y1, y2, y3):
        """
        Generates random samples from a broken power law distribution using inverse transform sampling.
        
        The probability density function (PDF) is proportional to:
        f(x) \propto x^{-y1} for x_min <= x < x1
        f(x) \propto c1 * x^{-y2} for x1 <= x < x2
        f(x) \propto c2 * x^{-y3} for x2 <= x <= x_max
        
        where c1 and c2 are constants ensuring continuity at x1 and x2.
        
        Parameters:
        n_samples (int): Number of samples to generate.
        x_min (float): Minimum bound of the domain.
        x1 (float): First break point.
        x2 (float): Second break point.
        x_max (float): Maximum bound of the domain.
        y1 (float): Power law exponent before x1.
        y2 (float): Power law exponent between x1 and x2.
        y3 (float): Power law exponent after x2.
        
        Returns:
        numpy.ndarray: Array of generated samples.
        """
        if not (0 < x_min < x1 < x2 < x_max):
            raise ValueError("Domain bounds must satisfy 0 < x_min < x1 < x2 < x_max")

        # Ensure continuity at the break points:
        # f(x1-) = f(x1+) => x1**(-y1) = c1 * x1**(-y2) => c1 = x1**(y2 - y1)
        c1 = x1**(y2 - y1)
        # f(x2-) = f(x2+) => c1 * x2**(-y2) = c2 * x2**(-y3) => c2 = c1 * x2**(y3 - y2)
        c2 = c1 * x2**(y3 - y2)

        def integral(x, y):
            """Helper to calculate indefinite integral of x^{-y}."""
            if y == 1.0:
                return np.log(x)
            else:
                return (x**(1 - y)) / (1 - y)

        # Segment 1: x_min to x1
        area1 = integral(x1, y1) - integral(x_min, y1)
        
        # Segment 2: x1 to x2
        area2 = c1 * (integral(x2, y2) - integral(x1, y2))
        
        # Segment 3: x2 to x_max
        area3 = c2 * (integral(x_max, y3) - integral(x2, y3))

        # Total area for normalization
        total_area = area1 + area2 + area3

        # These represent the probability mass in each segment
        p1 = area1 / total_area
        p2 = area2 / total_area
        p3 = area3 / total_area
        
        # Cumulative probability at break points
        cdf_x1 = p1
        cdf_x2 = p1 + p2

        u = np.random.uniform(0, 1, n_samples)
        samples = np.zeros(n_samples)

        mask1 = u <= cdf_x1
        u1 = u[mask1]
        # For u in segment 1:
        # u = (integral(x, y1) - integral(x_min, y1)) / total_area
        # u * total_area + integral(x_min, y1) = integral(x, y1)
        target1 = u1 * total_area + integral(x_min, y1)
        if y1 == 1.0:
            samples[mask1] = np.exp(target1)
        else:
            samples[mask1] = (target1 * (1 - y1))**(1 / (1 - y1))

        mask2 = (u > cdf_x1) & (u <= cdf_x2)
        u2 = u[mask2]
        # For u in segment 2:
        # u = cdf_x1 + c1 * (integral(x, y2) - integral(x1, y2)) / total_area
        # (u - cdf_x1) * total_area / c1 + integral(x1, y2) = integral(x, y2)
        target2 = (u2 - cdf_x1) * total_area / c1 + integral(x1, y2)
        if y2 == 1.0:
            samples[mask2] = np.exp(target2)
        else:
            samples[mask2] = (target2 * (1 - y2))**(1 / (1 - y2))

        mask3 = u > cdf_x2
        u3 = u[mask3]
        # For u in segment 3:
        # u = cdf_x2 + c2 * (integral(x, y3) - integral(x2, y3)) / total_area
        # (u - cdf_x2) * total_area / c2 + integral(x2, y3) = integral(x, y3)
        target3 = (u3 - cdf_x2) * total_area / c2 + integral(x2, y3)
        if y3 == 1.0:
            samples[mask3] = np.exp(target3)
        else:
            samples[mask3] = (target3 * (1 - y3))**(1 / (1 - y3))

        return samples
    
    def flux_stacker(self, num_stack, D=17.969, NHI=8.492e20, Ts=50, line_offset=0, tol=0.001):
        absline = AbsorptionLine(NHI, Ts, D, self.central_freq, line_offset)
        tau = absline.taunu(self.nu)
        I_norm = np.exp(-tau)

        bound_array = np.argwhere(1 - I_norm > tol)
        if bound_array.size == 0:
            print("Warning: No absorption detected above the specified tolerance.")
            return np.zeros_like(self.nu), 0, 0, 0, 0
        else:
            lower_bound_idx = bound_array[0][0]
            upper_bound_idx = bound_array[-1][0]
        
        if self.scint_flag:
            I_nu = self.I_nu_input[:num_stack]
        else:
            I_nu = np.ones((num_stack, len(self.nu)))

        Inu_abs = I_nu * I_norm

        if not self.fluxdist_flag:
            Inu_abs_dist = [Inu_abs[i] * self.fluxes for i in range(Inu_abs.shape[0])]
        else:
            Inu_abs_dist = [Inu_abs[i] * self.fluxes[i] for i in range(Inu_abs.shape[0])]

        if self.noise_flag:
            SEFD = np.ones_like(Inu_abs) * self.Tsys / self.gain  # Add SEFD = Tsys / gain
            flux_nu = Inu_abs_dist + self.noise(num_stack, SEFD, 0.001, self.dnu)
        else:
            flux_nu = Inu_abs_dist

        fluxnu_stack = np.sum(flux_nu[:num_stack], axis=0)
        # print("Less than zero", np.argwhere(fluxnu_stack < 0))
        fluxnu_stack[np.argwhere(fluxnu_stack < 0)] = 0
        # flux_avg = np.mean(fluxnu_stack)

        fluxnu_stack_reduced = np.concatenate((fluxnu_stack[0:lower_bound_idx], fluxnu_stack[upper_bound_idx+1:]))
        flux_avg = np.mean(fluxnu_stack_reduced)
        fluxnu_stack_reduced /= flux_avg  # Normalize to average
        fluxnu_stack = fluxnu_stack / flux_avg  # Normalize to average
        flux_avg = np.mean(fluxnu_stack_reduced)
        flux_std = np.std(fluxnu_stack_reduced)

        # fluxnu_stack /= flux_avg  # Normalize to average
        fluxnu_stack_within = fluxnu_stack[lower_bound_idx:upper_bound_idx+1]
        deviation = np.abs(1 - np.min(fluxnu_stack_within)) / flux_std
        return fluxnu_stack, flux_avg, flux_std, deviation
    
    def search1d_NHI(self, NHI_arr, num_stack, D=17.969, Ts=50, line_offset=0, tol=0.001, alpha=-1.75, fmin=0.001, fmax=5000, seed=None, broken=0, breaks=[3.2, 7.8]):
        if self.fluxdist_flag:
            self.flux_distribution(num_stack, alpha=alpha, fmin=fmin, fmax=fmax, seed=seed, broken=broken, breaks=breaks)  # Use log-uniform distribution for fluxes
        
        deviation = np.zeros_like(NHI_arr)
        for i in range(len(NHI_arr)):
            _, _, _, deviation[i] = self.flux_stacker(num_stack, D, NHI_arr[i], Ts, line_offset, tol)
        return deviation
    
    def search1d_alpha(self, alpha_arr, num_stack, D=17.969, Ts=50, line_offset=0, tol=0.001, NHI=1e21, fmin=0.001, fmax=5000, seed=None, broken=0, breaks=[3.2, 7.8]):
        if not self.fluxdist_flag:
            raise ValueError("Flux distribution flag is not set. Set fluxdist_flag=True to use this method.")
        
        deviation = np.zeros_like(alpha_arr)
        for i in range(len(alpha_arr)):
            self.flux_distribution(num_stack, alpha=alpha_arr[i], fmin=fmin, fmax=fmax, seed=seed, broken=broken, breaks=breaks)
            _, _, _, deviation[i] = self.flux_stacker(num_stack, D, NHI, Ts, line_offset, tol)
        return deviation

    def search2d_alphaNHI(self, alpha_arr, NHI_arr, num_stack, D=17.969, Ts=50, line_offset=0, tol=0.001, fmin=0.001, fmax=5000, seed=None):
        if not self.fluxdist_flag:
            raise ValueError("Flux distribution flag is not set. Set fluxdist_flag=True to use this method.")
        
        deviation = np.zeros((len(alpha_arr), len(NHI_arr)))
        for i in range(len(alpha_arr)):
            self.flux_distribution(num_stack, alpha_arr[i], fmin=fmin, fmax=fmax, seed=seed)
            for j in range(len(NHI_arr)):
                _, _, _, deviation[i][j] = self.flux_stacker(num_stack, D, NHI_arr[j], Ts, line_offset, tol)
        return deviation
