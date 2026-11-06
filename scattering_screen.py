import numpy as np
from IPython.display import clear_output

class ScatteringScreen:
    c = 29979245800 # cm/s
    def __init__(self, scale, lpi, Rf, nu0, dnu, steps=1000):
        self.dr = 1
        self.lpi = lpi * self.dr
        self.Rf = Rf * self.lpi
        self.nu0 = nu0
        self.dnu = dnu
        self.scale = scale
        self.Rsc = int(self.Rf**2/(np.pi * self.lpi))
        self.rgrid = np.arange(0, self.Rsc+self.dr, self.dr)
        self.nu = np.arange(nu0 - dnu*steps, nu0+dnu*(steps+1), dnu)
        self.D = self.screendist(Rf*scale, nu0)
        self.dnu_sc = nu0*np.pi/(Rf/lpi)**2
          

    def geo_phase(self, r, rf, n):
        """
        Return the geometric phase of n eddies at radius r from the center of the screen.
        """
        return np.ones(n) * (r/rf)**2 * np.pi

    def scat_phase(self, r, lpi, n):
        """
        Return the phase contributed by scintillation in n eddies with diffractive scale
        lpi located at radius r from center of the screen
        This function returns random samples drawn from a normal distribution with mean=0
        and std=Phase structure function
        """
        return np.random.randn(n)*np.pi*(r/lpi)**(5./6.)

    def screendist(self, rf, nu):
        """
        Return the distance of the scattering screen based on Fresnel radius and
        observing frequency.
        Input rf in units of cm, and nu in Hz
        """
        return rf**2 * nu / self.c
    
    def simulate(self):
        amp_nu = np.zeros(self.nu.shape[0]) + np.zeros(self.nu.shape[0])*1j
        for r,j in enumerate(self.rgrid):
            npolar = int(2*np.pi*np.sqrt(r)/self.dr)
            scatp = self.scat_phase(r, self.lpi, npolar)
            for i in range(self.nu.shape[0]):
                amp_nu[i] += np.sum(np.exp(1j * (self.geo_phase(r, self.Rf, npolar) + scatp*self.nu0/self.nu[i])) / (2*np.pi * self.Rf**2)) #This should be multiplied by (dr)^2, but because dr=1, we do not as it reduces unnecessary computation
            if j%1000 == 0:
                clear_output(wait=True)
                print(j, flush=True)
        return amp_nu
    
    def acf(self, amp_nu, range_lag=150, step=1):
        I_nu = np.absolute(amp_nu)**2
        deltanu = np.arange(0.0, range_lag*self.dnu, step=step*self.dnu)
        length = I_nu.shape[0]
        corrlist = np.zeros(range_lag)
        I_avg = np.mean(I_nu)
        for i in range(0, range_lag, step):
            total = 0
            count = 0
            for j in range(0, length-range_lag):
                total += (I_nu[j]-I_avg) * (I_nu[j+i]-I_avg)
                count += 1
            corrlist[i] = total / (count * I_avg**2)
        return deltanu, corrlist
    
    