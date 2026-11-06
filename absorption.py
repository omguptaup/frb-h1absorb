import numpy as np
from scipy.special import wofz

class AbsorptionLine:
    """
    Class to model the HI 21cm absorption line profile using the Voigt-Hjerting function.
    """
    nu0 = 1.42040575177e9  # HI 21cm line frequency in Hz
    c = 299792.458 # km/s
    A21 = 2.88e-15  # Einstein A coefficient for HI 21cm line in s^-1
    h = 6.62607015e-34  # Planck's constant in J.s
    kB = 1.380649e-23  # Boltzmann's constant in J/K

    def __init__(self, NHI, Ts, cloud_diameter, nu_center, line_offset=0):
        self.line_offset = line_offset
        self.NHI = NHI
        self.Ts = Ts
        self.b = self.broadening(cloud_diameter)
        self.nu_center = nu_center
        ###
        # Return the optical depth at line center given HI column density NHI, the
        # spin temperature Ts and broadening parameter b
        #
        # c is converted to cm/s while the constants are in SI units because NHI is in cm^-2
        ###
        self.tau0 = 3.0/(32.*np.pi) * self.h/self.kB * (self.c*1e5)**2/self.nu0 * self.A21 * self.NHI / self.Ts / (np.pi**0.5 * self.b*self.nu0/self.c)

    def broadening(self, D):
        """
        Return the empirical broadening due to turbulence in molecular clouds
        from Heyer and Brunt (2004)
        """
        return 0.96 * D**0.59 #1.2 * D**0.65

    def u(self, nu):
        """
        Return the broadening normalized velocity
        """
        return self.c / self.b * (nu - (self.nu_center+self.line_offset)) / self.nu0

    def voigt(self, velnorm, gamma):
        """
        Return the Voigt line shape at x with Lorentzian component HWHM gamma
        and Gaussian component HWHM alpha.

        """
        return np.real(wofz((velnorm + 1j*gamma)))

    def taunu(self, nu):
        """
        Return the HI line profile using the Voigt-Hjerting function
        """
        vel = self.u(nu)
        line_profile = self.voigt(vel, self.A21/(4*np.pi * self.b*self.nu0/self.c))
        # return self.tau0 * np.exp(-vel**2)
        return self.tau0 * line_profile

    
class Injector:
    def __init__(self, line_comp, scat_model):
        self.line_comp = line_comp
        self.scat_model = scat_model

    def inject(self):
        profile = 1
        for i in range(len(self.line_comp)):
            profile *= np.exp(-self.line_comp[i].taunu(self.scat_model.freq))
        intensity = np.sum(np.abs(self.scat_model.realization())**2*profile, axis=0).flatten()
        return intensity/np.max(intensity)