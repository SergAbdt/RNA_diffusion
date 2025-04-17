import fipy as fp
import numpy as np


class free_energy_RNA_Chrom_FH:
    """
    Defining a class capturing the free energy of interaction between RNA, protein and chromatin
    """
    
    def __init__(self, NP, NR, chi_p, chi_pr, chi_r, c_RNAchr, sigma_RNAchr, wall, wall_k, neg_max):
        
        self.NP = NP
        self.NR = NR
        self.chi_p = chi_p
        self.chi_pr = chi_pr
        self.chi_r = chi_r
        self.c_RNAchr = c_RNAchr
        self.sigma_RNAchr = sigma_RNAchr
        self.wall = wall
        self.wall_k = wall_k
        self.neg_max = neg_max
    
    def f(self, phi_p, phi_r, d_chr):
        r"""
        Returns overall free-energy per unit volume:
        """
        
        phi_p=np.copy(phi_p.value)
        phi_p[(phi_p<0)|(phi_p>1)]=np.nan
        phi_r=np.copy(phi_r.value)
        phi_r[(phi_r<0)|(phi_r>1)]=np.nan
        
        fe = (phi_p*np.log(phi_p)/self.NP + (1.0-phi_p)*np.log(1.0-phi_p) - self.chi_p*phi_p**2 - self.chi_pr*phi_p*phi_r +
             0.5*phi_r**2*phi_p + 0.5*phi_r*phi_p**2 + 0.5*phi_r**2*phi_p**2 + self.chi_r*phi_r**2 -
             self.c_RNAchr/d_chr*np.exp(-d_chr/self.sigma_RNAchr)*phi_r + phi_r*np.log(phi_r)/self.NR) # +
             #self.c_LJ*((self.sigma_LJ/d_chr)**12 - (self.sigma_LJ/d_chr)**6)*(phi_p + phi_r))

        if self.wall:
             fe += (self.wall_k*(phi_p > 1 - self.neg_max)*(phi_p-self.neg_max)**4 +
                    self.wall_k*(phi_r > 1 - self.neg_max)*(phi_r-self.neg_max)**4)

        return fe
        
    def mu_p(self, phi_p, phi_r, d_chr):
        r"""
        Returns protein chemical potential

        .. math::
            \mu_{p} = \\frac{df}{d \phi_{p}}
        """
        
        phi_p=np.copy(phi_p.value)
        phi_p[(phi_p<0)|(phi_p>1)]=np.nan

        mu_p = ((1.0+np.log(phi_p))/self.NP - 1.0 - np.log(1.0-phi_p) - 2*self.chi_p*phi_p - 
                self.chi_pr*phi_r + 0.5*phi_r**2 + phi_r*phi_p + phi_r**2*phi_p) # +
                #self.c_LJ*((self.sigma_LJ/d_chr)**12 - (self.sigma_LJ/d_chr)**6))

        if self.wall:
            mu_p += 4*self.wall_k*(phi_p > 1 - self.neg_max)*(phi_p-self.neg_max)**3
        
        return mu_p

    def mu_r(self, phi_p, phi_r, d_chr):
        r"""
        Returns RNA chemical potential

        .. math::
            \mu_{r} = \\frac{df}{d \phi_{r}}
        """
        
        phi_r=np.copy(phi_r.value)
        phi_r[(phi_r<0)|(phi_r>1)]=np.nan
        
        mu_r = (-self.chi_pr*phi_p + phi_p*phi_r  + 0.5*phi_p**2 + phi_r*phi_p**2 + 2.0*self.chi_r*phi_r -
                self.c_RNAchr/d_chr*np.exp(-d_chr/self.sigma_RNAchr) + (1.0+np.log(phi_r))/self.NR) # +
                #self.c_LJ*((self.sigma_LJ/d_chr)**12 - (self.sigma_LJ/d_chr)**6))

        if self.wall:
            mu_r += 4*self.wall_k*(phi_r > 1 - self.neg_max)*(phi_r-self.neg_max)**3

        return mu_r         
        
    def dmu_p_dphi_p(self, phi_p, phi_r):     
        r"""
        Returns derivative of protein chemical potential with protein concentration multiplied by protein concentration

        .. math::
             \frac{d^{2}f}{d \phi_{p}^{2}} * \phi_{p}
        """

        dmu_p_dphi_p = (1/self.NP + phi_p*(1.0-phi_p)**(-1) - 2*self.chi_p*phi_p + 
                        phi_p*phi_r + phi_p*phi_r**2)

        if self.wall:
            dmu_p_dphi_p += 3*4*self.wall_k*(phi_p > 1 - self.neg_max)*(phi_p-self.neg_max)**2
            
        return dmu_p_dphi_p

    def dmu_p_dphi_r(self, phi_p, phi_r):
        r"""
        Returns mixed second derivative of free-energy multiplied by protein concentration

        .. math::
             \frac{d^{2}f}{d \phi_{p} \phi_{r}} * \phi_{p}
        """
        
        return (-self.chi_pr + phi_r + phi_p + 2*phi_p*phi_r)*phi_p  

    def dmu_r_dphi_p(self, phi_p, phi_r):
        r"""
        Returns mixed second derivative of free-energy multiplied by RNA concentration

        .. math::
             \frac{d^{2}f}{d \phi_{p} \phi_{r}} * \phi_{r}
        """

        return (-self.chi_pr + phi_r + phi_p + 2*phi_p*phi_r)*phi_r

    def dmu_r_dphi_r(self, phi_p, phi_r):
        r"""
        Returns derivative of RNA chemical potential with RNA concentration multiplied by RNA concentration

        .. math::
             \frac{d^{2}f}{d \phi_{r}^{2}} * \phi_{r}
        """

        dmu_r_dphi_r = 1/self.NR + (2.0*self.chi_r + phi_p +  phi_p**2)*phi_r

        if self.wall:
            dmu_r_dphi_r += 3*4*self.wall_k*(phi_r > 1 - self.neg_max)*(phi_r-self.neg_max)**2
            
        return dmu_r_dphi_r
    
    def specific_DH(self, mesh, d_chr):
        r"""
        Returns specific Debye–Hückel potential (for dense chromatin)
        
        ..math::
            \underline{U_{RNAchr}}[d_{chr}]
        """
        
        return fp.CellVariable(mesh = mesh, name = r'$\underline{U_{RNAchr}}[d_{chr}]$', value=-self.c_RNAchr/d_chr * np.exp(-d_chr/self.sigma_RNAchr))

    def specific_RNAchr(self, mesh, d_chr):
        r"""
        Returns specific gaussian RNA-Chromatin interaction potential (for sparse chromatin)
        
        ..math::
            \underline{U_{RNAchr}}[d_{chr}]
        """
        
        return fp.CellVariable(mesh = mesh, name = r'$\underline{U_{RNAchr}}[d_{chr}]$', value=-self.c_RNAchr * np.exp(-d_chr**2/self.sigma_RNAchr**2))
    
    
class RNA_reactions:
    """
    Defining a class describing RNA production and degradation
    """

    def __init__(self, mesh, k_p_max, k_degradation, spread, center, sigma_chr, phi_threshold=0.0, n_hill=1.0, sparse=0):

        self.k_p_max = k_p_max
        self.k_d = k_degradation
        self.std = spread
        self.center = center
        self.sigma_chr = sigma_chr
        self.phi_threshold = phi_threshold
        self.n_hill = n_hill
        self.sparse = sparse

        if not self.sparse:
            self.k_p_x = self.k_p_max*fp.CellVariable(mesh = mesh, name=r'$k_p(x)$', value = np.exp(-((self.center[0]-mesh.cellCenters[0])**2 + (self.center[1]-mesh.cellCenters[1])**2 + (self.center[2]-mesh.cellCenters[2])**2 - self.sigma_chr**2)/(self.std+self.sigma_chr)**2))
        else:
            self.k_p_x = self.k_p_max*fp.CellVariable(mesh = mesh, name=r'$k_p(x)$', value = np.exp(-((self.center[0]-mesh.cellCenters[0])**2 + (self.center[1]-mesh.cellCenters[1])**2 + (self.center[2]-mesh.cellCenters[2])**2)/self.std**2))

    def production_rate(self, phi):

        return self.k_p_x*(phi - self.phi_threshold)*(phi > self.phi_threshold)

    def production_rate_hill_gaussian(self, phi):

        return self.k_p_x*(phi**self.n_hill)*(phi**self.n_hill + self.phi_threshold**self.n_hill)**(-1)

    def production_rate_flat_in_space(self, phi):

        return self.k_p_max*(phi - self.phi_threshold)*(phi > self.phi_threshold)

    def production_rate_thresholded(self, phi):

        return self.k_p_max*(phi > self.phi_threshold)

    def production_rate_no_concentration_dependence(self):

        return self.k_p_x

    def degradation_rate(self, phi):

        return self.k_d*phi


# Heaviside step function to ensure that the fluxes are non-zero only when the species concentrations are non zero

def heaviside_limit_flux(phi_var):
    heaviside_multiplier = np.ones(len(phi_var.value))
    heaviside_multiplier[phi_var <= 0.0] = 0.0
    return heaviside_multiplier