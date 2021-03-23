#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
################################################################################
#   Copyright M. G. Blaber 2011
#   This file is part of SFP = Street Fighting Plasmonics
#   Street Fighting Plasmonics is free software: you can redistribute it
#   and/or modify it under the terms of the GNU General Public License as
#   published by the Free Software Foundation, either version 3 of the
#   License, or (at your option) any later version.
#
#   SFP is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with SFP.  If not, see <http://www.gnu.org/licenses/>.
################################################################################


31 [1] Noguez, C.; Roma ́n-Vela ́zquez, C. E. Phys. ReV. B Condens. Matt. 2004, 70 (19), 195412.
32 [2] Link, S.; El-Sayed, M. A. Int. ReV. Phys. Chem. 2000, 19 (3), 409– 453.
33 [3] Kittel, C. Introduction to Solid State Physics, 7th ed.; John Wiley and Sons: New York, 1996.
34 [4] LaVilla, R.; Mendlowitz, H. Phys. ReV. Lett. 1962, 9 (4), 149.
35 [4] Smith, N. V.; Spicer, W. E. Phys. ReV. 1969, 188 (2), 593.
[5] Blaber, M. G.; Arnold, M. D.; Ford, M. J., J. Phys. Chem. C 2009, 113, 8, 3041–3045

"""

import pathlib
import matplotlib.pyplot as plt

import numpy as np
import scipy.constants

mgb_colors = ["#A42828", "#A46628", "#A4A428", "#66A428",
              "#28A428", "#28A466", "#28A4A4", "#2866A4",
              "#2828A4", "#6628A4", "#A428A4", "#A42866"]

drude_parameters = []
drude_parameters['Na']={'omega_p' : 5.71,
                        'gamma' : 0.0276,
                        'Ref wp' : 'Kittel, C. Introduction to Solid State Physics, 7th ed.; John Wiley and Sons: New York, 1996.',
                        'Ref gamma' : 'Smith, N. V.; Spicer, W. E. Phys. ReV. 1969, 188 (2), 593.'}                      
drude_parameters['K']={'omega_p' : 3.72,
                       'gamma' : 0.0184,
                       'Ref wp' : 'Kittel, C. Introduction to Solid State Physics, 7th ed.; John Wiley and Sons: New York, 1996',
                       'Ref gamma' : 'Smith, N. V.; Spicer, W. E. Phys. ReV. 1969, 188 (2), 593.'}
drude_parameters['Al']={'omega_p' : 15.3,
                       'gamma' : 0.5984,
                       'Ref wp' : 'Kittel, C. Introduction to Solid State Physics, 7th ed.; John Wiley and Sons: New York, 1996.',
                       'Ref gamma' : 'LaVilla, R.; Mendlowitz, H. Phys. ReV. Lett. 1962, 9 (4), 149.'}
drude_parameters['Ag']={'omega_p' : 9.6,
                       'gamma' : 0.0228,
                       'Ref wp' : 'Noguez, C.; Roma ́n-Vela ́zquez, C. E. Phys. ReV. B Condens. Matt. 2004, 70 (19), 195412.',
                       'Ref gamma' : 'Link, S.; El-Sayed, M. A. Int. ReV. Phys. Chem. 2000, 19 (3), 409– 453.'}
drude_parameters['Au']={'omega_p' : 8.55,
                       'gamma' : 0.0184,
                       'Ref wp' : 'Noguez, C.; Roma ́n-Vela ́zquez, C. E. Phys. ReV. B Condens. Matt. 2004, 70 (19), 195412.',
                       'Ref gamma' : 'Link, S.; El-Sayed, M. A. Int. ReV. Phys. Chem. 2000, 19 (3), 409– 453.'}


def drude_energy(w, wp, gamma, einf=1, return_as_2d_array=False):
    """
    Calculates the complex drude dielectric function in energy (or frequency) units (J, eV, cm^-1)
                      wp^2                wp^2         i w g
    eps = einf - ----------- = einf -  --------- + -------------
                 w (w + i g)           w^2 + g^2   w (w^2 + g^2)
                 
    Where:
        eps = epsilon = permittivity = dielectric function
        einf = Over a fixed frequency range, this parameter can be used to account for the contri-
               bution of high frequency electronic transitions. It can be complex.
    Args:
        w (float): w = omega = frequency or energy of interest
        wp (float): wp = omega_p = Plasma frequency in energy or frequency units
        gamma (float): g = gamma = scattering frequency or equivalent energy.
        einf (float or complex, optional): DESCRIPTION. see above
        return_as_2d_array (TYPE, optional): When you provide an array of input w's, the code will
                                             return an array of complex numbers. If you would like
                                             to return a 2D list of floats, where the real and
                                             imaginary numbers make up D2, then use this option.
    Returns: Permittivity
        TYPE: float if input is a float. array if input is a list or array.

    """
    w = np.array(w)
    ii = complex(0., 1.)
    array_of_complex = einf - ( (wp**2) / (w * (w + ii * gamma)) )

    if return_as_2d_array:
        array_2d = np.stack((array_of_complex.real, array_of_complex.imag), -1)
        return array_2d
    else:
        return array_of_complex


def ev_to_nm(energy_in_ev):
    """
    Converts wavelength in nanometers to energy in electron-volts

    Args:
        energy_in_ev (float): electron-volts.

    Returns: 
        float: wavelngth in nm 

    """
    # 1239.841984 nm per ev
    ev = np.array(energy_in_ev)
    return 1.e9 * scipy.constants.physical_constants['inverse meter-electron volt relationship'][0]/ev


def nm_to_ev(wavelength_in_nm):
    """
    Converts energy in electron-volts to wavelength in nanometers
        Energy (eV) = 1239.841984 / Wavelength (nm) 
    Args:
        wavelength_in_nm (float): as the name suggests.

    Returns: 
        float: energy_in_ev : electron-volts.

    """
    nm = np.array(wavelength_in_nm)
    return scipy.constants.physical_constants['inverse meter-electron volt relationship'][0]/(nm/1.e9)


def plot_multi_drude(wp=[3.72, 5.71, 8.55, 15.3], gamma=[0.0184, 0.0276, 0.0184, 0.598], 
                     label=None, xmax=None,real_ylim=None, imag_ylim=None):
    
    max_wp = np.max(wp)
    
    if xmax == None:
        xmax=int(max_wp+1)
    
    if label == None:
        label = [None]*len(wp)

    w = np.arange(0.5, int(max_wp+2), 0.1)

    # plt.title(r'$\alpha > \beta$')
    fig = plt.figure(figsize=(12/2.54, 15/2.54))

    #Top Plot: Imag Permittivity
    ax1 = fig.add_subplot(211) 

    ax1.set_xticks([x for x in range(1, xmax+2)])
    ax1.set_xlim(np.min(w), xmax)
    
    ax1.set_yscale('log')
    ax1.set_ylim(0.01,10)
    ax1.set_yticks([0.001,0.01,0.1,1,10])
    ax1.set_yticklabels([r'$10^{-3}$','0.01','0.1','1','10'])
    ax1.set_ylabel(r'Imaginary Permittivity, Im($\epsilon$)')
    
    #if imag_ylim is not None: 
    #    ax1.set_ylim(top=imag_ylim)
   
    #ax1.set_xscale('log')
    #ax1.set_ylabel(r'Dielectric Function, $\epsilon(\omega)$ (eV)')
    
    secax = ax1.secondary_xaxis('top', functions=(nm_to_ev, ev_to_nm))
    secax.set_xlabel(r'$\lambda$(nm)')
    secax.set_xticks([1240, 620, 413, 310, 248, 207, 177, 155, 138, 124])

    # Axis 2
    ax2 = fig.add_subplot(212, sharex = ax1)
    ax2.axhline(y=0, color='k', linestyle="--")
    if real_ylim is not None: 
        ax2.set_ylim(bottom=real_ylim)
        
    ax2.set_ylabel(r'Real Permittivity, Re($\epsilon$)')

    eps = []
    eps_real = []
    eps_imag = []

    for wp_i, gamma_i, label_i, lc in zip(wp, gamma, label, mgb_colors[::2]):

        eps.append(drude_energy(w, wp_i, gamma_i))

        eps_real.append(np.real(eps[-1]))
        eps_imag.append(np.imag(eps[-1]))
        line, = ax2.plot(w, eps_real[-1], lw=2, label=label_i, color=lc)
        line, = ax1.plot(w, eps_imag[-1], lw=2, ls="--", label=label_i, color=lc)

    ax2.set_xlabel(r'$\omega$(eV)')

    plt.legend(loc="lower right")
    

    return fig


if __name__ == "__main__":
    
    drude_parameters['Na']
    
    fig = plot_multi_drude(wp=[3.72, 5.71, 8.55, 15.3], 
                           gamma=[0.0184, 0.0276, 0.0184, 0.598], 
                           label=[r'K, $\omega_p=3.72, \gamma=0.018$',
                                   r'Na, $\omega_p=5.71, \gamma=0.028$',
                                   r'Au, $\omega_p=8.55, \gamma=0.018$',
                                   r'Al, $\omega_p=15.3, \gamma=0.598$'],
                           xmax=10,real_ylim=-20,imag_ylim=5,
                           )
    
    plt.savefig("Drude Permittivity of K, Na, Au and Al.svg")
    plt.show()

    plot_folder = pathlib.Path("plots/")
    plot_folder.mkdir(parents=True, exist_ok=True)



