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

[1] Noguez, C.; Roma ́n-Vela ́zquez, C. E. Phys. ReV. B Condens. Matt. 2004, 70 (19), 195412.
[2] Link, S.; El-Sayed, M. A. Int. ReV. Phys. Chem. 2000, 19 (3), 409– 453.
[3] Kittel, C. Introduction to Solid State Physics, 7th ed.; John Wiley and Sons: New York, 1996.
[4] Smith, N. V.; Spicer, W. E. Phys. ReV. 1969, 188 (2), 593.
[5] Blaber, M. G.; Arnold, M. D.; Ford, M. J., J. Phys. Chem. C 2009, 113, 8, 3041–3045

"""


import pathlib
import matplotlib.pyplot as plt

import numpy as np
import scipy.constants

import SFP_dielectrics

mgb_colors = ["#A42828", "#A46628", "#A4A428", "#66A428",
              "#28A428", "#28A466", "#28A4A4", "#2866A4",
              "#2828A4", "#6628A4", "#A428A4", "#A42866"]


def ev_to_nm(ev):
    # 1239.841984 nm per ev
    ev = np.array(ev)
    return 1.e9 * scipy.constants.physical_constants['inverse meter-electron volt relationship'][0]/ev


def nm_to_ev(nm):
    # 1239.841984 nm per
    nm = np.array(nm)
    return scipy.constants.physical_constants['inverse meter-electron volt relationship'][0]/(nm/1.e9)


def plot_drude(wp=4, gamma=0.4):

    w = np.arange(0.5, int(wp+2), 0.1)
    eps = SFP_dielectrics.drude_ev(w, wp, gamma)
    eps_real = np.real(eps)
    eps_imag = np.imag(eps)

    # plt.title(r'$\alpha > \beta$')
    fig = plt.figure()

    ax1 = fig.add_subplot(111)
    ax1.set_xlabel(r'$\omega$(eV)')
    ax1.set_ylabel(r'Dielectric Function, $\epsilon(\omega)$ (eV)')

    ax1.axhline(y=0, color='k', linestyle="--")
    ax1.axvline(x=wp, color='k', linestyle="--")

    ax1.set_xlim(np.min(w), np.max(w))
    ax1.set_xticks([x for x in range(1, int(np.round(np.max(w)+1)))])

    secax = ax1.secondary_xaxis('top', functions=(nm_to_ev, ev_to_nm))
    secax.set_xlabel(r'$\lambda$(nm)')
    secax.set_xticks([1240, 620, 413, 310, 248, 207, 177, 155, 138, 124])

    line, = ax1.plot(w, eps_real, lw=2)

    plt.show()
    return fig


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

        eps.append(SFP_dielectrics.drude_ev(w, wp_i, gamma_i))

        eps_real.append(np.real(eps[-1]))
        eps_imag.append(np.imag(eps[-1]))
        line, = ax2.plot(w, eps_real[-1], lw=2, label=label_i, color=lc)
        line, = ax1.plot(w, eps_imag[-1], lw=2, ls="--", label=label_i, color=lc)

    ax2.set_xlabel(r'$\omega$(eV)')

    plt.legend(loc="lower right")
    
    plt.savefig("Drude_complex_permittivity_of_good_metals.svg")
    plt.show()

    return fig


if __name__ == "__main__":
    plot_folder = pathlib.Path("plots/")
    plot_folder.mkdir(parents=True, exist_ok=True)

    #fig = plot_drude(3.72, 0.0184)
    #fig = plot_drude(5.71, 0.0276)
    fig = plot_multi_drude(wp=[3.72, 5.71, 8.55, 15.3], 
                           gamma=[0.0184, 0.0276, 0.0184, 0.598], 
                           label=[r'K, $\omega_p=3.72, \gamma=0.018$',
                                   r'Na, $\omega_p=5.71, \gamma=0.028$',
                                   r'Au, $\omega_p=8.55, \gamma=0.018$',
                                   r'Al, $\omega_p=15.3, \gamma=0.598$'],
                           xmax=10,real_ylim=-20,imag_ylim=5)


    plot_folder = pathlib.Path("plots/")
    plot_folder.mkdir(parents=True, exist_ok=True)

    #fig = plot_drude(3.72, 0.0184)
    #fig = plot_drude(5.71, 0.0276)
    #fig = plot_drude_vs_leru()



