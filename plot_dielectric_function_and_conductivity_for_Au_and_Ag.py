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


"""

import pathlib
import matplotlib.pyplot as plt

import numpy as np
import scipy.constants

mgb_colors = ["#A42828", "#A46628", "#A4A428", "#66A428",
              "#28A428", "#28A466", "#28A4A4", "#2866A4",
              "#2828A4", "#6628A4", "#A428A4", "#A42866"]

optical_parameters={}
optical_parameters['Au']={
    'References':[
        {
            'Feature':'Model',
            'Reference':'P. G. Etchegoin, E. C. Le Ru, and M. Meyer, J. Chem. Phys. 127, 189901 (2007)' 
        },{
            'Feature':'Parameters',
            'Reference':'Martin G. Blaber, Unpublished.'
        },{
            'Feature':'Data',
            'Reference':'D. W. Lynch, W. R. Hunter in "Handbook of Optical Constants of Solids"; E. D. Palik, Ed.; Academic Press: 1985; Vol. 1.'
    }],
    'Oscillators':[
        {
            'type':'Drude_Wave',
            'name':'Drude',
            'einf' : 1.786232666464,
            'einfi' : 0.474161581212655,
            'lambda_p' : 135.865937688189,
            'gamma_p' : 16088.906016254           
        },{
            'type':'Leru_Oscillator',
            'name':'Peak 1',
            'amplitude': -0.789833578871083,
            'lambda': 314.245873751911,
            'gamma': 1469.6864723325,
            'phase': -9.84007589746256
        },{ 
            'type':'Leru_Oscillator',
            'name':'Peak 2',
            'amplitude': 1.0499806855811,
            'lambda': 463.507318325189,
            'gamma': 2290.02237298882,
            'phase': -0.951356810003336
        },{
            'type':'Leru_Oscillator',
            'name':'Peak 3',
            'amplitude': -0.789447146394695,
            'lambda': 156.777953986622,
            'gamma': 283.587000505755,
            'phase': 9.29750037444383
        },{
            'type':'Leru_Oscillator',
            'name':'Peak 4',
            'amplitude': 0.0973282595247586,
            'lambda': 57.9550132461995,
            'gamma': 497.421718909885,
            'phase': 6.44681517896198
        }]
    }


def evaluate_oscillators_wavelength(wavelength, parameter_dictionary):            
    
    wavelength = np.array(wavelength)
    permittivity = np.zeros_like(wavelength,dtype=complex)
    
    for oscillator_parameters in parameter_dictionary['Oscillators']:
        permittivity += evaluate_one_oscillator_wavelength(wavelength, oscillator_parameters)
    
    return permittivity

def evaluate_oscillators_energy(energy_in_ev, parameter_dictionary):            
    wavelength_in_nm = ev_to_nm(energy_in_ev)
    return evaluate_oscillators_wavelength(wavelength_in_nm, parameter_dictionary)
    
def evaluate_one_oscillator_wavelength(wavelength, oscillator_parameters):            
    
    if oscillator_parameters['type']=='Drude_Wave':
        return drude_wave(wavelength, oscillator_parameters)
    elif oscillator_parameters['type']=='Leru_Oscillator':
        return leru_oscillator(wavelength, oscillator_parameters)
    else:
        raise NotImplementedError("Invalid oscillator type",oscillator_parameters['type'])
    
def evaluate_one_oscillator_energy(energy_in_ev, parameter_dictionary):            
    wavelength_in_nm = ev_to_nm(energy_in_ev)
    return evaluate_one_oscillator_wavelength(wavelength_in_nm, parameter_dictionary)    
    
def drude_wave(wavelength, oscillator_parameters):

    ii = complex(0., 1.)
    
    l = np.array(wavelength)
    
    eps = oscillator_parameters['einf']
    eps_i = oscillator_parameters['einfi']
    lp = oscillator_parameters['lambda_p']
    gp = oscillator_parameters['gamma_p']

    # Drude Part and Total
    permittivty = eps + ii*eps_i - 1.00 / ((lp**2.0)*(l**-2.0 + ii/(gp*l)))

    return permittivty    
    
def leru_oscillator(wavelength, oscillator_parameters):
    
    ii = complex(0., 1.)
    
    l = np.array(wavelength)
    
    a0 = oscillator_parameters['amplitude']
    l0 = oscillator_parameters['lambda']
    g0 = oscillator_parameters['gamma']
    p0 = oscillator_parameters['phase']   
    
    iba = np.exp(ii*p0) / (1.0/l0 - 1.0/l - ii/g0)
    ibb = np.exp(-ii*p0) / (1.0/l0 + 1.0/l + ii/g0)
    permittivity = (a0/l0)*(iba+ibb)
    
    return permittivity
    

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


def plot_decomposed_oscillator_model(parameter_dictionary,
                                     min_w=0.8,max_w=6,
                                     xmax=6,epsr_ylim=None):


    w = np.arange(min_w, max_w, 0.1)

    # plt.title(r'$\alpha > \beta$')
    fig = plt.figure(figsize=(12/2.54, 15/2.54))

    #Top Plot: Imag Permittivity
    ax1 = fig.add_subplot(211) 

    ax1.set_xticks([x for x in range(1, xmax+2)])
    ax1.set_xlim(np.min(w), xmax)
    
    #ax1.set_yscale('log')
    ax1.set_ylim(-1,10)
    #ax1.set_ylim(0.01,10)
    #ax1.set_yticks([0.001,0.01,0.1,1,10])
    #ax1.set_yticklabels([r'$10^{-3}$','0.01','0.1','1','10'])
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
    if epsr_ylim is not None: 
        ax2.set_ylim(top=10, bottom=epsr_ylim)
        
    ax2.set_ylabel(r'Real Permittivity, Re($\epsilon$)')

    p = parameter_dictionary
    oscs = p['Oscillators']
    
    eps=[]
    eps_real = []
    eps_imag = []

    for osc, lc in zip(oscs, mgb_colors[::2]):
        eps.append(evaluate_one_oscillator_energy(w,osc))
        eps_real.append(np.real(eps[-1]))
        eps_imag.append(np.imag(eps[-1]))
        
        line, = ax2.plot(w, eps_real[-1], lw=2, ls="--", label=osc['name'], color=lc)
        line, = ax1.plot(w, eps_imag[-1], lw=2, ls="--", label=osc['name'], color=lc)
        
    eps_all = evaluate_oscillators_energy(w, p)
    eps_all_real = np.real(eps_all)
    eps_all_imag = np.imag(eps_all)
    
    line, = ax2.plot(w, eps_all_real, lw=2, ls="-", color='k')
    line, = ax1.plot(w, eps_all_imag, lw=2, ls="-", color='k')
        
    
    ax2.set_xlabel(r'$\omega$(eV)')

    plt.legend(loc="lower right")
    

    return fig
    
    

if __name__ == "__main__":
    
    op = optical_parameters['Au']
    
    eo_ref = complex(-2.907354262668494+3.2789154844594184j)
    eo = evaluate_oscillators_wavelength(500.0, op)
    print(eo,eo - eo_ref)
    
    if True:
        fig = plot_decomposed_oscillator_model(op,epsr_ylim=-30)
        
        plt.savefig("Permittivity of Au.svg")
        plt.savefig("Permittivity of Au.png",dpi=600)
        plt.show()
        

