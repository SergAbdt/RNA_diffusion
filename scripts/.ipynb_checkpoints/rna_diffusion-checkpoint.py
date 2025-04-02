#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main function that is called to initialize and run phase-field dynamics
"""

from __future__ import print_function
import fipy as fp
import numpy as np
from scipy.spatial import KDTree

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse
import os
import sys


sys.path.append(os.path.dirname(os.path.abspath('rna_diffusion.py')))

from utils.free_energy import *
from utils.graphics import *
from utils.graphics_3d import *
from utils.input_parse import *
from utils.mesh_generation import *
from utils.utils import *
from utils.write_files import *


def run_CH(args):
    """
    Function takes in path to input params, output_folder, and optionally params file that
    are passed while running code. With these parameters, this function initializes and runs
    phase-field simulations while writing output to files.

    **Input variables**

    -   args.i = Path to input params files (required)
    -   args.o = path to the first prefix of the output folder (required)
    -   args.r = path to the second prefix of the output folder (optional)
    -   args.p = path to parameter file (optional)
    -   args.pN = Nth parameter to use from input (optional)

    """

    # Read parameters from the input_parameters.txt file
    input_parameters = input_parse(args.i)

    # Define spatial variable value limits for plots
    val_lim = {}
    for spvar in ["phi_p", "phi_r", "phi_m", "mu_p", "mu_r", "mu_m", "free_energy"]:
        val_lim[spvar] = (input_parameters[f'{spvar}_min_plot'], input_parameters[f'{spvar}_max_plot'])

    # Read the parameters in the param_list.txt file and choose the appropriate parameter to change
    if args.p:
        params = input_parse(args.p, params_flag=True)
        par_name = str(list(params.keys())[0])
        par_values = params[par_name]   
    
    if args.pN:
        par_values = par_values[int(args.pN)-1]
        input_parameters[par_name] = par_values 

    # Create output dir:
    p_o = lambda x, i=input_parameters: x + "_" + str(i[x]) + "_"
    out_params = ['n_cells', 'sigma_chr', 'c_RNAchr', 'sigma_RNAchr', 'k_p_max_lncRNA'] # Modify to add parametes to the output dir name
    output_dir = (input_parameters['output_dir'] + args.o + '/' +
                  args.r + '_'*bool(args.r) + '_'.join(f'{x}_{str(input_parameters[x])}' for x in out_params))
        
    if not os.path.exists(output_dir): # exists_ok parameter is not used for python2.7 compatibility
        os.makedirs(output_dir)
        
    write_input_params(output_dir + '/input_params.txt', input_parameters)
    
    print(f'Results: {output_dir}')

    # Make a directory to store the image files of concentration profiles
    if int(input_parameters['plot_flag']) and not os.path.exists(output_dir + '/Images/'):
        os.makedirs(output_dir + '/Images/')
    
    # Define the mesh
    dim = input_parameters['dimension']
    sparse = input_parameters['sparse']
    chrom_f = input_parameters['chrom_f']
    chrom_size = input_parameters['chrom_size']
    space_scale = input_parameters['space_scale']
    min_size = input_parameters['min_size']
    scale = input_parameters['scale']
    sigma_chr = input_parameters['sigma_chr']
    
    mesh_f = f"{output_dir.rsplit('/',1)[0]}/mesh_min_size_{min_size}_scale_{scale}_sigma_chr_{sigma_chr}.msh2"
    
    if dim==2:
        bin_r = np.asarray([(5*np.cos(2*np.pi/9*x), 5*np.sin(2*np.pi/9*x)) for x in range(9)])
        rna_nucleus_location = bin_r[1]
        if not os.path.exists(mesh_f):
            chrom_2d(bin_r, space_size, min_size, scale, sigma_chr, mesh_f)
        mesh = fp.Gmsh2D(mesh_f)
        
    elif dim==3:
        bin_r, bin_r_spline = parse_chrom(chrom_f, chrom_size, space_scale)
        rna_nucleus_location = bin_r[int(input_parameters['lncRNA_gene_bin'])]
        if not sparse:
            bin_r_spline = lin_interp(bin_r, 10)
        if not os.path.exists(mesh_f):
            if not sparse:
                chrom_3d(bin_r_spline, min_size, scale, sigma_chr, mesh_f, space_scale=space_scale, spheres=1, cylinders=1, splines=1)
            else:
                chrom_3d_sparse(bin_r_spline, min_size, scale, sigma_chr, mesh_f, space_scale=space_scale)
        mesh = fp.Gmsh3D(mesh_f)

    # Define a CellVariable that store the distance to chromatin for the cellCenters and write it to a file (if not exists)
    d_chr = dist_to_chrom(mesh, sparse, bin_r_spline, sigma_chr, space_scale)
    dist_f = f"{output_dir.rsplit('/',1)[0]}/dist_min_size_{min_size}_scale_{scale}_sigma_chr_{sigma_chr}.hdf5"
    write_dist(dist_f, d_chr, bin_r)
    
    # Define the CellVariables that store the concentrations of lncRNA and Protein within the small control volumes
    phi_p = fp.CellVariable(mesh=mesh, name=r'$\phi_{prot}$', hasOld=True, value = input_parameters['phi_p_0'])
    phi_r = fp.CellVariable(mesh=mesh, name=r'$\phi_{lncRNA}$', hasOld=True, value = input_parameters['phi_r_0'])

    # Define the free energy class object
    FE = free_energy_RNA_Chrom_FH(
                     NP=input_parameters['NP'], 
                     NR=input_parameters['NR'],
                     chi_p=input_parameters['chi_p'],
                     chi_pr=input_parameters['chi_pr'], 
                     chi_r=input_parameters['chi_r'],
                     c_RNAchr = input_parameters['c_RNAchr'],
                     sigma_RNAchr = input_parameters['sigma_RNAchr'],
                     wall = input_parameters['wall'],
                     wall_k = input_parameters['wall_k'],
                     neg_max = input_parameters['neg_max'])

    specific_RNAchr = FE.specific_DH(mesh, d_chr) if not sparse else FE.specific_RNAchr(mesh, d_chr)

    # Define parameters associated with the numerical method to solve the PDEs
    t = fp.Variable(0.0)
    dt = float(input_parameters['dt'])
    dt_max = float(input_parameters['dt_max'])
    dt_min = float(input_parameters['dt_min'])
    min_change = float(input_parameters['min_change'])
    max_change = float(input_parameters['max_change'])
    total_steps = int(input_parameters['total_steps'])
    duration = input_parameters['duration'];
    time_step = fp.Variable(dt)

    # Define the form of the PDEs
    D_protein = float(input_parameters['D_protein'])
    D_rna = float(input_parameters['D_rna'])

    eqn0 = (fp.TransientTerm(coeff=1.,var=phi_p)
            == fp.DiffusionTerm(coeff=D_protein*FE.dmu_p_dphi_p(phi_p,phi_r), var=phi_p)
            + fp.DiffusionTerm(coeff=D_protein*FE.dmu_p_dphi_r(phi_p,phi_r), var=phi_r))

    if input_parameters['lncRNA_reactions_flag']:
        lncRNA_reactions = RNA_reactions(mesh=mesh, sparse=sparse, k_p_max=input_parameters['k_p_max_lncRNA'], k_degradation=input_parameters['k_degradation_lncRNA'],
                                         spread=input_parameters['spread_kp_lncRNA'], center=rna_nucleus_location, sigma_chr=sigma_chr,
                                         phi_threshold = input_parameters['protein_threshold_lncRNA_production'])
        
        eqn1 = (fp.TransientTerm(coeff=1.,var=phi_r)
                == fp.DiffusionTerm(coeff=D_rna*FE.dmu_r_dphi_p(phi_p,phi_r), var=phi_p)
                + fp.DiffusionTerm(coeff=D_rna*FE.dmu_r_dphi_r(phi_p,phi_r), var=phi_r)
                + fp.PowerLawConvectionTerm(coeff=D_rna*specific_RNAchr.grad, var=phi_r)
                + lncRNA_reactions.production_rate(phi_p)
                - lncRNA_reactions.degradation_rate(phi_r))
        
    else:
        eqn1 = (fp.TransientTerm(coeff=1.,var=phi_r)
                == fp.DiffusionTerm(coeff=D_rna*FE.dmu_r_dphi_p(phi_p,phi_r), var=phi_p)
                + fp.DiffusionTerm(coeff=D_rna*FE.dmu_r_dphi_r(phi_p,phi_r), var=phi_r)
                + fp.PowerLawConvectionTerm(coeff=D_rna*specific_RNAchr.grad, var=phi_r))

    # Loop over time and solve the PDE
    max_sweeps = int(input_parameters['max_sweeps'])

    elapsed = 0.0
    steps = 0 
    
    phi_p.updateOld()
    phi_r.updateOld()

    if input_parameters['mrna_flag']:
        phi_m.updateOld()

    while (elapsed <= duration) and (steps <= total_steps) and (dt > dt_min):
                
#        assert max(phi_r+phi_m) < phi_r_max, "Phi_r value surpassed 1.0. Aborting due to inaccurate approximations"  
        
        sweeps = 0
        
        while sweeps < max_sweeps:
            res1 = eqn0.sweep(dt=dt)
            res2 = eqn1.sweep(dt=dt)
            if input_parameters['mrna_flag']:
                res3 = eqn2.sweep(dt=dt)
            sweeps += 1        
            
        if input_parameters['mrna_flag']:
            delta_state = np.max([np.abs(np.max((phi_p-phi_p.old).value)),np.abs(np.max((phi_r-phi_r.old).value)),np.abs(np.max((phi_m-phi_m.old).value))])
        else:
            delta_state = np.max([np.abs(np.max((phi_p-phi_p.old).value)),np.abs(np.max((phi_r-phi_r.old).value))])
        
        # Write out simulation data to text files
        if steps % input_parameters['text_log'] == 0:
            
            # Write some simulation statistics for every "text_log" time steps to a text file
            write_stats(t=t.value, dt=dt, steps=steps, phi_p=phi_p, phi_r=phi_r, phi_m=None, mesh=mesh, FE=FE, res = (res1+res2)/2, delta_s= delta_state,  output_dir=output_dir, d_chr=d_chr)
            
        # Making figures and storing simulation data relevant to making figures
        if steps % input_parameters['image_checkpoint'] == 0:
            
            # Create image files containing concentration profiles of the species
            if (dim == 2) and (int(input_parameters['plot_flag'])):
                plot_spvars(dim=dim, mesh=mesh, sparse=sparse, spatial_variable=phi_p, variable_name='phi_p',
                                       colormap="Blues", output_dir=f'{output_dir}/Images', steps=steps,
                                       bin_r=bin_r_spline)
                plot_spvars(dim=dim, mesh=mesh, sparse=sparse, spatial_variable=phi_r, variable_name='phi_r',
                                       colormap="Reds", output_dir=f'{output_dir}/Images', steps=steps,
                                       bin_r=bin_r_spline)

            # Write spatial variables into a HDF5 file
            list_of_variables = write_spatial_vars_to_hdf5_file(phi_p=phi_p, phi_r=phi_r, phi_m = None, 
                                    FE=FE, output_dir=output_dir, 
                                    recorded_step=int(steps/input_parameters['image_checkpoint']), 
                                    total_recorded_steps=
                                    int(np.ceil(input_parameters['total_steps']/input_parameters['image_checkpoint']))+1,
                                    d_chr=d_chr)
            
        steps += 1
        elapsed += dt
        t.value = t.value+dt
        
        if delta_state > max_change:
            dt *= input_parameters['dt_down']
        else:
            dt *= input_parameters['dt_up']
            dt = min(dt, dt_max)

        if (delta_state/dt) < min_change:
            break
        
        time_step.value = dt;
        phi_p.updateOld()
        phi_r.updateOld()
        if input_parameters['mrna_flag']:
            phi_m.updateOld()


    if int(input_parameters['movie_flag']):
        if int(input_parameters['plot_flag']):
            write_movie_from_images(output_dir, list_of_variables[:2], fps=2)
        else:
            write_movie_from_hdf5(output_dir, list_of_variables, dim, mesh, fps=2, val_lim=val_lim, bin_r=bin_r_spline,
                                  plane_pos=plane_pos, thres=min_size/2, d_chr=d_chr.value, sigma_chr=sigma_chr)



if __name__ == "__main__":
    """
        Function is called when python code is run on command line and calls run_CH
        to initialize the simulation
    """
    parser = argparse.ArgumentParser(description='Take output filename to run CH simulations')
    parser.add_argument('--i',help="Name of the input params file", required = True)
    parser.add_argument('--o',help="Name the first prefix of the output folder", required = True)
    parser.add_argument('--r',help="Name the second prefix of the output folder", required = False)
    parser.add_argument('--p',help="Name of the parameter file", required = False)
    parser.add_argument('--pN',help="Parameter number from the parameter file (indexed from 1)", required = False)
    args = parser.parse_args()

    run_CH(args)
