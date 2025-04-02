import h5py
import os
import numpy as np


def write_dist(dist_f, d_chr, bin_r):
    if not os.path.exists(dist_f):
            with h5py.File(dist_f, 'a') as f:
                f.create_dataset("d_chr", (len(d_chr)))
                f.create_dataset("d_bin", (len(bin_r), len(d_chr)))
                f["d_chr"][:] = d_chr.value
                for bin_ in range(len(bin_r)):
                    f["d_bin"][bin_,:] = np.linalg.norm(mesh.cellCenters.value.T - bin_r[bin_], axis=1)


def sum_in_domain(mesh, var, mask):
    sum_ = np.sum(mesh.cellVolumes[mask]*var.value[mask])
    return sum_


def av_in_domain(mesh, var, mask):
    av = sum_in_domain(mesh, var, mask)/np.sum(mesh.cellVolumes[mask])
    return av


def write_stats(t, dt, steps, phi_p, phi_r, phi_m, mesh, FE, res, delta_s, output_dir, d_chr=None, chr_mask=None):
    
    if type(phi_m) == type(None):
        stats_list = ["step", "t", "dt","phi_p_avg","phi_p_min","phi_p_max","phi_r_avg","phi_r_min","phi_r_max","residuals","max_rate_of_change","free_energy"]
    else:
        stats_list = ["step", "t", "dt","phi_p_avg","phi_p_min","phi_p_max","phi_r_avg","phi_r_min","phi_r_max","phi_m_avg","phi_m_min","phi_m_max","residuals","max_rate_of_change","free_energy"]
    
    if steps == 0:
        with open(output_dir + "/stats.txt", 'w+') as stats:
            stats.write("\t".join(stats_list) + "\n")
            
    if type(phi_m) == type(None):
        fe = FE.f(phi_p,phi_r,) if d_chr is None else FE.f(phi_p,phi_r,d_chr)
        stats_simulation = ["{}".format(int(steps)),
                        "{:.8f}".format(t),
                        "{:.3e}".format(dt),
                        "{:.4e}".format(phi_p.cellVolumeAverage.value if chr_mask is None else av_in_domain(mesh, phi_p, ~chr_mask)),
                        "{:.4e}".format(min(phi_p)),
                        "{:.4e}".format(max(phi_p)),
                        "{:.4e}".format(phi_r.cellVolumeAverage.value if chr_mask is None else av_in_domain(mesh, phi_r, ~chr_mask)),
                        "{:.4e}".format(min(phi_r)),
                        "{:.4e}".format(max(phi_r)),
                        "{:.4e}".format(res),
                        "{:.4e}".format(delta_s),
                        "{:.4e}".format(np.sum((fe*mesh.cellVolumes).value if chr_mask is None else sum_in_domain(mesh, phi_p, ~chr_mask)))
                        ]
    else:
        fe = FE.f(phi_p,phi_r,phi_m) if d_chr is None else FE.f(phi_p,phi_r,phi_m,d_chr)
        stats_simulation = ["{}".format(int(steps)),
                        "{:.8f}".format(t),
                        "{:.3e}".format(dt),
                        "{:.4e}".format(phi_p.cellVolumeAverage.value if chr_mask is None else av_in_domain(mesh, phi_p, ~chr_mask)),
                        "{:.4e}".format(min(phi_p)),
                        "{:.4e}".format(max(phi_p)),
                        "{:.4e}".format(phi_r.cellVolumeAverage.value if chr_mask is None else av_in_domain(mesh, phi_r, ~chr_mask)),
                        "{:.4e}".format(min(phi_r)),
                        "{:.4e}".format(max(phi_r)),
                        "{:.4e}".format(phi_m.cellVolumeAverage.value if chr_mask is None else av_in_domain(mesh, phi_m, ~chr_mask)),
                        "{:.4e}".format(min(phi_m)),
                        "{:.4e}".format(max(phi_m)),
                        "{:.4e}".format(res),
                        "{:.4e}".format(delta_s),
                        "{:.4e}".format(np.sum((fe*mesh.cellVolumes).value if chr_mask is None else sum_in_domain(mesh, phi_p, ~chr_mask)))
                        ]
    
    assert len(stats_list) == len(stats_simulation), "Aborting as the number of fields in the simulation statistics file is different from the number of values supplied!"

    with open(output_dir + "/stats.txt", 'a') as stats:
        stats.write("\t".join(stats_simulation) + "\n")


def write_spatial_vars_to_hdf5_file(phi_p, phi_r, phi_m, FE, output_dir, recorded_step, total_recorded_steps, d_chr=None):
    """
    Function to dump simulation data into HDF5 files
    """

    if type(phi_m) == type(None):
        list_of_spatial_variables = ["phi_p", "phi_r", "mu_p", "mu_r", "free_energy"] 
    else:
        list_of_spatial_variables = ["phi_p", "phi_r", "phi_m", "mu_p", "mu_r", "mu_m", "free_energy"] 
    
    if recorded_step == 0:
        
        number_of_CVs = len(phi_p)   

        with h5py.File(os.path.join(output_dir + '/spatial_variables.hdf5'),'w') as f:
            for sv in list_of_spatial_variables:
                f.create_dataset(sv, (total_recorded_steps,number_of_CVs))
    
    with h5py.File(output_dir + '/spatial_variables.hdf5', 'a') as f:
        
        if type(phi_m) == type(None):
            f["phi_p"][recorded_step,:] = phi_p.value
            f["phi_r"][recorded_step,:] = phi_r.value            
            f["mu_p"][recorded_step,:] = FE.mu_p(phi_p, phi_r) if d_chr is None else FE.mu_p(phi_p, phi_r, d_chr)
            f["mu_r"][recorded_step,:] = FE.mu_r(phi_p, phi_r) if d_chr is None else FE.mu_r(phi_p, phi_r, d_chr)
            f["free_energy"][recorded_step,:] = FE.f(phi_p, phi_r) if d_chr is None else FE.f(phi_p, phi_r, d_chr)
            # f["det_J"][recorded_step,:] = FE.det_J(phi_p, phi_r, X_CV, Y_CV) 
        else:
            f["phi_p"][recorded_step,:] = phi_p.value
            f["phi_r"][recorded_step,:] = phi_r.value
            f["phi_m"][recorded_step,:] = phi_m.value            
            f["mu_p"][recorded_step,:] = FE.mu_p(phi_p, phi_r, phi_m) if d_chr is None else FE.mu_p(phi_p, phi_r, phi_m, d_chr)
            f["mu_r"][recorded_step,:] = FE.mu_r(phi_p, phi_r, phi_m) if d_chr is None else FE.mu_r(phi_p, phi_r, phi_m, d_chr)
            f["mu_m"][recorded_step,:] = FE.mu_m(phi_p, phi_r, phi_m) if d_chr is None else FE.mu_m(phi_p, phi_r, phi_m)
            f["free_energy"][recorded_step,:] = FE.f(phi_p, phi_r, phi_m) if d_chr is None else FE.f(phi_p, phi_r, phi_m, d_chr)
            # f["det_J"][recorded_step,:] = FE.det_J(phi_p, phi_r, phi_m, X_CV, Y_CV) 

    return list_of_spatial_variables


def write_stats_dynamics(t, dt, phi_p, phi_r, phi_m, X_CV, Y_CV, mesh, FE, output_dir, recorded_step, res, delta_s, input_parameters, d_chr=None):
    
    if type(phi_m) == type(None):
        list_of_dynamical_variables = ['t','dt','step','FE','min_protein_concentration','max_protein_concentration','average_protein_concentration','min_lncrna_concentration','max_lncrna_concentration','average_lncrna_concentration','average_protein_in_condensate', 'average_lncrna_in_condensate', 'condensate_area'] 
    else:
        list_of_dynamical_variables = ['t','dt','step','FE','min_protein_concentration','max_protein_concentration','average_protein_concentration','min_lncrna_concentration','max_lncrna_concentration','average_lncrna_concentration','min_mrna_concentration','max_mrna_concentration','average_mrna_concentration', 'average_protein_in_condensate', 'average_lncrna_in_condensate', 'average_mrna_in_condensate', 'condensate_area', 'total_amount_of_mrna', 'molar_rate_of_mrna_production', 'net_rate_of_mrna_increase'] 
    
    total_recorded_steps = int(input_parameters['total_steps']/input_parameters['text_log'])

    if recorded_step == 0:

        with h5py.File(os.path.join(output_dir + '/dynamical_variables.hdf5'),'w') as f:
            for sv in list_of_dynamical_variables:
                f.create_dataset(sv, (total_recorded_steps+1,1))
        
    recorded_step_index = int(recorded_step/input_parameters['text_log'])
    
    with h5py.File(output_dir + '/dynamical_variables.hdf5', 'a') as f:
        
        if type(phi_m) == type(None):
            fe = FE.f(phi_p,phi_r,X_CV,Y_CV) if d_chr is None else FE.f(phi_p,phi_r,X_CV,Y_CV,d_chr)
            f["t"][recorded_step_index] = t
            f["dt"][recorded_step_index] = dt            
            f["step"][recorded_step_index] = recorded_step
            f["FE"][recorded_step_index] = np.sum(fe.value)
            f["min_protein_concentration"][recorded_step_index] = min(phi_p)
            f["max_protein_concentration"][recorded_step_index] = max(phi_p)
            f["average_protein_concentration"][recorded_step_index] = phi_p.cellVolumeAverage.value
            f["min_lncrna_concentration"][recorded_step_index] = min(phi_r)
            f["max_lncrna_concentration"][recorded_step_index] = max(phi_r)
            f["average_lncrna_concentration"][recorded_step_index] = phi_r.cellVolumeAverage.value
            
            indices = phi_p.value > input_parameters['condensate_concentration_cutoff']
            
            if np.any(indices):
                f['average_protein_in_condensate'][recorded_step_index] = np.sum(phi_p.value[indices]*mesh.cellVolumes[indices])/np.sum(mesh.cellVolumes[indices])
                f['average_lncrna_in_condensate'][recorded_step_index] = np.sum(phi_r.value[indices]*mesh.cellVolumes[indices])/np.sum(mesh.cellVolumes[indices])
                if f['average_lncrna_in_condensate'][recorded_step_index] < 0.0:
                    f['average_lncrna_in_condensate'][recorded_step_index] = 0.0
                f['condensate_area'][recorded_step_index] = np.sum(mesh.cellVolumes[indices])
                
            else:
                f['average_protein_in_condensate'][recorded_step_index] = 0.0
                f['average_lncrna_in_condensate'][recorded_step_index] = 0.0
                f['condensate_area'][recorded_step_index] = 0.0

        else:
            fe = FE.f(phi_p,phi_r,phi_m,X_CV,Y_CV) if d_chr is None else FE.f(phi_p,phi_r,phi_m,X_CV,Y_CV,d_chr)
            f["t"][recorded_step_index] = t
            f["dt"][recorded_step_index] = dt            
            f["step"][recorded_step_index] = recorded_step
            f["FE"][recorded_step_index] = np.sum(fe.value)
            f["min_protein_concentration"][recorded_step_index] = min(phi_p)
            f["max_protein_concentration"][recorded_step_index] = max(phi_p)
            f["average_protein_concentration"][recorded_step_index] = phi_p.cellVolumeAverage.value
            f["min_lncrna_concentration"][recorded_step_index] = min(phi_r)
            f["max_lncrna_concentration"][recorded_step_index] = max(phi_r)
            f["average_lncrna_concentration"][recorded_step_index] = phi_r.cellVolumeAverage.value
            f["min_mrna_concentration"][recorded_step_index] = min(phi_m)
            f["max_mrna_concentration"][recorded_step_index] = max(phi_m)
            f["average_mrna_concentration"][recorded_step_index] = phi_m.cellVolumeAverage.value
            
            indices = phi_p.value > input_parameters['condensate_concentration_cutoff']
            if np.any(indices):
                f['average_protein_in_condensate'][recorded_step_index] = np.sum(phi_p.value[indices]*mesh.cellVolumes[indices])/np.sum(mesh.cellVolumes[indices])
                f['average_lncrna_in_condensate'][recorded_step_index] = np.sum(phi_r.value[indices]*mesh.cellVolumes[indices])/np.sum(mesh.cellVolumes[indices])
                if f['average_lncrna_in_condensate'][recorded_step_index] < 0.0:
                    f['average_lncrna_in_condensate'][recorded_step_index] = 0.0
                f['condensate_area'][recorded_step_index] = np.sum(mesh.cellVolumes[indices])
                f['average_mrna_in_condensate'][recorded_step_index] = np.sum(phi_m.value[indices]*mesh.cellVolumes[indices])/np.sum(mesh.cellVolumes[indices])
            else:
                f['average_protein_in_condensate'][recorded_step_index] = 0.0
                f['average_lncrna_in_condensate'][recorded_step_index] = 0.0
                f['condensate_area'][recorded_step_index] = 0.0
                f['average_mrna_in_condensate'][recorded_step_index] = 0.0
                
            f['total_amount_of_mrna'][recorded_step_index] = np.sum(phi_m.value*mesh.cellVolumes)
            if input_parameters['reaction_rate'] == 0.0:
                f['molar_rate_of_mrna_production'][recorded_step_index] =  np.sum((input_parameters['k_p_max']*(phi_p - input_parameters['protein_threshold_mRNA_production'])*(phi_p > input_parameters['protein_threshold_mRNA_production'])).value*mesh.cellVolumes)
                f['net_rate_of_mrna_increase'][recorded_step_index] =  np.sum((input_parameters['k_p_max']*(phi_p - input_parameters['protein_threshold_mRNA_production'])*(phi_p > input_parameters['protein_threshold_mRNA_production'])-input_parameters['k_degradation']*phi_m).value*mesh.cellVolumes)
            elif input_parameters['reaction_rate'] == 1.0:
                f['molar_rate_of_mrna_production'][recorded_step_index] =  np.sum((input_parameters['k_p_max']*(phi_p > input_parameters['protein_threshold_mRNA_production'])).value*mesh.cellVolumes)
                f['net_rate_of_mrna_increase'][recorded_step_index] =  np.sum((input_parameters['k_p_max']*(phi_p > input_parameters['protein_threshold_mRNA_production'])-input_parameters['k_degradation']*phi_m).value*mesh.cellVolumes)