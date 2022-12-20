import argparse
import re
import pandas as pd
import h5py
import os
import moviepy.editor as mp
import numpy as np

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

from matplotlib import rc
rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)
plt.rcParams['xtick.labelsize']=20
plt.rcParams['ytick.labelsize']=20
plt.rcParams['font.size']=20
plt.rcParams["text.usetex"]=True

def write_movies(PATH, hdf5file_name, x, y, fps=5):
	
	def key_funct(x):
		return int(x.split('_')[-4].rstrip('.png'))

	# make directory
	try:
		os.mkdir(os.path.join(PATH, 'movies'))
	except:
		print("/movies directory already exists")
		
	with h5py.File(os.path.join(PATH, hdf5file_name), mode="r") as df_total:
	
		# find value to plot for max/min
		max_protein_val = -1e6
		min_protein_val = 1e6
		max_rna_val = -1e6
		min_rna_val = 1e6

		
		df_p = df_total['mu_p']
		df_r = df_total['mu_r']
		# df_m = df_total['phi_m']

		for i in range(df_p.shape[0]):
			min_p_val = np.min(df_p[i])
			min_protein_val = np.min([min_p_val, min_protein_val])
			min_r_val = np.min(df_r[i])
			min_rna_val = np.min([min_r_val, min_rna_val])
			max_p_val = np.max(df_p[i])
			max_protein_val = np.max([max_p_val, max_protein_val])
			max_r_val = np.max(df_r[i])
			max_rna_val = np.max([max_r_val, max_rna_val])


		
		for i in range(df_p.shape[0]):
			# plot and save individuals
			
			if np.all(df_p[i] == 0):
				break

			fig, ax = plt.subplots(1,2, figsize=(14,6))
			cs1 = ax[0].tricontourf(x, y, df_p[i], cmap='Blues', levels=np.linspace(min_protein_val,max_protein_val,100))
			cs2 = ax[1].tricontourf(x, y, df_r[i], cmap='Reds', levels=np.linspace(min_rna_val,max_rna_val,100))
			#cs3 = ax[2].tricontourf(x, y, df_m[i], levels=np.linspace(min_mrna_val,max_mrna_val,100), cmap='Reds')
			ax[0].tick_params(axis='both', which='major', labelsize=25)
			ax[1].tick_params(axis='both', which='major', labelsize=25)
			#ax[2].tick_params(axis='both', which='major', labelsize=20)

			cbar1 = fig.colorbar(cs1, ax=ax[0], ticks=np.linspace(min_protein_val,max_protein_val,6))
			cbar2 = fig.colorbar(cs2, ax=ax[1], ticks=np.linspace(min_rna_val,max_rna_val,6))
			#cbar3 = fig.colorbar(cs3, ax=ax[1], ticks=np.linspace(min_mrna_val,max_mrna_val,6))
			cbar1.ax.tick_params(labelsize=25)
			cbar2.ax.tick_params(labelsize=25)
			#cbar3.ax.tick_params(labelsize=20)
			
			ax[0].set_title(r'Protein chemical potential ($\mu_P$)', fontsize=20)
			ax[1].set_title(r'RNA chemical potential ($\mu_R$)', fontsize=20)
			#ax[2].set_title(r'mRNA concentration ($\phi_m$)', fontsize =20)

			fig.savefig(fname=PATH +'/movies/step_{step}_mRNA_lRNA_Protein.png'.format(step=i),dpi=300,format='png')
			plt.close(fig)
		
		file_names = sorted(list((fn for fn in os.listdir(os.path.join(PATH, 'movies')) if fn.endswith('mRNA_lRNA_Protein.png'))), key=key_funct)

		file_paths = [os.path.join(PATH, 'movies', f) for f in file_names]
		clips = []
		
		for file in file_paths:
			clips.append(mp.ImageClip(file).set_duration(2))
		
		concat_clip = mp.concatenate_videoclips(clips, method="compose")
		concat_clip.write_videofile(os.path.join(PATH, 'movies','chemical_potential_RNA_Protein.mp4'), fps=fps)
		concat_clip.close()

		# delete individual images
		for f in file_paths:
			os.remove(f)

if __name__ == "__main__":
	"""
		Function is called when python code is run on command line and calls the function that generates the
		movies from hdf5 files
		
		This script assumes that each directory containing a hdf5 file also contains another subdirectory called 
		/Mesh that contains the values of mesh coordinates
	"""
	parser = argparse.ArgumentParser(description='Take directory name to walk through and generate movies')
	parser.add_argument('--i',help="Root directory containing the hdf5 files", required = True)
	parser.add_argument('--f',help="Name of hdf5 file", required = True)
	parser.add_argument('--mf',help="Path to mesh file", required = True)
	# parser.add_argument('--pN',help="Parameter number from file (indexed from 1)", required = False);

	# parser.add_argument('--o',help="Name of output folder", required = True);
	args = parser.parse_args();

	base_path = args.i
	regex = re.compile(r'.*'+str(args.f))
	found_at_least_one = 0

	for root, dirs, files in os.walk(base_path): # Loop through all subdirectories in the directory "base_path"

		for fi in files: # Loop through all files in each subdirectory to see if we have a hdf5 file with the name given in args.f
			
			match = re.search(regex, fi)

			if match != None: 		# Found a hdf5 file!

				found_at_least_one = 1
				
				# Read mesh coordinates:
				df = pd.read_csv(root + '/' + args.mf , '\t')
				x = df.values[:,0]
				y = df.values[:,1]
				write_movies(root, args.f, x, y, fps = 3)

	if not found_at_least_one:
		print('Could not find any hdf5 files in the supplied directory!')

