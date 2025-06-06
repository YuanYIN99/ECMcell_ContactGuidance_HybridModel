import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
#import seaborn as sns
import pickle
from numpy import linalg as LA
from matplotlib.patches import Circle
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import os
from numpy.linalg import norm 
import itertools
from scipy.stats import entropy
from scipy.spatial import distance
from scipy.spatial import Delaunay
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import itertools
from scipy.interpolate import griddata


def cellcell_dis_orien(cell_coords, y_len, x_len, y_periodic_bool, x_periodic_bool):
    ''' 
    This function finds the pairwise distance and orientation bewtween cells.
    INPUT:
    'cell_coords': a 2D np array of size N*2 storing N cell coordinates
    'y_len': height of the rectangular domain
    'x_len': width of the rectangular domain
    'y_periodic_bool': a boolean variable specifying whether we have periodic boundary condition in y direction 
    'x_periodic_bool': a boolean variable specifying whether we have periodic boundary condition in x direction 
    OUTPUT:
    'dist': a 2D np symmetrix array of size N*N storing pairwise cell distance
    'orientation': a 2D np array of size N*N storing pairwise cell orientation

    DEBUGGED
    '''

    # pairwise distance in x and y directions:
    dx = cell_coords[:, np.newaxis][:, :, 0] - cell_coords[:, 0]
    dy = cell_coords[:, np.newaxis][:, :, 1] - cell_coords[:, 1]

    # y-periodicity if specified:
    if y_periodic_bool: 
        dy_abs = abs(dy)
        dy_H = y_len - dy_abs
        mask1 = (dy_abs >= dy_H) & (cell_coords[:, 1] < y_len/2)
        dy[mask1] = - dy_H[mask1]
        mask2 = (dy_abs >= dy_H) & (cell_coords[:, 1] >= y_len/2)
        dy[mask2] = dy_H[mask2]

    # x-periodicity if specified:
    if x_periodic_bool: 
        dx_abs = abs(dx)
        dx_H = x_len - dx_abs
        mask1 = (dx_abs >= dx_H) & (cell_coords[:, 0] < x_len/2)
        dx[mask1] = - dx_H[mask1]
        mask2 = (dx_abs >= dx_H) & (cell_coords[:, 0] >= x_len/2)
        dx[mask2] = dx_H[mask2]

    dist = np.sqrt(dx**2 + dy**2)
    orientation = np.arctan2(dy, dx) 

    # for cells overlapping, randomly select its migrating orientation:
    overlap_mask = (dx == 0.0) & (dy == 0.0) 
    orientation[overlap_mask] = np.random.uniform(-np.pi, np.pi, len(orientation[overlap_mask]))

    return dist, orientation

def CGstrength(plot_overall_folder, overall_folder, option_folders, option_subfolders, CGstrength_file_name, ts, xticklabels, xlabel, colors):
    for opt in range(len(option_folders)):
        if not os.path.exists(plot_overall_folder + option_folders[opt]):
            os.makedirs(plot_overall_folder + option_folders[opt])
            
        for t in range(len(ts)):
            datalist = []
            save_file_path = plot_overall_folder + option_folders[opt] + CGstrength_file_name + f'_t{ts[t]}.png'
            plt.rcParams['font.size'] = str(30)
            fig, ax = plt.subplots(figsize=(16, 9))
            for sub_opt in range(len(option_subfolders[opt])):
                input_info_path = overall_folder + option_folders[opt] + option_subfolders[opt][sub_opt] + 'CG_angles_alltimes.pickle'
                with open(input_info_path, 'rb') as file:
                    CG_strength_alltimes_allreps = pickle.load(file)
                datalist.append(np.abs(np.array(CG_strength_alltimes_allreps[t])))

            sns.violinplot(data=datalist, density_norm='width', common_norm=False, palette=colors, inner="point")
            ax.set_yticklabels([r'$0^{\circ}$', r'$15^{\circ}$', r'$30^{\circ}$', r'$45^{\circ}$', r'$60^{\circ}$', \
                                        r'$75^{\circ}$', r'$90^{\circ}$'])
            ax.set_yticks([0, 15, 30, 45, 60, 75, 90])
            xticks = np.arange(0, len(option_subfolders[opt]), 1)
            ax.set_xticks(list(xticks))
            ax.set_xticklabels(xticklabels)
            ax.set_xlabel(xlabel)
            ax.set_ylabel('|angle modulated by contact guidance|')
            ax.set_ylim([-20, 110])
            ax.tick_params(axis='both', which='major', labelsize=24)
            ax.set_title(f'At time {ts[t]} min')
            fig.savefig(save_file_path, dpi = 150, format='png')
            plt.close()
    pass

def compare_fibre_distributions(overall_folder, option_folders, t_extract_times, option_subfolders, plot_overall_folder, aniso_filename, v1_filename, xticklabels, yticklabels, xlabel, ylabel):
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['axes.labelweight'] = 'bold'  # Bold axis labels
    plt.rcParams['axes.titleweight'] = 'bold'  # Bold title
    plt.rcParams['font.weight'] = 'bold'
    names = ['_leftright.pickle', '_topbottom.pickle']
    plotnames = ['_leftright.png', '_topbottom.png']
    plot_names = ['(LR)', '(TB)']
    for opt in range(len(option_folders)):
        for t in range(len(t_extract_times)):
            kl_anisotropy_overall = np.zeros((len(option_subfolders[opt]), len(option_subfolders[opt])))
            kl_v1_overall = np.zeros((len(option_subfolders[opt]), len(option_subfolders[opt])))
            for i in range(2):
                kl_anisotropy = np.zeros((len(option_subfolders[opt]), len(option_subfolders[opt])))
                kl_v1 = np.zeros((len(option_subfolders[opt]), len(option_subfolders[opt])))
                for sub_opt_i in range(len(option_subfolders[opt])):
                    input_path_i = overall_folder + option_folders[opt] + option_subfolders[opt][sub_opt_i] + f'fibreInfo_t{t_extract_times[t]}' + names[i]
                    with open(input_path_i, 'rb') as f:
                        _ = pickle.load(f)
                        anisotropy_t_allreps_i = pickle.load(f)
                        v1_t_allreps_i = pickle.load(f)
                    anisotropy_t_allreps_i = np.concatenate(anisotropy_t_allreps_i)
                    anisotropy_t_allreps_i = anisotropy_t_allreps_i[~np.isnan(anisotropy_t_allreps_i)]
                    hist_aniso_i, _ = np.histogram(anisotropy_t_allreps_i, bins=30, density=True)
                    v1_t_allreps_i = np.concatenate(v1_t_allreps_i)
                    v1_t_allreps_i = v1_t_allreps_i[~np.isnan(v1_t_allreps_i)]
                    hist_v1_i, _ = np.histogram(v1_t_allreps_i, bins=60, density=True)

                    for sub_opt_j in range(len(option_subfolders[opt])):
                        input_path_j = overall_folder + option_folders[opt] + option_subfolders[opt][sub_opt_j] + f'fibreInfo_t{t_extract_times[t]}' + names[i]
                        with open(input_path_j, 'rb') as f:
                            _ = pickle.load(f)
                            anisotropy_t_allreps_j = pickle.load(f)
                            v1_t_allreps_j = pickle.load(f)

                        anisotropy_t_allreps_j = np.concatenate(anisotropy_t_allreps_j)
                        anisotropy_t_allreps_j = anisotropy_t_allreps_j[~np.isnan(anisotropy_t_allreps_j)]
                        hist_aniso_j, _ = np.histogram(anisotropy_t_allreps_j, bins=30, density=True)
                        kl_divergence_aniso = distance.jensenshannon(hist_aniso_i, hist_aniso_j)
                        #kl_divergence_aniso = entropy(hist_aniso_i+1e-10, hist_aniso_j+1e-10)
                        kl_anisotropy[sub_opt_i, sub_opt_j] = kl_divergence_aniso
                        kl_anisotropy_overall[sub_opt_i, sub_opt_j] = kl_anisotropy_overall[sub_opt_i, sub_opt_j] + kl_divergence_aniso

                        v1_t_allreps_j = np.concatenate(v1_t_allreps_j)
                        v1_t_allreps_j = v1_t_allreps_j[~np.isnan(v1_t_allreps_j)]
                        hist_v1_j, _ = np.histogram(v1_t_allreps_j, bins=60, density=True)
                        #kl_divergence_v1 = entropy(hist_v1_i+1e-10, hist_v1_j+1e-10)
                        kl_divergence_v1 = distance.jensenshannon(hist_v1_i, hist_v1_j)
                        kl_v1[sub_opt_i, sub_opt_j] = kl_divergence_v1
                        kl_v1_overall[sub_opt_i, sub_opt_j] = kl_v1_overall[sub_opt_i, sub_opt_j] + kl_divergence_v1


                xticks = np.arange(0, len(option_subfolders[opt]), 1)
                yticks = np.arange(0, len(option_subfolders[opt]), 1)
                save_file_path_aniso = plot_overall_folder + option_folders[opt] + aniso_filename + f'_t{t_extract_times[t]}' + plotnames[i]
                fig1, ax1 = plt.subplots(figsize=(12, 9))
                cax = ax1.imshow(kl_anisotropy, cmap='viridis', interpolation='nearest')
                cbar = fig1.colorbar(cax, ax=ax1)
                cbar.ax.tick_params(labelsize=24)
                ax1.set_title(r'$a$: KL Divergence' + f' at t = {t_extract_times[t]/60} hrs ' + plot_names[i], fontsize=30)
                ax1.set_xticks([])
                ax1.set_xticklabels([])
                ax1.set_yticks([])
                ax1.set_yticklabels([])  
                ax1.tick_params(axis='y', which='major', labelsize=30)
                ax1.tick_params(axis='x', which='major', labelsize=30)
                #ax1.set_xlabel(xlabel, fontsize=30)
                #ax1.set_ylabel(ylabel, fontsize=30)
                plt.tight_layout()
                fig1.savefig(save_file_path_aniso, dpi = 150, format='png')
                plt.close()

                save_file_path_v1 = plot_overall_folder + option_folders[opt] + v1_filename + f'_t{t_extract_times[t]}' + plotnames[i]
                fig2, ax2 = plt.subplots(figsize=(12, 9))
                cax = ax2.imshow(kl_v1, cmap='viridis', interpolation='nearest')
                cbar = fig2.colorbar(cax, ax=ax2)
                cbar.ax.tick_params(labelsize=24)
                ax2.set_title(r'$\angle \mathbf{v}_1$: KL Divergence' + f' at t = {t_extract_times[t]/60} hrs ' + plot_names[i], fontsize=30)
                ax2.set_xticks([])
                ax2.set_xticklabels([])
                ax2.set_yticks([])
                ax2.set_yticklabels([])
                ax2.tick_params(axis='y', which='major', labelsize=30)
                ax2.tick_params(axis='x', which='major', labelsize=30)
                #ax2.set_xlabel(xlabel, fontsize=30)
                #ax2.set_ylabel(ylabel, fontsize=30)
                plt.tight_layout()
                fig2.savefig(save_file_path_v1, dpi = 150, format='png')
                plt.close()

            #LR+TB: v1
            save_file_path_v1 = plot_overall_folder + option_folders[opt] + v1_filename + f'_t{t_extract_times[t]}' + '_LRTB.png'
            fig2, ax2 = plt.subplots(figsize=(12, 9))
            cax = ax2.imshow(kl_v1_overall, cmap='viridis', interpolation='nearest')
            cbar = fig2.colorbar(cax, ax=ax2)
            cbar.ax.tick_params(labelsize=24)
            ax2.set_title(r'$\angle \mathbf{v}_1$: KL Divergence' + f' at t = {t_extract_times[t]/60} hrs ' + '(LR+TB)', fontsize=30)
            ax2.set_xticks([])
            ax2.set_xticklabels([])
            ax2.set_yticks([])
            ax2.set_yticklabels([])
            ax2.tick_params(axis='y', which='major', labelsize=30)
            ax2.tick_params(axis='x', which='major', labelsize=30)
            #ax2.set_xlabel(xlabel, fontsize=30)
            #ax2.set_ylabel(ylabel, fontsize=30)
            plt.tight_layout()
            fig2.savefig(save_file_path_v1, dpi = 150, format='png')
            plt.close()

            #LR+TB: a
            save_file_path_v1 = plot_overall_folder + option_folders[opt] + aniso_filename + f'_t{t_extract_times[t]}' + '_LRTB.png'
            fig2, ax2 = plt.subplots(figsize=(12, 9))
            cax = ax2.imshow(kl_anisotropy_overall, cmap='viridis', interpolation='nearest')
            cbar = fig2.colorbar(cax, ax=ax2)
            cbar.ax.tick_params(labelsize=24)
            ax2.set_title(r'$a$: KL Divergence' + f' at t = {t_extract_times[t]/60} hrs ' + '(LR+TB)', fontsize=30)
            ax2.set_xticks([])
            ax2.set_xticklabels([])
            ax2.set_yticks([])
            ax2.set_yticklabels([])
            ax2.tick_params(axis='y', which='major', labelsize=30)
            ax2.tick_params(axis='x', which='major', labelsize=30)
            #ax2.set_xlabel(xlabel, fontsize=30)
            #ax2.set_ylabel(ylabel, fontsize=30)
            plt.tight_layout()
            fig2.savefig(save_file_path_v1, dpi = 150, format='png')
            plt.close()

            #LR+TB: v1+a
            save_file_path_v1 = plot_overall_folder + option_folders[opt] + 'v1Anisotropy' + f'_t{t_extract_times[t]}' + '_LRTB.png'
            fig2, ax2 = plt.subplots(figsize=(12, 9))
            cax = ax2.imshow(kl_v1_overall+kl_anisotropy_overall, cmap='viridis', interpolation='nearest')
            cbar = fig2.colorbar(cax, ax=ax2)
            cbar.ax.tick_params(labelsize=24)
            ax2.set_title(r'$\angle \mathbf{v}_1$ and $a$: KL Divergence' + f' at t = {t_extract_times[t]/60} hrs ' + '(LR+TB)', fontsize=30)
            ax2.set_xticks([])
            ax2.set_xticklabels([])
            ax2.set_yticks([])
            ax2.set_yticklabels([])
            ax2.tick_params(axis='y', which='major', labelsize=30)
            ax2.tick_params(axis='x', which='major', labelsize=30)
            #ax2.set_xlabel(xlabel, fontsize=30)
            #ax2.set_ylabel(ylabel, fontsize=30)
            plt.tight_layout()
            fig2.savefig(save_file_path_v1, dpi = 150, format='png')
            plt.close()

    pass


# ---------------------------- YY ----------------------------

# For Figures 3 and onwards:

def ave_cellnum_x(plot_overall_folder, overall_folder, subfolder, legend_, color_, y_middist):

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['axes.labelweight'] = 'bold'  # Bold axis labels
    plt.rcParams['axes.titleweight'] = 'bold'  # Bold title
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['font.size'] = str(40)

    t_extract_time = 42
    t_filename = 21

    save_path = plot_overall_folder + overall_folder + f'ave_cellnum_x_t{t_extract_time}.png'
    if not os.path.exists(plot_overall_folder + overall_folder):
        os.makedirs(plot_overall_folder + overall_folder)

    fig, ax = plt.subplots(figsize=(18, 11))
    fig.subplots_adjust(bottom=0.15)
    # load info + plot:
    index = 0
    for j in subfolder:
        input_info_path = overall_folder + j + f'cellnum_avex_allreps_t{t_filename}.txt'
        cellnum_avex_allreps_t = np.loadtxt(input_info_path, delimiter=' ')
        mean = np.mean(cellnum_avex_allreps_t, axis=0)
        std = np.std(cellnum_avex_allreps_t, axis=0)  

        ax.plot(y_middist, mean, label=legend_[index], linewidth=4.5, color=color_[index])
        ax.errorbar(y_middist, mean, yerr=std, color=color_[index], ecolor=color_[index], elinewidth=3, capsize=6)
        index += 1
    # format
    ax.set_ylabel('cell number')
    ax.set_xlabel(r'$x$ ($\mu$m)')
    ax.set_ylim([0, 60])
    ax.legend(fontsize=32)
    #ax.set_title(f'Ave cell number along $x$ at ${t_extract_time}$ hrs for different $\lambda_1+\lambda_2$', pad=20)
    #ax.set_title(f'Ave cell number along $x$ at ${t_extract_time}$ hrs for different $a=1-\lambda_2/\lambda_1$', pad=20)
    ax.set_title(f'Ave cell number along $x$ at ${t_extract_time}$ hrs for different degradation rates', pad=20)
    #ax.set_title(f'Ave cell number along $x$ at ${t_extract_time}$ hrs for different secretion rates', pad=20)
    ax.tick_params(axis='both', which='major', labelsize=32)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))

    fig.savefig(save_path, dpi = 150, format='png')
    plt.close()

    pass

def cell_population(plot_overall_folder, overall_folder, subfolder, legend_, color_, times):
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['axes.labelweight'] = 'bold'  # Bold axis labels
    plt.rcParams['axes.titleweight'] = 'bold'  # Bold title
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['font.size'] = str(40)

    save_path = plot_overall_folder + overall_folder + f'ave_cellpopulation.png'
    if not os.path.exists(plot_overall_folder + overall_folder):
        os.makedirs(plot_overall_folder + overall_folder)

    fig, ax = plt.subplots(figsize=(16, 9))
    fig.subplots_adjust(bottom=0.15)

    # load info + plot:
    index = 0
    for j in subfolder:
        input_info_path = overall_folder + j + f'totalCellNum_alltimes.txt'
        cellpop_all_reps = np.loadtxt(input_info_path, delimiter=' ')
        mean = np.mean(cellpop_all_reps, axis=0)
        std = np.std(cellpop_all_reps, axis=0)  
        ax.plot(times, mean, label=legend_[index], linewidth=4.5, color=color_[index])
        index += 1

    ax.set_ylabel('average cell number')
    ax.set_xlabel(r'time $t$ (hrs)')
    ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
    ax.set_xticklabels([r'$0$', r'$10$', r'$20$', r'$30$', r'$40$', r'$50$', r'$60$'])
    ax.set_ylim([0, 700])
    ax.legend(fontsize=32)
    ax.tick_params(axis='both', which='major', labelsize=30)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))

    fig.savefig(save_path, dpi = 150, format='png')
    plt.close()
    pass

def cell_scatter_plot(plot_overall_folder, overall_folder, subfolder, x_min, x_max, y_min, y_max, sigma, title_):
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['axes.labelweight'] = 'bold'  # Bold axis labels
    plt.rcParams['axes.titleweight'] = 'bold'  # Bold title
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['font.size'] = str(32)

    t_filename = 21

    index = 0
    for j in subfolder:
        input_info_path = overall_folder + j + f'rep_0.pickle'
        with open(input_info_path, 'rb') as f:
            cell_coords_T = pickle.load(f)
        cell_coords_t = cell_coords_T[t_filename]

        save_path = plot_overall_folder + overall_folder + j + f'cell_scatter_t{t_filename}.png'
        if not os.path.exists(plot_overall_folder + overall_folder + j):
            os.makedirs(plot_overall_folder + overall_folder + j)

        fig, ax = plt.subplots()
        ax.set_aspect('equal', adjustable='box')
        # plot the cells:
        for x, y in zip(cell_coords_t[:, 0], cell_coords_t[:, 1]):
            cell = Circle((x, y), sigma/2, facecolor='none', edgecolor='blue', linewidth=1)
            ax.add_patch(cell)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title_[index])

        fig.savefig(save_path, dpi = 150, format='png')
        plt.close()

        index += 1
    pass

def extract_fibre_totaldensity_anisotropy(Omega_t):

    # This function extract fibre's total density as well as the anisotroy degree at one time step.

    # get rid of numerical artifacts:
    mask = np.abs(Omega_t) <= 10**(-10)
    Omega_t[mask] = 0.0

    # init:
    total_density = np.full((np.shape(Omega_t)[0], np.shape(Omega_t)[1]), np.nan)
    anisotropy = np.full((np.shape(Omega_t)[0], np.shape(Omega_t)[1]), np.nan)
    v1_degree = np.full((np.shape(Omega_t)[0], np.shape(Omega_t)[1]), np.nan)

    for i in range(np.shape(Omega_t)[0]):
        for j in range(np.shape(Omega_t)[1]):
            eigenvalues_old, eigenvectors = LA.eig(Omega_t[i, j])
            eigenvalues = np.sort(eigenvalues_old)   
            lambda_1, lambda_2 =  eigenvalues[-1], eigenvalues[0]
            # get rid of numerical artifacts:
            if np.abs(lambda_1) <= 10**(-10):
                lambda_1 = 0.0
            if np.abs(lambda_2) <= 10**(-10):
                lambda_2 = 0.0
            if isinstance(lambda_1, complex) or isinstance(lambda_2, complex): # Warning: if somehow gives us complex number
                print('Complex lambdas!')
                print(Omega_t[i, j])
            if lambda_1 < 0.0 or lambda_2 < 0.0: # Warning: invalid density ratios
                print('Negative lambdas!')
                print(Omega_t[i, j])
            if lambda_2 > lambda_1: # Warning: wrong order!
                print('lambda1 < lambda2!')
            
            total_fibre = lambda_1 + lambda_2
            # there is fibre:
            if total_fibre > 0: 
                total_density[i, j] = total_fibre
                anisotropy[i, j] = 1 - lambda_2/lambda_1
                v1 = eigenvectors[:, np.where(eigenvalues_old == lambda_1)[0][0]]
                v1_angle = np.degrees(np.arctan2(v1[1], v1[0]))
                if v1_angle >= -180 and v1_angle < 0:
                    v1_angle = v1_angle + 180
                v1_degree[i, j] = v1_angle

    return total_density, anisotropy, v1_degree

def cell_fibre_dynamics(plot_overall_folder, overall_folder, subfolder, fibre_x_2D, fibre_y_2D, x_min, x_max, y_min, y_max, sigma, title_):
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['axes.labelweight'] = 'bold'  # Bold axis labels
    plt.rcParams['axes.titleweight'] = 'bold'  # Bold title
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['font.size'] = str(30) # change font size globally

    t_filenames = [0, 6, 12, 18, 24, 30]
    #t_filename = 21

    for t_filename in t_filenames:
        index = 0
        for j in subfolder:
            input_info_path = overall_folder + j + f'rep_0.pickle'
            with open(input_info_path, 'rb') as f:
                cell_coords_T = pickle.load(f)
                Omega_T = pickle.load(f)
            cell_coords = cell_coords_T[t_filename]
            Omega_t = Omega_T[t_filename]
            total_density, anisotropy, v1_degree = extract_fibre_totaldensity_anisotropy(Omega_t)

            save_path = plot_overall_folder + overall_folder + j + f'cell_fibre_dynamics_t{t_filename*2}.png'
            if not os.path.exists(plot_overall_folder + overall_folder + j):
                os.makedirs(plot_overall_folder + overall_folder + j)
        
            fig = plt.figure(figsize=(22, 6), constrained_layout=True)
            ax1 = plt.subplot(1, 4, 1) # cell dynamics 
            ax2 = plt.subplot(1, 4, 2) # fibre density
            ax3 = plt.subplot(1, 4, 3) # major fibre orientation
            ax4 = plt.subplot(1, 4, 4) # anisotropy degree

            # cell scatter plot:
            for x, y in zip(cell_coords[:, 0], cell_coords[:, 1]):
                cell = Circle((x, y), sigma/2, facecolor='none', edgecolor='blue', linewidth=1)
                ax1.add_patch(cell)
            ax1.set_aspect('equal', adjustable='box')
            ax1.set_xlim(x_min, x_max)
            ax1.set_ylim(y_min, y_max)
            ax1.set_xticks([])
            ax1.set_yticks([])
            #ax1.set_title(title_[index])
            #ax1.set_title(title_[index], color='red', fontsize=24) 
            ax1.set_title(f't = {t_filename*2} hrs')

            # total fibre density:
            grey_gradient = cm.Greys(np.linspace(0,1, num=50)) # 50 values from 0 to 1
            colors = grey_gradient
            mycmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
            c2 = ax2.pcolormesh(fibre_x_2D, fibre_y_2D, total_density, cmap=mycmap, vmin=0.0, vmax=1)
            ax2.set_aspect('equal', adjustable='box')
            if index == 0:
                ax2.set_title("Total area fraction")
            ax2.set_xlim(x_min, x_max)
            ax2.set_ylim(y_min, y_max)
            ax2.set_xticks([])
            ax2.set_yticks([])
            colorbar = fig.colorbar(c2, ax=ax2, shrink=0.8, orientation='horizontal')
            custom_ticks = [0.0, 0.5, 1.0]  # Replace with your desired tick values
            colorbar.set_ticks(custom_ticks)
            tick_labels = [r'$0.0$', r'$0.5$', r'$1.0$']  
            colorbar.set_ticklabels(tick_labels)

            # Major fibre orientation
            nan_color = (1, 1, 0, 0.1) #[(1, 1, 0, 0.5)]*5 # 8 values from -pi/8 to 0
            green_gradient1 = cm.Greens(np.linspace(1,0, num=25)) # 32 values from 0 to pi/2
            green_gradient2 = cm.Greens(np.linspace(0,1, num=25)) # 32 values from pi/2 to pi
            colors = np.vstack((green_gradient1, green_gradient2))
            mycmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
            c3 = ax3.pcolormesh(fibre_x_2D, fibre_y_2D, v1_degree, cmap=mycmap, vmin=0.0, vmax=180)
            c3.cmap.set_bad(color=nan_color)
            ax3.set_aspect('equal', adjustable='box')
            if index == 0:
                ax3.set_title("Major orientation")
            ax3.set_xlim(x_min, x_max)
            ax3.set_ylim(y_min, y_max)
            ax3.set_xticks([])
            ax3.set_yticks([])
            colorbar = fig.colorbar(c3, ax=ax3, orientation='horizontal', shrink=0.8)
            custom_ticks = [0, 90, 180]  # Replace with your desired tick values
            colorbar.set_ticks(custom_ticks)
            tick_labels = [r'$0^\circ$', r'$90^\circ$', r'$180^\circ$']  
            colorbar.set_ticklabels(tick_labels)

            # Anisotropy degree 
            nan_color = (1, 1, 0, 0.1) #[(1, 1, 0, 0.5)]*5 # 5 values from -0.1 to 0
            blue_gradient = cm.Blues(np.linspace(0,1, num=50)) # 50 values from 0 to 1
            colors = blue_gradient
            mycmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
            c4 = ax4.pcolormesh(fibre_x_2D, fibre_y_2D, anisotropy, cmap=mycmap, vmin= 0.0, vmax=1)
            c4.cmap.set_bad(color=nan_color)
            ax4.set_aspect('equal', adjustable='box')
            if index == 0:
                ax4.set_title("Anisotropy degree")
            ax4.set_xlim(x_min, x_max)
            ax4.set_ylim(y_min, y_max)
            ax4.set_xticks([])
            ax4.set_yticks([])
            colorbar = fig.colorbar(c4, ax=ax4, shrink=0.8, orientation='horizontal')
            custom_ticks = [0, 0.5, 1]  # Replace with your desired tick values
            colorbar.set_ticks(custom_ticks)
            tick_labels = [r'$0.0$', r'$0.5$', r'$1.0$']  
            colorbar.set_ticklabels(tick_labels)
        
            fig.savefig(save_path, dpi = 150, format='png')
            plt.close()

            index += 1

    pass

def fibre_summarystats(plot_overall_folder, overall_folder, subfolder, xticklabel, xlabel, colors_):

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['axes.labelweight'] = 'bold'  # Bold axis labels
    plt.rcParams['axes.titleweight'] = 'bold'  # Bold title
    plt.rcParams['font.weight'] = 'bold'

    t_filename = 21
    xticks = np.arange(0, 4, 1) # there are 4 different parameter sets to compare

    index = 0

    density_list, aniso_list, v1_list, frac_changed_list = [], [], [], [] # compare for different parameter sets

    if not os.path.exists(plot_overall_folder + overall_folder):
        os.makedirs(plot_overall_folder + overall_folder)

    for j in subfolder:
        input_info_path = overall_folder + j + f'fibreInfo_LR_t21.pickle'
        with open(input_info_path, 'rb') as f:
            total_density_t_allreps_LR = pickle.load(f)
            aisotropy_t_allreps_LR = pickle.load(f)
            v1_degree_t_allreps_LR = pickle.load(f)
            frac_changed_t_allreps_LR = pickle.load(f)
        density_list.append(np.concatenate(total_density_t_allreps_LR))
        aniso_list.append(np.concatenate(aisotropy_t_allreps_LR))
        v1_list.append(np.concatenate(v1_degree_t_allreps_LR))
        frac_changed_list.append(np.array(frac_changed_t_allreps_LR))

    # compare total fibre density:
    fig1, ax1 = plt.subplots(figsize=(17, 9))
    save_file_path1 = plot_overall_folder + overall_folder + f'fibre_summarystats_density_t{t_filename}_LR.png'
    sns.boxplot(data=density_list, ax=ax1, palette=colors_)
    ax1.set_ylim(-0.4, 1.4)
    ax1.set_xticks(list(xticks))
    ax1.set_xticklabels(xticklabel)
    ax1.set_yticklabels([r'$0.0$', r'$0.5$', r'$1.0$'])
    ax1.set_yticks([0, 0.5, 1])
    ax1.set_title(r'Total area fraction $\lambda_1+\lambda_2$ (LR)', fontsize=40)
    ax1.tick_params(axis='y', which='major', labelsize=32)
    ax1.tick_params(axis='x', which='major', labelsize=32)
    ax1.set_xlabel(xlabel, fontsize=32)
    plt.tight_layout()
    fig1.savefig(save_file_path1, dpi = 150, format='png')
    plt.close()

    # compare anisotropy:
    fig2, ax2 = plt.subplots(figsize=(17, 9))
    save_file_path2 = plot_overall_folder + overall_folder + f'fibre_summarystats_anisotropy_t{t_filename}_LR.png'
    sns.boxplot(data=aniso_list, ax=ax2, palette=colors_)
    ax2.set_ylim(-0.4, 1.4)
    ax2.set_xticks(list(xticks))
    ax2.set_xticklabels(xticklabel)
    ax2.set_yticklabels([r'$0.0$', r'$0.5$', r'$1.0$'])
    ax2.set_yticks([0, 0.5, 1])
    ax2.set_title(r'Anisotropy degree $a=1 - \lambda_2/\lambda_1$ (LR)', fontsize=40)
    ax2.tick_params(axis='y', which='major', labelsize=30)
    ax2.tick_params(axis='x', which='major', labelsize=30)
    ax2.set_xlabel(xlabel, fontsize=32)
    plt.tight_layout()
    fig2.savefig(save_file_path2, dpi = 150, format='png')
    plt.close()   

    # compare v1: 
    fig3, ax3 = plt.subplots(figsize=(17, 9))
    save_file_path3 = plot_overall_folder + overall_folder + f'fibre_summarystats_v1_t{t_filename}_LR.png'
    sns.violinplot(data=v1_list,  density_norm='width', common_norm=True, ax=ax3, palette=colors_, inner="point")
    ax3.set_ylim(-40, 240)
    ax3.set_xticks(list(xticks))
    ax3.set_xticklabels(xticklabel)
    ax3.set_yticklabels([r'$0^{\circ}$', r'$30^{\circ}$', r'$60^{\circ}$', r'$90^{\circ}$', r'$120^{\circ}$', \
                                r'$150^{\circ}$', r'$180^{\circ}$'])
    ax3.set_yticks([0, 30, 60, 90, 120, 150, 180])
    ax3.set_title(r'Major fibre orientation $\angle \mathbf{v}_1$ (LR)', fontsize=40)
    ax3.tick_params(axis='y', which='major', labelsize=32)
    ax3.tick_params(axis='x', which='major', labelsize=32)
    ax3.set_xlabel(xlabel, fontsize=32)
    plt.tight_layout()
    fig3.savefig(save_file_path3, dpi = 150, format='png')

    plt.close()
    pass

def count_rep_neigh(cell_coords, y_periodic_bool, x_periodic_bool, y_len, x_len, rep_adh_len):
    # this function counts the number of repulsive neighbours for each cell. 
    dx = cell_coords[:, np.newaxis][:, :, 0] - cell_coords[:, 0]
    dy = cell_coords[:, np.newaxis][:, :, 1] - cell_coords[:, 1]
    if y_periodic_bool: 
        dy_abs = abs(dy)
        dy_H = y_len - dy_abs
        mask1 = (dy_abs >= dy_H) & (cell_coords[:, 1] < y_len/2)
        dy[mask1] = - dy_H[mask1]
        mask2 = (dy_abs >= dy_H) & (cell_coords[:, 1] >= y_len/2)
        dy[mask2] = dy_H[mask2]
    if x_periodic_bool: 
        dx_abs = abs(dx)
        dx_H = x_len - dx_abs
        mask1 = (dx_abs >= dx_H) & (cell_coords[:, 0] < x_len/2)
        dx[mask1] = - dx_H[mask1]
        mask2 = (dx_abs >= dx_H) & (cell_coords[:, 0] >= x_len/2)
        dx[mask2] = dx_H[mask2]
    dist = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(dist, np.nan)
    rep_mask = dist<=rep_adh_len
    num_rep_neigh = np.nansum(rep_mask,axis=1)
    return num_rep_neigh

def num_rep_neighbour(Numrep, plot_overall_folder, overall_folder, subfolder, y_periodic_bool, y_len, x_periodic_bool, x_len, rep_adh_len, xlabel, xticklabels):

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['axes.labelweight'] = 'bold'  # Bold axis labels
    plt.rcParams['axes.titleweight'] = 'bold'  # Bold title
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['font.size'] = str(26)

    t_filename = 21

    save_path = plot_overall_folder + overall_folder + f'num_repneighbours_t{t_filename}.png'
    if not os.path.exists(plot_overall_folder + overall_folder):
        os.makedirs(plot_overall_folder + overall_folder)

    # extract info:
    datalist = [[] for ff in range(len(subfolder))]
    index = 0
    for j in subfolder:
        for rep in range(Numrep): 
            input_info_path = overall_folder + j + f'rep_{rep}.pickle'
            with open(input_info_path, 'rb') as file:
                cell_coord_T = pickle.load(file)
            cell_coords = cell_coord_T[t_filename]

            # extract number of repulsive neighbours:
            num_rep_neigh = count_rep_neigh(cell_coords, y_periodic_bool, x_periodic_bool, y_len, x_len, rep_adh_len)
            datalist[index].extend(list(num_rep_neigh))
        index += 1
    output_filename = overall_folder + j + f'num_repul_neighbours_t{t_filename}.pickle'
    with open(output_filename, 'wb') as file:
        pickle.dump(datalist, file)

    # plot:
    fig, ax = plt.subplots(figsize=(16, 9)) 
    fig.subplots_adjust(bottom=0.15)
    histograms = [np.histogram(arr, bins=np.linspace(0, 7, 8), density=False)[0]/Numrep for arr in datalist]
    data_matrix = np.column_stack(histograms)
    sns.heatmap(data_matrix, cmap="BuPu", annot=True, cbar=True, vmin=0, vmax=130)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, fontsize=26)
    ax.set_xticklabels(xticklabels)
    ax.set_ylabel(f'average number of repulsive neighbours')
    ax.set_title(r'$t = 42$ hrs')
    ax.tick_params(axis='x', which='major', labelsize=24)
    fig.savefig(save_path, dpi = 150, format='png')
    plt.close()
            
    pass

def fraction_domain_withFibres_T(NumRep, plot_overall_folder, overall_folder, subfolder, legend_, color_, times):
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['axes.labelweight'] = 'bold'  # Bold axis labels
    plt.rcParams['axes.titleweight'] = 'bold'  # Bold title
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['font.size'] = str(40)

    fig, ax = plt.subplots(figsize=(16, 9))
    fig.subplots_adjust(bottom=0.15)

    save_path = plot_overall_folder + overall_folder + f'fractionDomain_wFibres_overtime.png'
    if not os.path.exists(plot_overall_folder + overall_folder):
        os.makedirs(plot_overall_folder + overall_folder)

    index = 0
    for j in subfolder:
        fibre_density_T = np.zeros((NumRep, len(times)), dtype=float)
        for rep in range(NumRep):
            input_path = overall_folder + j + f'rep_{rep}.pickle'
            with open(input_path, 'rb') as f:
                _ = pickle.load(f)
                Omega_T = pickle.load(f)
            for t in range(len(times)):
                Omega_t = Omega_T[t]
                diagonal_elements = Omega_t[..., 0, 0] + Omega_t[..., 1, 1]
                fibre_density_t = np.sum(diagonal_elements) / (Omega_t.shape[0]*Omega_t.shape[1])
                fibre_density_T[rep, t] = fibre_density_t
        fibre_density_T_mean = np.mean(fibre_density_T, axis=0)
        fibre_density_T_std = np.std(fibre_density_T, axis=0)

        ax.plot(times, fibre_density_T_mean, label=legend_[index], linewidth=4.5, color=color_[index])
        ax.fill_between(times, fibre_density_T_mean-fibre_density_T_std, fibre_density_T_mean+fibre_density_T_std, alpha=0.2, color=color_[index])
        index += 1
        
    ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
    ax.set_xticklabels([r'$0$', r'$10$', r'$20$', r'$30$', r'$40$', r'$50$', r'$60$'])
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(r'time $t$ (hrs)')
    ax.legend(fontsize=32)
    ax.set_title(f'Fraction of domain occupied by collagen fibres')
    ax.tick_params(axis='both', which='major', labelsize=32)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

    fig.savefig(save_path, dpi = 150, format='png')
    plt.close()

    pass

def initial_conditions(plot_overall_folder, overall_folder, subfolder, fibre_x_2D, fibre_y_2D, x_min, x_max, y_min, y_max, sigma, IC_title_):

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['axes.labelweight'] = 'bold'  # Bold axis labels
    plt.rcParams['axes.titleweight'] = 'bold'  # Bold title
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['font.size'] = str(30) # change font size globally

    t_filename = 0

    index = 0
    for j in subfolder:
        input_info_path = overall_folder + j + f'rep_0.pickle'
        with open(input_info_path, 'rb') as f:
            cell_coords_T = pickle.load(f)
            Omega_T = pickle.load(f)
        cell_coords = cell_coords_T[t_filename]
        Omega_t = Omega_T[t_filename]
        total_density, anisotropy, v1_degree = extract_fibre_totaldensity_anisotropy(Omega_t)

        save_path = plot_overall_folder + overall_folder + j + f'cell_fibre_dynamics_t{t_filename}.png'
        if not os.path.exists(plot_overall_folder + overall_folder + j):
            os.makedirs(plot_overall_folder + overall_folder + j)
    
        fig = plt.figure(figsize=(22, 6), constrained_layout=True)
        ax1 = plt.subplot(1, 4, 1) # cell dynamics 
        ax2 = plt.subplot(1, 4, 2) # fibre density
        ax3 = plt.subplot(1, 4, 3) # major fibre orientation
        ax4 = plt.subplot(1, 4, 4) # anisotropy degree

        # cell scatter plot:
        for x, y in zip(cell_coords[:, 0], cell_coords[:, 1]):
            cell = Circle((x, y), sigma/2, facecolor='none', edgecolor='blue', linewidth=1)
            ax1.add_patch(cell)
        ax1.set_aspect('equal', adjustable='box')
        ax1.set_xlim(x_min, x_max)
        ax1.set_ylim(y_min, y_max)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_title(IC_title_)

        # total fibre density:
        grey_gradient = cm.Greys(np.linspace(0,1, num=50)) # 50 values from 0 to 1
        colors = grey_gradient
        mycmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
        c2 = ax2.pcolormesh(fibre_x_2D, fibre_y_2D, total_density, cmap=mycmap, vmin=0.0, vmax=1)
        ax2.set_aspect('equal', adjustable='box')
        if index == 0:
            ax2.set_title("Total area fraction")
        ax2.set_xlim(x_min, x_max)
        ax2.set_ylim(y_min, y_max)
        ax2.set_xticks([])
        ax2.set_yticks([])
        colorbar = fig.colorbar(c2, ax=ax2, shrink=0.8, orientation='horizontal')
        custom_ticks = [0.0, 0.5, 1.0]  # Replace with your desired tick values
        colorbar.set_ticks(custom_ticks)
        tick_labels = [r'$0.0$', r'$0.5$', r'$1.0$']  
        colorbar.set_ticklabels(tick_labels)

        # Major fibre orientation
        nan_color = (1, 1, 0, 0.1) #[(1, 1, 0, 0.5)]*5 # 8 values from -pi/8 to 0
        green_gradient1 = cm.Greens(np.linspace(1,0, num=25)) # 32 values from 0 to pi/2
        green_gradient2 = cm.Greens(np.linspace(0,1, num=25)) # 32 values from pi/2 to pi
        colors = np.vstack((green_gradient1, green_gradient2))
        mycmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
        c3 = ax3.pcolormesh(fibre_x_2D, fibre_y_2D, v1_degree, cmap=mycmap, vmin=0.0, vmax=180)
        c3.cmap.set_bad(color=nan_color)
        ax3.set_aspect('equal', adjustable='box')
        if index == 0:
            ax3.set_title("Major orientation")
        ax3.set_xlim(x_min, x_max)
        ax3.set_ylim(y_min, y_max)
        ax3.set_xticks([])
        ax3.set_yticks([])
        colorbar = fig.colorbar(c3, ax=ax3, orientation='horizontal', shrink=0.8)
        custom_ticks = [0, 90, 180]  # Replace with your desired tick values
        colorbar.set_ticks(custom_ticks)
        tick_labels = [r'$0^\circ$', r'$90^\circ$', r'$180^\circ$']  
        colorbar.set_ticklabels(tick_labels)

        # Anisotropy degree 
        nan_color = (1, 1, 0, 0.1) #[(1, 1, 0, 0.5)]*5 # 5 values from -0.1 to 0
        blue_gradient = cm.Blues(np.linspace(0,1, num=50)) # 50 values from 0 to 1
        colors = blue_gradient
        mycmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
        c4 = ax4.pcolormesh(fibre_x_2D, fibre_y_2D, anisotropy, cmap=mycmap, vmin= 0.0, vmax=1)
        c4.cmap.set_bad(color=nan_color)
        ax4.set_aspect('equal', adjustable='box')
        if index == 0:
            ax4.set_title("Anisotropy degree")
        ax4.set_xlim(x_min, x_max)
        ax4.set_ylim(y_min, y_max)
        ax4.set_xticks([])
        ax4.set_yticks([])
        colorbar = fig.colorbar(c4, ax=ax4, shrink=0.8, orientation='horizontal')
        custom_ticks = [0, 0.5, 1]  # Replace with your desired tick values
        colorbar.set_ticks(custom_ticks)
        tick_labels = [r'$0.0$', r'$0.5$', r'$1.0$']  
        colorbar.set_ticklabels(tick_labels)
    
        fig.savefig(save_path, dpi = 150, format='png')
        plt.close()

        index += 1

        break

    pass

def parametersweep_xave_heatmap(plot_overall_folder, overall_folder):
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    fig, ax = plt.subplots(figsize=(16, 9))

    t_extract_time = 21

    input_path = overall_folder + f'KL_xavespan_t{t_extract_time}.pickle'
    with open(input_path, 'rb') as f:
        KLmatrix_xavespan = pickle.load(f)

    save_path = plot_overall_folder + overall_folder + f'KL_xavespan_t{t_extract_time}.png'
    if not os.path.exists(plot_overall_folder + overall_folder):
        os.makedirs(plot_overall_folder + overall_folder)

    cax = ax.imshow(KLmatrix_xavespan, cmap='viridis', interpolation='nearest')

    cbar = fig.colorbar(cax, ax=ax)
    cbar.ax.tick_params(labelsize=24)
    cbar.set_label(r'JSD comparaed with $d=0$ $\rm{min}^{-1}$', fontsize=24)

    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.tick_params(axis='y', which='major', labelsize=30)
    ax.tick_params(axis='x', which='major', labelsize=30)
    ax.set_yticks([10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
    ax.set_yticklabels(['0', r'$2^{-10}$', r'$2^{-9}$', r'$2^{-8}$', r'$2^{-7}$', r'$2^{-6}$', r'$2^{-5}$', r'$2^{-4}$', r'$2^{-3}$', r'$2^{-2}$', r'$2^{-1}$'], fontsize=24)
    ax.set_xticks([0, 10])
    ax.set_xticklabels([r'$2^{-1}$',r'$0$'], fontsize=24) 

    contour = ax.contour(KLmatrix_xavespan, levels=[0.05, 0.2], colors='red', linewidths=2.5)
    ax.clabel(contour, inline=True, fontsize=24)

    ax.set_xlabel(r'secretion rate $s$ $(\rm{min}^{-1})$', fontsize=30)
    ax.set_ylabel(r'degradation rate $d$ $(\rm{min}^{-1})$', fontsize=30)
    ax.set_title(r'Similarity in average cell density along $x$' + r' at $t = 42$ hrs', fontsize=40)
    
    plt.gca().invert_xaxis()  # Reverse the x-axis
    plt.tight_layout()

    fig.savefig(save_path, dpi = 150, format='png')
    plt.close()

    pass

# For figrue 2 (initial void of fibres with a bigger domain)

def initVoidFibres_cell_fibre_dynamics(plot_overall_folder, fibre_x_2D, fibre_y_2D, x_min, x_max, y_min, y_max, sigma):

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['axes.labelweight'] = 'bold'  # Bold axis labels
    plt.rcParams['axes.titleweight'] = 'bold'  # Bold title
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['font.size'] = str(30) # change font size globally

    t_filenames = [0, 3, 21, 36] # for 0, 6, 32, 40 hrs
    times = [0, 6, 42, 72]

    input_info_path = 'InitNOfibres_/sec0.025_deg0.0025_xlen540.0_ylen540.0/rep_0.pickle'
    with open(input_info_path, 'rb') as f:
        cell_coords_T = pickle.load(f)
        Omega_T = pickle.load(f)

    index = 0
    for t_filename in t_filenames:
        time = times[index]

        cell_coords = cell_coords_T[t_filename]
        Omega_t = Omega_T[t_filename]
        total_density, anisotropy, v1_degree = extract_fibre_totaldensity_anisotropy(Omega_t)

        save_path = plot_overall_folder + f'InitNOfibres/sec0.025_deg0.0025_xlen540.0_ylen540.0/cell_fibre_dynamics_t{t_filename}.png'
        if not os.path.exists(plot_overall_folder + f'InitNOfibres/sec0.025_deg0.0025_xlen540.0_ylen540.0/'):
            os.makedirs(plot_overall_folder + f'InitNOfibres/sec0.025_deg0.0025_xlen540.0_ylen540.0/')
    
        fig = plt.figure(figsize=(22, 6), constrained_layout=True)
        ax1 = plt.subplot(1, 4, 1) # cell dynamics 
        ax2 = plt.subplot(1, 4, 2) # fibre density
        ax3 = plt.subplot(1, 4, 3) # major fibre orientation
        ax4 = plt.subplot(1, 4, 4) # anisotropy degree

        # cell scatter plot:
        for x, y in zip(cell_coords[:, 0], cell_coords[:, 1]):
            cell = Circle((x, y), sigma/2, facecolor='none', edgecolor='blue', linewidth=1)
            ax1.add_patch(cell)
        ax1.set_aspect('equal', adjustable='box')
        ax1.set_xlim(x_min, x_max)
        ax1.set_ylim(y_min, y_max)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_title(f'Figure 2')

        # total fibre density:
        grey_gradient = cm.Greys(np.linspace(0,1, num=50)) # 50 values from 0 to 1
        colors = grey_gradient
        mycmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
        c2 = ax2.pcolormesh(fibre_x_2D, fibre_y_2D, total_density, cmap=mycmap, vmin=0.0, vmax=1)
        ax2.set_aspect('equal', adjustable='box')
        if index in [0, 1]:
            ax2.set_title("Total area fraction")
        ax2.set_xlim(x_min, x_max)
        ax2.set_ylim(y_min, y_max)
        ax2.set_xticks([])
        ax2.set_yticks([])
        colorbar = fig.colorbar(c2, ax=ax2, shrink=0.8, orientation='horizontal')
        custom_ticks = [0, 0.5, 1]  # Replace with your desired tick values
        colorbar.set_ticks(custom_ticks)
        tick_labels = [r'$0.0$', r'$0.5$', r'$1.0$']  
        colorbar.set_ticklabels(tick_labels)

        # Major fibre orientation
        nan_color = (1, 1, 0, 0.1) #[(1, 1, 0, 0.5)]*5 # 8 values from -pi/8 to 0
        green_gradient1 = cm.Greens(np.linspace(1,0, num=25)) # 32 values from 0 to pi/2
        green_gradient2 = cm.Greens(np.linspace(0,1, num=25)) # 32 values from pi/2 to pi
        colors = np.vstack((green_gradient1, green_gradient2))
        mycmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
        c3 = ax3.pcolormesh(fibre_x_2D, fibre_y_2D, v1_degree, cmap=mycmap, vmin=0.0, vmax=180)
        c3.cmap.set_bad(color=nan_color)
        ax3.set_aspect('equal', adjustable='box')
        if index in [0, 1]:
            ax3.set_title("Major orientation")
        ax3.set_xlim(x_min, x_max)
        ax3.set_ylim(y_min, y_max)
        ax3.set_xticks([])
        ax3.set_yticks([])
        colorbar = fig.colorbar(c3, ax=ax3, orientation='horizontal', shrink=0.8)
        custom_ticks = [0, 90, 180]  # Replace with your desired tick values
        colorbar.set_ticks(custom_ticks)
        tick_labels = [r'$0^\circ$', r'$90^\circ$', r'$180^\circ$']  
        colorbar.set_ticklabels(tick_labels)

        # Anisotropy degree 
        nan_color = (1, 1, 0, 0.1) #[(1, 1, 0, 0.5)]*5 # 5 values from -0.1 to 0
        blue_gradient = cm.Blues(np.linspace(0,1, num=50)) # 50 values from 0 to 1
        colors = blue_gradient
        mycmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
        c4 = ax4.pcolormesh(fibre_x_2D, fibre_y_2D, anisotropy, cmap=mycmap, vmin= 0.0, vmax=1)
        c4.cmap.set_bad(color=nan_color)
        ax4.set_aspect('equal', adjustable='box')
        if index in [0, 1]:
            ax4.set_title("Anisotropy degree")
        ax4.set_xlim(x_min, x_max)
        ax4.set_ylim(y_min, y_max)
        ax4.set_xticks([])
        ax4.set_yticks([])
        colorbar = fig.colorbar(c4, ax=ax4, shrink=0.8, orientation='horizontal')
        custom_ticks = [0, 0.5, 1]  # Replace with your desired tick values
        colorbar.set_ticks(custom_ticks)
        tick_labels = [r'$0.0$', r'$0.5$', r'$1.0$']  
        colorbar.set_ticklabels(tick_labels)
    
        fig.savefig(save_path, dpi = 150, format='png')
        plt.close()

        index += 1

    pass

def initVoidFibres_compare_v1(plot_overall_folder):

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['axes.labelweight'] = 'bold'  # Bold axis labels
    plt.rcParams['axes.titleweight'] = 'bold'  # Bold title
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['font.size'] = str(40)
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.subplots_adjust(bottom=0.15)  # Adjust the bottom margin to make space for labels

    input_path_TB = 'InitNOfibres/sec0.025_deg0.0025_xlen540.0_ylen540.0/fibreInfo_TB_t36.pickle'
    with open(input_path_TB, 'rb') as f:
        _ = pickle.load(f)
        _ = pickle.load(f)
        v1_degree_TB = pickle.load(f)

    input_path_LR = 'InitNOfibres/sec0.025_deg0.0025_xlen540.0_ylen540.0/fibreInfo_LR_t36.pickle'
    with open(input_path_LR, 'rb') as f:
        _ = pickle.load(f)
        _ = pickle.load(f)
        v1_degree_LR = pickle.load(f)

    savename = plot_overall_folder + f'InitNOfibres/sec0.025_deg0.0025_xlen540.0_ylen540.0/v1compare_LRTB_t{72}.png'

    data = [v1_degree_LR, v1_degree_TB]

    sns.violinplot(data=data)
    ax.set_yticks([0, 45, 90, 135, 180])
    ax.set_yticklabels([r'$0^{\circ}$', r'$45^{\circ}$', r'$90^{\circ}$', r'$135^{\circ}$', r'$180^{\circ}$'])
    ax.set_ylim([-40, 220])
    ax.set_xticklabels(['LR', 'TB'])
    plt.title(r'$t = 72$ hrs')
    plt.ylabel(r'Major fibre orientation $\angle \hat{\mathbf{v}}_1$')
    plt.tight_layout()
    fig.savefig(savename, dpi = 150, format='png')
    plt.close()

    pass

def extract_fibre_info_withdistance(Omega_reshape_t, init_density, distances_reshape):
    # get rid of numerical artifacts:
    mask1 = np.abs(Omega_reshape_t) <= 10**(-10)
    Omega_reshape_t[mask1] = 0.0

    # init:
    total_density = []
    aisotropy = []
    v1_degree = []

    total_area = float(len(Omega_reshape_t))
    changed_area = 0.0

    for i in range(len(Omega_reshape_t)):
        eigenvalues_old, eigenvectors = LA.eig(Omega_reshape_t[i])
        eigenvalues = np.sort(eigenvalues_old)   
        lambda_1, lambda_2 =  eigenvalues[-1], eigenvalues[0]
        total_fibre = lambda_1 + lambda_2
        # get rid of numerical artifacts:
        if np.abs(lambda_1) <= 10**(-10):
            lambda_1 = 0.0
        if np.abs(lambda_2) <= 10**(-10):
            lambda_2 = 0.0

        # simple check:
        if isinstance(lambda_1, complex) or isinstance(lambda_2, complex): 
            print('Warning: at least one of the lambdas is complex!\n')
        if lambda_1 < 0.0 or lambda_2 < 0.0: 
            print('Warning: at least onr of the lambdas is negative!\n')
        if lambda_2 > lambda_1:
            print('Warning: we shall reverse the order of lambdas!\n')

        # not consider fibre grid points that have not been altered by cells:
        if np.abs(total_fibre-init_density) > 10**(-10):
            changed_area = changed_area + 1

            total_density.append(total_fibre)
            if lambda_1 > 0:
                aisotropy.append(np.array([distances_reshape[i], 1-lambda_2/lambda_1]))
            else:
                aisotropy.append(np.array([np.nan, np.nan]))
            v1 = eigenvectors[:, np.where(eigenvalues_old == lambda_1)[0][0]]
            v1_angle = np.rad2deg(np.arctan2(v1[1], v1[0]))
            if v1_angle >= -180 and v1_angle < 0:
                v1_angle = v1_angle + 180
            v1_degree.append(v1_angle)

    # return back to np.array rather than list:
    frac_changed = changed_area / total_area
    total_density = np.array(total_density)
    aisotropy = np.array(aisotropy)
    v1_degree = np.array(v1_degree)

    return total_density, aisotropy, v1_degree, frac_changed

def initVoidFibres_anisotropy(plot_overall_folder, fibre_x_2D, fibre_y_2D, x_mid, y_mid):

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['axes.labelweight'] = 'bold'  # Bold axis labels
    plt.rcParams['axes.titleweight'] = 'bold'  # Bold title
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['font.size'] = str(40)
    fig, ax = plt.subplots(figsize=(16, 9))

    savename = plot_overall_folder + f'InitNOfibres/sec0.025_deg0.0025_xlen540.0_ylen540.0/anisotropy_VS_dist.png'

    distances_to_mid = np.sqrt((fibre_x_2D-x_mid)**2 + (fibre_y_2D-y_mid)**2)

    init_density = 0.0

    bins = np.linspace(0, 270, 55) # bin the distance from the domain center
    lenth = (bins[1]-bins[0])/2
    anisotropy_mean_3 = np.full(len(bins)-1, np.nan)
    anisotropy_mean_21 = np.full(len(bins)-1, np.nan)
    anisotropy_mean_36 = np.full(len(bins)-1, np.nan)

    input_path = 'InitNOfibres/sec0.025_deg0.0025_xlen540.0_ylen540.0/rep_0.pickle'
    with open(input_path, 'rb') as f:
        _ = pickle.load(f)
        Omega_T = pickle.load(f)
    Omega_3_ = Omega_T[3]
    Omega_21_ = Omega_T[21]
    Omega_36_ = Omega_T[36]

    for i in range(len(bins)-1):
        l, r = bins[i], bins[i+1]
        mask = (distances_to_mid>=l) & (distances_to_mid<r)
        distances_all = distances_to_mid[mask]

        Omega_3, Omega_21, Omega_36 = Omega_3_[mask], Omega_21_[mask], Omega_36_[mask]

        total_elements_3 = Omega_3.size
        nm_3 = int(total_elements_3/4)
        Omega_3_reshape = Omega_3.reshape((nm_3, 2, 2))
        distances_reshape_3 = distances_all.reshape(nm_3)
        _, aisotropy_3, _, _ = extract_fibre_info_withdistance(Omega_3_reshape, init_density, distances_reshape_3)
        if len(aisotropy_3) > 0:
            anisotropy_mean_3[i] = np.mean(aisotropy_3[:,1])

        total_elements_21 = Omega_21.size
        nm_21 = int(total_elements_21/4)
        Omega_21_reshape = Omega_21.reshape((nm_21, 2, 2))
        distances_reshape_21 = distances_all.reshape(nm_21)
        _, aisotropy_21, _, _ = extract_fibre_info_withdistance(Omega_21_reshape, init_density, distances_reshape_21)
        if len(aisotropy_21) > 0:
            anisotropy_mean_21[i] = np.mean(aisotropy_21[:,-1])

        total_elements_36 = Omega_36.size
        nm_36 = int(total_elements_36/4)
        Omega_36_reshape = Omega_36.reshape((nm_36, 2, 2))
        distances_reshape_36 = distances_all.reshape(nm_36)
        _, aisotropy_36, _, _ = extract_fibre_info_withdistance(Omega_36_reshape, init_density, distances_reshape_36)
        if len(aisotropy_36) > 0:
            anisotropy_mean_36[i] = np.mean(aisotropy_36[:,-1])

    anisotropy_mean_3 = anisotropy_mean_3[~np.isnan(anisotropy_mean_3)]
    anisotropy_mean_21 = anisotropy_mean_21[~np.isnan(anisotropy_mean_21)]
    anisotropy_mean_36 = anisotropy_mean_36[~np.isnan(anisotropy_mean_36)]

    ax.plot((bins+lenth)[0:len(anisotropy_mean_3)], anisotropy_mean_3, linewidth=5.5, alpha=0.15, color='blue', label='6 hrs')
    ax.plot((bins+lenth)[0:len(anisotropy_mean_21)], anisotropy_mean_21, linewidth=5.5, alpha=0.45, color='blue', label='42 hrs')
    ax.plot((bins+lenth)[0:len(anisotropy_mean_36)], anisotropy_mean_36, linewidth=5.5, alpha=1.0, color='blue', label='72 hrs')

    ax.set_xlabel(r'distance to domain centre ($\mu$m)')
    ax.set_ylabel(r'average anisotropy degree $a$', labelpad=20)
    ax.set_ylim([0, 1.01])
    ax.legend(fontsize=40)
    plt.tight_layout()
    fig.savefig(savename, dpi = 150, format='png')
    plt.close()
    pass

def angle_between_rows(A, B):
    # Compute the dot product between corresponding rows
    dot_product = np.einsum('ij,ij->i', A, B)
    
    # Compute the magnitudes of each row
    norm_A = np.linalg.norm(A, axis=1)
    norm_B = np.linalg.norm(B, axis=1)
    
    # Calculate cosine of the angle
    cos_theta = dot_product / (norm_A * norm_B)
    
    # Ensure the values are within the valid range for arccos (avoid floating-point issues)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    # Calculate the angles in radians
    angles = np.rad2deg(np.arccos(cos_theta))
    
    return angles

def compare_CG_strength(plot_overall_folder, overall_folder, subfolder):

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['axes.labelweight'] = 'bold'  # Bold axis labels
    plt.rcParams['axes.titleweight'] = 'bold'  # Bold title
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['font.size'] = str(40)
    fig, axs = plt.subplots(1, 3, figsize=(16, 8))

    t = 21
    savename = plot_overall_folder + overall_folder + f'compare_CG_strength_t{t}.png'

    index = 0
    for j in subfolder:
        with open(overall_folder + j + 'rep_0.pickle', 'rb') as file:
            cell_coords_T = pickle.load(file)
            _ = pickle.load(file)
            cell_vel_T = pickle.load(file)
            cell_vel_NoCG_T = pickle.load(file)

        cell_vel = cell_vel_NoCG_T[t]
        cell_coords = cell_coords_T[t][0:len(cell_vel)]
        X, Y, U, V = cell_coords[:, 0], cell_coords[:, 1], cell_vel[:, 0], cell_vel[:, 1]
        cell_vel_CG = cell_vel_T[t]
        U2, V2 = cell_vel_CG[:, 0], cell_vel_CG[:, 1]
        angles_j = angle_between_rows(cell_vel, cell_vel_CG)

        grid_x, grid_y = np.mgrid[0:360:200j, 0:360:200j]  
        grid_z = griddata(cell_coords, angles_j, (grid_x, grid_y), method='linear')

        im_j = axs[index].imshow(grid_z.T, extent=(0, 360, 0, 360), origin='lower', cmap='viridis', vmin=0, vmax=90)
        axs[index].set_xticks([])
        axs[index].set_yticks([])

        index += 1
    
    cbar = plt.colorbar(im_j, ax=axs, fraction=0.015, pad=0.04, orientation='vertical')
    cbar.set_label('angle modulated by CG')
    cbar.set_ticks([0, 45, 90])
    cbar.set_ticklabels([r'$0^\circ$', r'$45^\circ$', r'$90^\circ$'])
    fig.savefig(savename, dpi = 150, format='png')
    
    pass


if __name__ == "__main__":

    plot_overall_folder = 'Plots/'
    if not os.path.exists(plot_overall_folder):
        os.makedirs(plot_overall_folder)

    # For Figures 3 and onwards:
    Numrep = 40
    overall_folders = ['fibredensity_/', 
                       'fibreisotropy_/', 
                       'degradation_/', 
                       'secretion_/', 
                       'SI_degradation_secretion_/', 
                       'secdeg_parametersweep_/', 
                       'ShiftTau_/', 
                       'DiffusionEpsilon_slowProlif_/', 
                       'DiffusionEpsilon_/', 
                       'ShiftTau_sec{s}_deg{d}/', 
                       'DiffusionEpsilon_quickProlif_sec0.05_deg0.0005/',
                       'DiffusionEpsilon_quickProlif_sec0.005050505050505051_deg0.5/', 
                       'ShiftTau_sec0.005050505050505051_deg0.5/', 
                       'DiffusionEpsilon_sec0.05_deg0.0005_normalprolif/'
                       ]
    subfolders = [[
                   #'VerticalFibre_totaldensity0.0_iso0.1/', 
                   'VerticalFibre_totaldensity0.1_iso0.1/', 
                   #'VerticalFibre_totaldensity0.3_iso0.1/', 
                   'VerticalFibre_totaldensity0.4_iso0.1/', 
                   #'VerticalFibre_totaldensity0.5_iso0.1/', 
                   'VerticalFibre_totaldensity1.0_iso0.1/'], 

                  [
                   #'VerticalFibre_totaldensity0.8_iso1.0/', 
                   'VerticalFibre_totaldensity0.8_iso0.75/', 
                   #'VerticalFibre_totaldensity0.8_iso0.5/', 
                   #'VerticalFibre_totaldensity0.8_iso0.25/', 
                   #'VerticalFibre_totaldensity0.8_iso0.1/', 
                   'VerticalFibre_totaldensity0.8_iso0.05/'], 

                  ['deg0.0025_ICdensity0.9_ICiso0.1/', 
                   'deg0.25_ICdensity0.9_ICiso0.1/'], 

                  ['sec0.0025_ICdensity0.1_ICiso0.1/', 
                   #'sec0.005_ICdensity0.1_ICiso0.1/', 
                   #'sec0.025_ICdensity0.1_ICiso0.1/', 
                   'sec0.25_ICdensity0.1_ICiso0.1/'], 

                  [
                   'deg0.0025_sec0.0_ICdensity0.5_ICiso0.1/', 
                   'deg0.25_sec0.0_ICdensity0.5_ICiso0.1/', 
                   'deg0.0_sec0.0005_ICdensity0.5_ICiso0.1/', 
                   'deg0.0_sec0.005_ICdensity0.5_ICiso0.1/',
                   'deg0.0_sec0.05_ICdensity0.5_ICiso0.1/', 
                   'deg0.0_sec0.5_ICdensity0.5_ICiso0.1/'
                   ]]
    s_values = [1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128, 1/256, 1/512, 1/1024, 0.0]
    d_values = [1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128, 1/256, 1/512, 1/1024, 0.0]
    sd_params = np.array([np.array([s, d]) for s, d in itertools.product(s_values, d_values)])
    secdeg_parametersweep_subfolders = [f'deg{d}_sec{s}_ICdensity0.5_ICiso0.1/' for s, d in sd_params]
    subfolders.append(secdeg_parametersweep_subfolders)

    shift_s = [0.005, 0.5, 0.9]
    tau_s = [1, 300, 1200]
    parameters = np.array([np.array([s, t]) for s, t in itertools.product(shift_s, tau_s)])
    shift_tau_subfolders = [f'shift{s}_tau{t}/' for s, t in parameters]
    subfolders.append(shift_tau_subfolders)

    sigma = 12.0
    D_s = [sigma/10, sigma/40, sigma/160]
    epsilons = [0.05, 0.1, 0.2]
    parameters = np.array([np.array([d, e]) for d, e in itertools.product(D_s, epsilons)])
    diffusion_subfolders = [f'Diffusion{d}_Epsilon{e}/' for d, e in parameters]
    subfolders.append(diffusion_subfolders)

    subfolders.append(diffusion_subfolders)

    subfolders.append(shift_tau_subfolders)

    subfolders.append(diffusion_subfolders)

    subfolders.append(diffusion_subfolders)

    subfolders.append(shift_tau_subfolders)

    subfolders.append(['Diffusion0.3_Epsilon0.05/'])



    legends = [[#r'$\lambda_1+\lambda_2 = 0.0$', 
                r'$\lambda_1+\lambda_2 = 0.1$', 
                #r'$\lambda_1+\lambda_2 = 0.3$', 
                r'$\lambda_1+\lambda_2 = 0.4$',
                #r'$\lambda_1+\lambda_2 = 0.5$', 
                r'$\lambda_1+\lambda_2 = 1.0$'],

               [#r'$a = 0.00$', 
                r'$a = 0.25$', 
                #r'$a = 0.50$', 
                #r'$a = 0.75$', 
                #r'$a = 0.90$', 
                r'$a = 0.95$'],

               [r'$d=0.0025$ $\rm{min}^{-1}$', 
                r'$d=0.25$ $\rm{min}^{-1}$'],

               [r'$s=0.0025$ $\rm{min}^{-1}$', 
                #r'$s=0.005$ $\rm{min}^{-1}$', 
                #r'$s=0.025$ $\rm{min}^{-1}$',
                r'$s=0.25$ $\rm{min}^{-1}$'],

               [
                r'$d=0.0025$ $\rm{min}^{-1}$', 
                r'$d=0.25$ $\rm{min}^{-1}$', 
                r'$s=0.0005$ $\rm{min}^{-1}$',
                r'$s=0.005$ $\rm{min}^{-1}$', 
                r'$s=0.05$ $\rm{min}^{-1}$',
                r'$s=0.5$ $\rm{min}^{-1}$'
                ]]

    cmap_dark = plt.get_cmap("Dark2")
    cmap_tab10 = plt.get_cmap("tab10")
    cmap_set1 = plt.get_cmap("Set1")
    cc = cmap_dark(np.linspace(0, 1, 6))
    colors = [cmap_dark(np.linspace(0, 1, len(subfolders[0]))),
              cmap_dark(np.linspace(0, 1, len(subfolders[1]))),
              #[cc[0], cc[2], cc[3], cc[5]],
              cmap_tab10(np.linspace(0, 1, len(subfolders[2]))),
              cmap_tab10(np.linspace(0, 1, len(subfolders[3]))),
              cmap_tab10(np.linspace(0, 1, len(subfolders[4])))]

    titles = [[r'$\lambda_1+\lambda_2 = 0.1$', 
               r'$\lambda_1+\lambda_2 = 0.4$', 
               r'$\lambda_1+\lambda_2 = 1.0$'], # cell scatter plot at 40 hours

              [r'$a = 0.25$', 
               r'$a = 0.95$'], # cell scatter plot at 40 hours

              [r'$d=0.0025$ $\rm{min}^{-1}$', 
              r'$d=0.25$ $\rm{min}^{-1}$'], # cell and fibre dynamic plot at 40 hours

              [r'$s=0.0025$ $\rm{min}^{-1}$', 
               r'$s=0.25$ $\rm{min}^{-1}$'], # cell and fibre dynamic plot at 40 hours

              [r'$d=0.0025$ $\rm{min}^{-1}$',
               r'$d=0.25$ $\rm{min}^{-1}$', # cell and fibre dynamic plot at 40 hours
               r'$s=0.0005$ $\rm{min}^{-1}$', # cell and fibre dynamic plot at 40 hours
               r'$s=0.005$ $\rm{min}^{-1}$',
               r'$s=0.05$ $\rm{min}^{-1}$', 
               r'$s=0.5$ $\rm{min}^{-1}$',] 
             ]

    IC_title_s = ['', '', 'Figure 4(a)', 'Figure 4(b)', 'Figure 5', '', '', '', '', '', '', '', '', 'Figure 6']

    xticklabels = [[r'$0.0$', r'$0.1$', r'$0.3$', r'$0.4$', r'$0.5$', r'$1.0$'],
                   [r'$0.00$', r'$0.25$', r'$0.50$', r'$0.75$', r'$0.90$', r'$0.95$'],
                   [], 
                   [0.0025, 0.005, 0.025, 0.25],
                   [0.0025, 0.25, 0.0005, 0.005, 0.05, 0.5]]  
    xlabels = [r'initial total area fraction $\lambda_1+\lambda_2$', 
               r'initial anisotropy degree $a$',]  

    init_densities = [[0.0, 0.1, 0.3, 0.4, 0.5, 1.0], 
                      [0.8], 
                      [0.9], 
                      [0.1], 
                      [0.5]]


    y_min, y_max, sigma, x_min, x_max = 0.0, 360.0, 12.0, 0.0, 360.0
    y_periodic_bool, x_periodic_bool, gridsize_f = True, True, 2.0
    x_len, y_len = x_max-x_min, y_max-y_min
    x_mid, y_mid = x_min+x_len/2, y_min+y_len/2
    centre = np.array([x_mid, y_mid])
    bin_len = sigma * 2
    bin_x, bin_y = np.arange(x_min, x_max+bin_len, bin_len), np.arange(y_min, y_max+bin_len, bin_len)
    x_middist, y_middist = (bin_x+bin_len/2)[0 : -1],  (bin_y+bin_len/2)[0 : -1]
    rep_adh_len = (2**(1/6))*sigma

    if y_periodic_bool:
        grid_y = np.arange(y_min, y_max, gridsize_f) 
    else:
        grid_y = np.arange(y_min, y_max+gridsize_f, gridsize_f) 
    if x_periodic_bool:
        grid_x = np.arange(x_min, x_max, gridsize_f) 
    else:
        grid_x = np.arange(x_min, x_max+gridsize_f, gridsize_f) 
    _, Y = np.meshgrid(grid_x, grid_y)
    fibre_x_2D, fibre_y_2D = np.meshgrid(grid_x, grid_y, indexing="xy")

    times = np.arange(0, 31, 1)


    folder_toextract = [10, 11, 12]
    folder_toextract = [13]
    for i in folder_toextract:
        overall_folder = overall_folders[i]
        subfolder = subfolders[i]
        if i < 5:
            legend_ = legends[i]
            color_ = colors[i]
            title_ = titles[i]
            xticklabel = xticklabels[i]
            IC_title_ = IC_title_s[i]

        # average cell number across x direction at 42 hours
        #ave_cellnum_x(plot_overall_folder, overall_folder, subfolder, legend_, color_, y_middist)

        # average cell population over time
        #cell_population(plot_overall_folder, overall_folder, subfolder, legend_, color_, times)

        # number of closest neighbours at 42 hours
        #xlabel = xlabels[i]
        #num_rep_neighbour(Numrep, plot_overall_folder, overall_folder, subfolder, y_periodic_bool, y_len, x_periodic_bool, x_len, rep_adh_len, xlabel, xticklabel)

        # cell scatter plot at 42 hours
        #cell_scatter_plot(plot_overall_folder, overall_folder, subfolder, x_min, x_max, y_min, y_max, sigma, title_)
        # cell and fibre dynamics at 42 hours
        title_ = ''
        cell_fibre_dynamics(plot_overall_folder, overall_folder, subfolder, fibre_x_2D, fibre_y_2D, x_min, x_max, y_min, y_max, sigma, title_)

        # fibre summary stats at 42 hours for the left and the right regions
        #xlabel = r'secretion rate $s$ ($\rm{min}^{-1}$)'
        #colors_ = cmap_tab10(np.linspace(0, 1, 4)) 
        #fibre_summarystats(plot_overall_folder, overall_folder, subfolder, xticklabel, xlabel, colors_)

        # extract fibre density over time for degradation only model
        #fraction_domain_withFibres_T(Numrep, plot_overall_folder, overall_folder, subfolder, legend_, color_, times)

        # plot initial conditions to be put in the supplementary information
        IC_title_ = 'Figure 6'
        initial_conditions(plot_overall_folder, overall_folder, subfolder, fibre_x_2D, fibre_y_2D, x_min, x_max, y_min, y_max, sigma, IC_title_)

        # plot the KL heatmap to compare average cell density along x at 42 hours
        #parametersweep_xave_heatmap(plot_overall_folder, overall_folder)

        # compare the dynamics: 
        #subfolder = ['deg0.0625_sec0.0009765625_ICdensity0.5_ICiso0.1/', # star 1
        #             'deg0.0625_sec0.0625_ICdensity0.5_ICiso0.1/', # star 2
        #             'deg0.0009765625_sec0.0625_ICdensity0.5_ICiso0.1/' # star 3
        #]
        #title_ = [r'$\star^1$', r'$\star^2$', r'$\star^3$']
        #cell_fibre_dynamics(plot_overall_folder, overall_folder, subfolder, fibre_x_2D, fibre_y_2D, x_min, x_max, y_min, y_max, sigma, title_)
        #compare_CG_strength(plot_overall_folder, overall_folder, subfolder)


    # For figrues 2 (initial void of fibres with a bigger domain)
    # plot cell and fibre dynamics:
    y_min, y_max, sigma, x_min, x_max = 0.0, 540.0, 12.0, 0.0, 540.0
    y_periodic_bool, x_periodic_bool, gridsize_f = True, True, 2.0
    x_len, y_len = x_max-x_min, y_max-y_min
    x_mid, y_mid = x_min+x_len/2, y_min+y_len/2
    centre = np.array([x_mid, y_mid])
    bin_len = sigma * 2
    bin_x, bin_y = np.arange(x_min, x_max+bin_len, bin_len), np.arange(y_min, y_max+bin_len, bin_len)
    x_middist, y_middist = (bin_x+bin_len/2)[0 : -1],  (bin_y+bin_len/2)[0 : -1]
    rep_adh_len = (2**(1/6))*sigma

    if y_periodic_bool:
        grid_y = np.arange(y_min, y_max, gridsize_f) 
    else:
        grid_y = np.arange(y_min, y_max+gridsize_f, gridsize_f) 
    if x_periodic_bool:
        grid_x = np.arange(x_min, x_max, gridsize_f) 
    else:
        grid_x = np.arange(x_min, x_max+gridsize_f, gridsize_f) 
    _, Y = np.meshgrid(grid_x, grid_y)
    fibre_x_2D, fibre_y_2D = np.meshgrid(grid_x, grid_y, indexing="xy")

    # cell and fibre dynamics
    #initVoidFibres_cell_fibre_dynamics(plot_overall_folder, fibre_x_2D, fibre_y_2D, x_min, x_max, y_min, y_max, sigma)

    # compare major fibre orientation (LR VS TB)
    #initVoidFibres_compare_v1(plot_overall_folder)

    # anisotropy versus distance to the domain center
    #initVoidFibres_anisotropy(plot_overall_folder, fibre_x_2D, fibre_y_2D, x_mid, y_mid)
