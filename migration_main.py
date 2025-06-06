import numpy as np
import random
import pickle
import os
import multiprocessing
import sys
from scipy.interpolate import RectBivariateSpline
from numpy.linalg import norm 
import matplotlib.pyplot as plt
import itertools
from itertools import product



from migration_subfunctions import *

def one_run(cell_coords, fibre_coords, Omega, tend_iternum, delta_t, x_min, x_max, x_len, y_min, y_max, y_len, grid_x, grid_y, eta, D, sigma, r_max, epsilon, rep_adh_len, F_adh_max, beta, \
            omega_0, s, d, tau, xi, Delta_0, rho_0, y_periodic_bool, x_periodic_bool, CG_bool, prolif_bool, seed, store_path, rep, shift, scale, n):
    '''
    # This function runs simulation for the 'rep'th time based on the input parameter values, and store the cell and fibre dynamics into 'store_path/rep_{rep}.pickle' at 't_extract_times'.
    '''

    store_name = store_path + f'rep_{rep}.pickle' # i.e. 'store_path/rep_{rep}.pickle'

    tstep = 120 # store every 120min=2hrs
    t_extract_times = list(np.arange(0, tend_iternum+tstep, tstep, dtype=int))

    random.seed(seed)
    np.random.seed(seed)

    # define lists to store cell coordinates, cell numbers, fibre field info, velocities before and after the contact guidance, which will be output into a pickle file:
    cell_coords_T = [] 
    Omega_T = [] 
    cell_vel_NoCG_T = []
    cell_vel_T = [] 

    # initialise 'Vel_cells', which is a list of N empty, each will be used to record the history of a cell's (inside the domain) previous velocities.
    # this is useful for fibre secretion by cells. 
    N = int(len(cell_coords))
    Vel_cells = [[] for _ in range(N)]

    # iniitalise cell-cell interaction forces before the loop for density proliferation purpose: 
    # obtain inter-cellular forces experienced by all the cells:
    dist, orientation = cellcell_dis_orien(cell_coords, y_len, x_len, y_periodic_bool, x_periodic_bool)
    F_cc, total_adh_magnitude, rep_dist_, num_rep_neigh, num_adh_neigh = total_F_cc(dist, orientation, sigma, epsilon, r_max, rep_adh_len, delta_t)

    # step throughout time by 1st order Euler:
    for t in range(tend_iternum+1):

        if t in t_extract_times:
            cell_coords_T.append(cell_coords.copy())
            Omega_T.append(Omega.copy())

        # proliferation if possible:
        if prolif_bool:
            cell_coords_bf = cell_coords.copy()
            child_coords, mother_coord = cell_proliferation(Delta_0, rho_0, xi, prolif_bool, cell_coords_bf, num_rep_neigh, num_adh_neigh, N, y_periodic_bool, x_periodic_bool, \
                                                            y_len, x_len, x_min, x_max, y_min, y_max, delta_t) # note that cells are all within the domain, so no need to remove those outside the domain
            # update 'cell_coords', 'Vel_cells' and 'N' for those children cells:
            cell_coords = np.vstack((cell_coords, child_coords))
            Vel_cells = Vel_cells + [[] for _ in range(len(child_coords))]
            N = int(len(cell_coords))

        # obtain random forces experienced by all the cells:
        F_rm = total_F_rm(D, N, delta_t)
        # obtain inter-cellular forces experienced by all the cells:
        dist, orientation = cellcell_dis_orien(cell_coords, y_len, x_len, y_periodic_bool, x_periodic_bool)
        F_cc, total_adh_magnitude, rep_dist_, num_rep_neigh, num_adh_neigh = total_F_cc(dist, orientation, sigma, epsilon, r_max, rep_adh_len, delta_t)

        # evaluate fibre info at cell central locations:
        Omega_cell_loc, total_fibre_cell_loc = fibre_cell_locs(grid_x, grid_y, Omega, cell_coords, N)
        zero_indices_cc, zero_CG_strength_cc, zero_indices_rm, zero_CG_strength_rm = [], [], [], []
        if CG_bool: # fibre contact guidance if possible:
            # contact guidance modulates random motion:
            M_rand_hat, zero_indices_rm, zero_CG_strength_rm = CG_rand(total_fibre_cell_loc, F_rm, Omega_cell_loc, N, shift, scale, n)
            # contact guidance modulates inter-cellular interactions:
            M_cc_hat, zero_indices_cc, zero_CG_strength_cc = CG_rand(total_fibre_cell_loc, F_cc, Omega_cell_loc, N, shift, scale, n)
            cf_dist, Omega_reshape, total_fibre_density = CGcc_Pf(cell_coords, N, r_max, y_len, x_len, y_periodic_bool, x_periodic_bool, fibre_coords, Omega, grid_x, grid_y)
        else:
            M_rand_hat = np.full((N, 2, 2), np.array([[1.0, 0.0], [0.0, 1.0]]))
            M_cc_hat = np.full((N, 2, 2), np.array([[1.0, 0.0], [0.0, 1.0]]))

        # migrate and update 'cell_coords' & 'Vel_cells':
        cell_coords, Vel_cells, vel_t = migrate_t(cell_coords, Vel_cells, M_cc_hat, M_rand_hat, F_rm, F_cc, eta, delta_t, y_periodic_bool, x_periodic_bool, x_min, x_max, x_len, y_min, y_max, y_len, N, 
                                                  zero_indices_cc, zero_CG_strength_cc, zero_indices_rm, zero_CG_strength_rm, Omega_cell_loc)
        # note that we are not removing rows of 'cell_coords', 'Vel_cells' and 'vel_t' when cells move outside the domain for their later usage in fibre dynamics. 
        if t in t_extract_times:
            cell_vel_T.append(vel_t.copy())
            vel_t_noCG = (F_rm + F_cc) / eta
            cell_vel_NoCG_T.append(vel_t_noCG.copy())

        # fibre dynamics driven by cells:
        # note that we use 'cf_dist', which is based on cell coordinates before migration. 
        # This makes sense, as we cells degrade and secrete fibres on grids that are under their impact areas BEFORE('_bf') migration.
        if CG_bool:
            Omega, Omega_reshape = fibre_degradation(Omega_reshape, cf_dist, sigma, d, omega_0, N, grid_x, grid_y, delta_t)
            Omega, Omega_reshape = fibre_secretion(Vel_cells, total_fibre_density, Omega_reshape, cf_dist, sigma, s, omega_0, N, grid_x, grid_y, delta_t, tau)

        # get rid of numerical underflow problems with Omega
        mask_underflow = np.abs(Omega) <= 10**(-10) 
        Omega[mask_underflow] = 0.0 

        # output into a pickle file 
        if t % 360.0 == 0.0:
            with open(store_name, 'wb') as file:
                pickle.dump(cell_coords_T, file)
                pickle.dump(Omega_T, file)
                pickle.dump(cell_vel_T, file)
                pickle.dump(cell_vel_NoCG_T, file)

    # output into a pickle file 
    with open(store_name, 'wb') as file:
        pickle.dump(cell_coords_T, file)
        pickle.dump(Omega_T, file)
        pickle.dump(cell_vel_T, file)
        pickle.dump(cell_vel_NoCG_T, file)

    pass



# A CELL INVASION SHOWCASE INVESTIGATION:
def cell_invasion(Numrep, tend_iternum, opt, numCPUs):

    if opt == 1: # veritcal fibre IC with different total fibre densities
        sub_path = 'fibredensity_/'
        if not os.path.exists(sub_path):
            os.makedirs(sub_path)

        # parameters:
        x_min, x_max, y_min, y_max, y_periodic_bool, x_periodic_bool = 0.0, 360.0, 0.0, 360.0, True, True
        x_len, y_len = x_max-x_min, y_max-y_min
        delta_t = 1.0
        eta, sigma, epsilon, beta = 1.0, 12.0, 0.1, 1.0
        D = sigma/40
        r_max, rep_adh_len, F_adh_max = 3*sigma, (2**(1/6))*sigma, 24*epsilon*(1/sigma)*(2*((7/26)**(13/6))-(7/26)**(7/6))
        rho_0 = 6.0
        xi = sigma/2
        CG_bool, gridsize_f = True, 2.0
        omega_0, s, d, tau = 1.0, 0.0, 0.0, 300.0
        prolif_bool, Delta_0 = True, 1440.0
        iso, Delta_theta = 0.1, np.pi/18 
        shift, scale, n = 0.4, 10.0, 1

        # init cell coordinates in the domain centre
        cellinit_filename = 'celldisc_coords_init.pickle'
        with open(cellinit_filename, 'rb') as f:
            cell_coords_init = pickle.load(f)

        # parameters to change:
        total_f_densities = [0.1, 0.3, 0.4, 0.5, 1.0, 0.0]
        seeds = np.arange(1, len(total_f_densities)*Numrep+1) 
        seeds = np.reshape(seeds, (len(total_f_densities), Numrep)) 

        for j in range(len(total_f_densities)):
            total_f_density = total_f_densities[j]
            lambda_1_const, lambda_2_const = total_f_density/(1+iso), total_f_density*iso/(1+iso)

            cell_coords = cell_coords_init.copy()
            Omega_init, fibre_coords, grid_x, grid_y = vAlignedNoise_fibre_constDensity_init(Delta_theta, lambda_1_const, lambda_2_const, x_min, x_max, y_min, y_max, gridsize_f, y_periodic_bool, x_periodic_bool)
            Omega = Omega_init.copy()

            sub_path_orient = sub_path + f'VerticalFibre_totaldensity{total_f_density}_iso{iso}/'
            if not os.path.exists(sub_path_orient):
                os.makedirs(sub_path_orient)
            with multiprocessing.Pool(processes=numCPUs) as pool:
                arguments = [(cell_coords, fibre_coords, Omega, tend_iternum, delta_t, x_min, x_max, x_len, y_min, y_max, y_len, grid_x, grid_y, eta, D, sigma, r_max, epsilon, rep_adh_len, F_adh_max, beta, \
                        omega_0, s, d, tau, xi, Delta_0, rho_0, y_periodic_bool, x_periodic_bool, CG_bool, prolif_bool, int(seeds[j, rep]), sub_path_orient, rep, shift, scale, n) for rep in range(Numrep)]
                pool.starmap(one_run, arguments)


    if opt == 2: # veritcal fibre IC with different isotropy degrees 
        sub_path = 'fibreisotropy_/'
        if not os.path.exists(sub_path):
            os.makedirs(sub_path)

        # parameters:
        x_min, x_max, y_min, y_max, y_periodic_bool, x_periodic_bool = 0.0, 360.0, 0.0, 360.0, True, True
        x_len, y_len = x_max-x_min, y_max-y_min
        delta_t = 1.0
        eta, sigma, epsilon, beta = 1.0, 12.0, 0.1, 1.0
        D = sigma/40
        r_max, rep_adh_len, F_adh_max = 3*sigma, (2**(1/6))*sigma, 24*epsilon*(1/sigma)*(2*((7/26)**(13/6))-(7/26)**(7/6))
        rho_0 = 6.0
        xi = sigma/2
        CG_bool, gridsize_f = True, 2.0
        omega_0, s, d, tau = 1.0, 0.0, 0.0, 300.0
        prolif_bool, Delta_0 = True, 1440.0
        total_f_density, Delta_theta = 0.8, np.pi/18 
        shift, scale, n = 0.4, 10.0, 1.0

        # init cell coordinates in the domain centre
        cellinit_filename = 'celldisc_coords_init.pickle'
        with open(cellinit_filename, 'rb') as f:
            cell_coords_init = pickle.load(f)

        # parameters to change:
        isotropies = [0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
        seeds = np.arange(1, len(isotropies)*Numrep+1) 
        seeds = np.reshape(seeds, (len(isotropies), Numrep)) 

        for j in range(len(isotropies)):
            iso = isotropies[j]
            lambda_1_const, lambda_2_const = total_f_density/(1+iso), total_f_density*iso/(1+iso)

            cell_coords = cell_coords_init.copy()
            Omega_init, fibre_coords, grid_x, grid_y = vAlignedNoise_fibre_constDensity_init(Delta_theta, lambda_1_const, lambda_2_const, x_min, x_max, y_min, y_max, gridsize_f, y_periodic_bool, x_periodic_bool)
            Omega = Omega_init.copy()

            sub_path_orient = sub_path + f'VerticalFibre_totaldensity{total_f_density}_iso{iso}/'
            if not os.path.exists(sub_path_orient):
                os.makedirs(sub_path_orient)
            with multiprocessing.Pool(processes=numCPUs) as pool:
                arguments = [(cell_coords, fibre_coords, Omega, tend_iternum, delta_t, x_min, x_max, x_len, y_min, y_max, y_len, grid_x, grid_y, eta, D, sigma, r_max, epsilon, rep_adh_len, F_adh_max, beta, \
                        omega_0, s, d, tau, xi, Delta_0, rho_0, y_periodic_bool, x_periodic_bool, CG_bool, prolif_bool, int(seeds[j, rep]), sub_path_orient, rep, shift, scale, n) for rep in range(Numrep)]
                pool.starmap(one_run, arguments)

    
    if opt == 3: # degradation only 
        sub_path = 'degradation_/'
        if not os.path.exists(sub_path):
            os.makedirs(sub_path)

        # parameters:
        x_min, x_max, y_min, y_max, y_periodic_bool, x_periodic_bool = 0.0, 360.0, 0.0, 360.0, True, True
        x_len, y_len = x_max-x_min, y_max-y_min
        delta_t = 1.0
        eta, sigma, epsilon, beta = 1.0, 12.0, 0.1, 1.0
        D = sigma/40
        r_max, rep_adh_len, F_adh_max = 3*sigma, (2**(1/6))*sigma, 24*epsilon*(1/sigma)*(2*((7/26)**(13/6))-(7/26)**(7/6))
        rho_0 = 6.0
        CG_bool, gridsize_f = True, 2.0
        omega_0, tau = 1.0, 300.0
        prolif_bool, Delta_0, xi = True, 1440.0, sigma/2  
        shift, scale, n = 0.4, 10.0, 1.0  

        # init cell coordinates in the domain centre
        cellinit_filename = 'celldisc_coords_init.pickle'
        with open(cellinit_filename, 'rb') as f:
            cell_coords_init = pickle.load(f)

        # init collagen fibre field: 
        total_f_density, iso, Delta_theta = 0.9, 0.1, np.pi/18 # degradation only
        #total_f_density, iso, Delta_theta = 0.5, 0.1, np.pi/18 # secretion only
        lambda_1_const, lambda_2_const = total_f_density/(1+iso), total_f_density*iso/(1+iso)
        Omega_init, fibre_coords, grid_x, grid_y = vAlignedNoise_fibre_constDensity_init(Delta_theta, lambda_1_const, lambda_2_const, x_min, x_max, y_min, y_max, gridsize_f, y_periodic_bool, x_periodic_bool)
    
        # parameters we vary:
        sd_params = np.array([[0.0, 0.0025], # degradation only
                              [0.0, 0.25]]) 

        seeds = np.arange(1, len(sd_params)*Numrep+1) 
        seeds = np.reshape(seeds, (len(sd_params), Numrep)) 

        for j in range(len(sd_params)):
            s, d = sd_params[j,0], sd_params[j,1]

            cell_coords = cell_coords_init.copy()
            Omega = Omega_init.copy()
            
            sub_path_tau = sub_path + f'deg{d}_ICdensity{total_f_density}_ICiso{iso}/'
            if not os.path.exists(sub_path_tau):
                os.makedirs(sub_path_tau)
            with multiprocessing.Pool(processes=numCPUs) as pool:
                arguments = [(cell_coords, fibre_coords, Omega, tend_iternum, delta_t, x_min, x_max, x_len, y_min, y_max, y_len, grid_x, grid_y, eta, D, sigma, r_max, epsilon, rep_adh_len, F_adh_max, beta, \
                        omega_0, s, d, tau, xi, Delta_0, rho_0, y_periodic_bool, x_periodic_bool, CG_bool, prolif_bool, int(seeds[j, rep]), sub_path_tau, rep, shift, scale, n) for rep in range(Numrep)]
                pool.starmap(one_run, arguments)


    if opt == 4: # secretion only 
        sub_path = 'secretion_/'
        if not os.path.exists(sub_path):
            os.makedirs(sub_path)

        # parameters:
        x_min, x_max, y_min, y_max, y_periodic_bool, x_periodic_bool = 0.0, 360.0, 0.0, 360.0, True, True
        x_len, y_len = x_max-x_min, y_max-y_min
        delta_t = 1.0
        eta, sigma, epsilon, beta = 1.0, 12.0, 0.1, 1.0
        D = sigma/40
        r_max, rep_adh_len, F_adh_max = 3*sigma, (2**(1/6))*sigma, 24*epsilon*(1/sigma)*(2*((7/26)**(13/6))-(7/26)**(7/6))
        rho_0 = 6.0
        CG_bool, gridsize_f = True, 2.0
        omega_0, tau = 1.0, 300.0
        prolif_bool, Delta_0, xi = True, 1440.0, sigma/2  
        shift, scale, n = 0.4, 10.0, 1.0  

        # init cell coordinates in the domain centre
        cellinit_filename = 'celldisc_coords_init.pickle'
        with open(cellinit_filename, 'rb') as f:
            cell_coords_init = pickle.load(f)

        # init collagen fibre field: 
        total_f_density, iso, Delta_theta = 0.1, 0.1, np.pi/18 
        #total_f_density, iso, Delta_theta = 0.5, 0.1, np.pi/18 # SI: secretion only
        lambda_1_const, lambda_2_const = total_f_density/(1+iso), total_f_density*iso/(1+iso)
        Omega_init, fibre_coords, grid_x, grid_y = vAlignedNoise_fibre_constDensity_init(Delta_theta, lambda_1_const, lambda_2_const, x_min, x_max, y_min, y_max, gridsize_f, y_periodic_bool, x_periodic_bool)
    
        # parameters we vary:
        sd_params = np.array([[0.0025, 0.0], # secretion only
                              [0.005, 0.0],
                              [0.025, 0.0],
                              [0.25, 0.0]]) 
        seeds = np.arange(1, len(sd_params)*Numrep+1) 
        seeds = np.reshape(seeds, (len(sd_params), Numrep)) 

        for j in range(len(sd_params)):
            s, d = sd_params[j,0], sd_params[j,1]

            cell_coords = cell_coords_init.copy()
            Omega = Omega_init.copy()
            
            sub_path_tau = sub_path + f'sec{s}_ICdensity{total_f_density}_ICiso{iso}/'
            if not os.path.exists(sub_path_tau):
                os.makedirs(sub_path_tau)
            with multiprocessing.Pool(processes=numCPUs) as pool:
                arguments = [(cell_coords, fibre_coords, Omega, tend_iternum, delta_t, x_min, x_max, x_len, y_min, y_max, y_len, grid_x, grid_y, eta, D, sigma, r_max, epsilon, rep_adh_len, F_adh_max, beta, \
                        omega_0, s, d, tau, xi, Delta_0, rho_0, y_periodic_bool, x_periodic_bool, CG_bool, prolif_bool, int(seeds[j, rep]), sub_path_tau, rep, shift, scale, n) for rep in range(Numrep)]
                pool.starmap(one_run, arguments)


    if opt == 5: # degradation or secretion only -- SI 
        sub_path = 'SI_degradation_secretion_/'
        if not os.path.exists(sub_path):
            os.makedirs(sub_path)

        # parameters:
        x_min, x_max, y_min, y_max, y_periodic_bool, x_periodic_bool = 0.0, 360.0, 0.0, 360.0, True, True
        x_len, y_len = x_max-x_min, y_max-y_min
        delta_t = 1.0
        eta, sigma, epsilon, beta = 1.0, 12.0, 0.1, 1.0
        D = sigma/40
        r_max, rep_adh_len, F_adh_max = 3*sigma, (2**(1/6))*sigma, 24*epsilon*(1/sigma)*(2*((7/26)**(13/6))-(7/26)**(7/6))
        rho_0 = 6.0
        CG_bool, gridsize_f = True, 2.0
        omega_0, tau = 1.0, 300.0
        prolif_bool, Delta_0, xi = True, 1440.0, sigma/2  
        shift, scale, n = 0.4, 10.0, 1.0  

        # init cell coordinates in the domain centre
        cellinit_filename = 'celldisc_coords_init.pickle'
        with open(cellinit_filename, 'rb') as f:
            cell_coords_init = pickle.load(f)

        # init collagen fibre field: 
        total_f_density, iso, Delta_theta = 0.5, 0.1, np.pi/18 
        lambda_1_const, lambda_2_const = total_f_density/(1+iso), total_f_density*iso/(1+iso)
        Omega_init, fibre_coords, grid_x, grid_y = vAlignedNoise_fibre_constDensity_init(Delta_theta, lambda_1_const, lambda_2_const, x_min, x_max, y_min, y_max, gridsize_f, y_periodic_bool, x_periodic_bool)
    
        # parameters we vary:
        sd_params = np.array([[0.0, 0.0025], # degradation only
                              [0.0, 0.25], 
                              [0.0005, 0.0], # secretion only
                              [0.005, 0.0], 
                              [0.05, 0.0], 
                              [0.5, 0.0]
                              ]) 
        seeds = np.arange(1, len(sd_params)*Numrep+1) 
        seeds = np.reshape(seeds, (len(sd_params), Numrep)) 

        for j in range(len(sd_params)):
            s, d = sd_params[j,0], sd_params[j,1]

            cell_coords = cell_coords_init.copy()
            Omega = Omega_init.copy()
            
            sub_path_tau = sub_path + f'deg{d}_sec{s}_ICdensity{total_f_density}_ICiso{iso}/'
            if not os.path.exists(sub_path_tau):
                os.makedirs(sub_path_tau)
            with multiprocessing.Pool(processes=numCPUs) as pool:
                arguments = [(cell_coords, fibre_coords, Omega, tend_iternum, delta_t, x_min, x_max, x_len, y_min, y_max, y_len, grid_x, grid_y, eta, D, sigma, r_max, epsilon, rep_adh_len, F_adh_max, beta, \
                        omega_0, s, d, tau, xi, Delta_0, rho_0, y_periodic_bool, x_periodic_bool, CG_bool, prolif_bool, int(seeds[j, rep]), sub_path_tau, rep, shift, scale, n) for rep in range(Numrep)]
                pool.starmap(one_run, arguments)


    if opt == 6: # degradation + secretion: parameter sweep
        sub_path = 'secdeg_parametersweep_/'
        if not os.path.exists(sub_path):
            os.makedirs(sub_path)

        # parameters:
        x_min, x_max, y_min, y_max, y_periodic_bool, x_periodic_bool = 0.0, 360.0, 0.0, 360.0, True, True
        x_len, y_len = x_max-x_min, y_max-y_min
        delta_t = 1.0
        eta, sigma, epsilon, beta = 1.0, 12.0, 0.1, 1.0
        D = sigma/40
        r_max, rep_adh_len, F_adh_max = 3*sigma, (2**(1/6))*sigma, 24*epsilon*(1/sigma)*(2*((7/26)**(13/6))-(7/26)**(7/6))
        rho_0 = 6.0
        CG_bool, gridsize_f = True, 2.0
        omega_0, tau = 1.0, 300.0
        prolif_bool, Delta_0, xi = True, 1440.0, sigma/2  
        shift, scale, n = 0.4, 10.0, 1.0  

        # init cell coordinates in the domain centre
        cellinit_filename = 'celldisc_coords_init.pickle'
        with open(cellinit_filename, 'rb') as f:
            cell_coords_init = pickle.load(f)

        # init collagen fibre field: 
        total_f_density, iso, Delta_theta = 0.5, 0.1, np.pi/18 
        lambda_1_const, lambda_2_const = total_f_density/(1+iso), total_f_density*iso/(1+iso)
        Omega_init, fibre_coords, grid_x, grid_y = vAlignedNoise_fibre_constDensity_init(Delta_theta, lambda_1_const, lambda_2_const, x_min, x_max, y_min, y_max, gridsize_f, y_periodic_bool, x_periodic_bool)
    
        # parameters we vary:
        s_values = [1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128, 1/256, 1/512, 1/1024, 0.0]
        d_values = [1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128, 1/256, 1/512, 1/1024, 0.0]
        sd_params = np.array([np.array([s, d]) for s, d in itertools.product(s_values, d_values)])
        sd_params = sd_params[85:]
        seeds = np.arange(1, len(sd_params)*Numrep+1) 
        seeds = np.reshape(seeds, (len(sd_params), Numrep)) 

        for j in range(len(sd_params)):
            s, d = sd_params[j,0], sd_params[j,1]

            cell_coords = cell_coords_init.copy()
            Omega = Omega_init.copy()
            
            sub_path_tau = sub_path + f'deg{d}_sec{s}_ICdensity{total_f_density}_ICiso{iso}/'
            if not os.path.exists(sub_path_tau):
                os.makedirs(sub_path_tau)
            with multiprocessing.Pool(processes=numCPUs) as pool:
                arguments = [(cell_coords, fibre_coords, Omega, tend_iternum, delta_t, x_min, x_max, x_len, y_min, y_max, y_len, grid_x, grid_y, eta, D, sigma, r_max, epsilon, rep_adh_len, F_adh_max, beta, \
                        omega_0, s, d, tau, xi, Delta_0, rho_0, y_periodic_bool, x_periodic_bool, CG_bool, prolif_bool, int(seeds[j, rep]), sub_path_tau, rep, shift, scale, n) for rep in range(Numrep)]
                pool.starmap(one_run, arguments)


    if opt == 7: # initial void of fibres in a bigger domian

        Numrep, tend_iternum = 1, 5400

        sub_path = 'InitNOfibres_/'
        if not os.path.exists(sub_path):
            os.makedirs(sub_path)

        # parameters:
        x_min, x_max, y_min, y_max, y_periodic_bool, x_periodic_bool = 0.0, 540.0, 0.0, 540.0, True, True
        x_len, y_len = x_max-x_min, y_max-y_min
        delta_t = 1.0
        eta, sigma, epsilon, beta = 1.0, 12.0, 0.1, 1.0
        D = sigma/40
        r_max, rep_adh_len, F_adh_max = 3*sigma, (2**(1/6))*sigma, 24*epsilon*(1/sigma)*(2*((7/26)**(13/6))-(7/26)**(7/6))
        rho_0 = 6.0
        CG_bool, gridsize_f = True, 2.0
        omega_0, tau = 1.0, 300.0
        prolif_bool, Delta_0, xi = True, 1440.0, sigma/2  
        shift, scale, n = 0.4, 10.0, 1.0  
        s, d = 0.025, 0.0025

        # init cell coordinates in the domain centre
        centre = np.array([x_min+x_len/2, y_min+y_len/2])
        dist = 5 * sigma
        cell_coords_init = cell_invasion_disc_init(centre, dist, sigma)
        cell_coords = cell_coords_init.copy()

        # init no fibres: 
        Omega_init, fibre_coords, grid_x, grid_y = nofibre_init(x_min, x_max, y_min, y_max, gridsize_f, y_periodic_bool, x_periodic_bool)
        Omega = Omega_init.copy()

        seeds = np.arange(1, Numrep+1) 
        sub_path_tau = sub_path + f'sec{s}_deg{d}_xlen{x_len}_ylen{y_len}/'
        if not os.path.exists(sub_path_tau):
            os.makedirs(sub_path_tau)
        with multiprocessing.Pool(processes=numCPUs) as pool:
            arguments = [(cell_coords, fibre_coords, Omega, tend_iternum, delta_t, x_min, x_max, x_len, y_min, y_max, y_len, grid_x, grid_y, eta, D, sigma, r_max, epsilon, rep_adh_len, F_adh_max, beta, \
                    omega_0, s, d, tau, xi, Delta_0, rho_0, y_periodic_bool, x_periodic_bool, CG_bool, prolif_bool, int(seeds[0]), sub_path_tau, rep, shift, scale, n) for rep in range(Numrep)]
            pool.starmap(one_run, arguments)


    if opt == 8: # different D and epsilon values with quick diffusion and small shift

        Numrep, tend_iternum = 1, 3600

        sub_path = 'DiffusionEpsilon_/'
        if not os.path.exists(sub_path):
            os.makedirs(sub_path)

        # parameters:
        x_min, x_max, y_min, y_max, y_periodic_bool, x_periodic_bool = 0.0, 360.0, 0.0, 360.0, True, True
        x_len, y_len = x_max-x_min, y_max-y_min
        delta_t = 1.0
        eta, sigma, beta = 1.0, 12.0, 1.0
        rho_0 = 6.0
        CG_bool, gridsize_f = True, 2.0
        omega_0, tau = 1.0, 300.0
        prolif_bool, Delta_0, xi = True, 1440.0, sigma/2  
        shift, scale, n = 0.005, 10.0, 1.0  

        D_s = [sigma/10, sigma/40, sigma/160]
        epsilons = [0.05, 0.1, 0.2]
        parameters = np.array([np.array([d, e]) for d, e in itertools.product(D_s, epsilons)])
        seeds = np.arange(1, len(parameters)*Numrep+1) 
        seeds = np.reshape(seeds, (len(parameters), Numrep)) 


        # init cell coordinates in the domain centre
        centre = np.array([x_min+x_len/2, y_min+y_len/2])
        dist = 5 * sigma
        cell_coords_init = cell_invasion_disc_init(centre, dist, sigma)

        # init little fibres: 
        lambda_12_const = 0.005
        lambda_sum = lambda_12_const * 2
        d = 0.0005
        s = d * (lambda_sum / (1-lambda_sum))
        Omega_init, fibre_coords, grid_x, grid_y = isotropic_fibre_constDensity_init(lambda_12_const, x_min, x_max, y_min, y_max, gridsize_f, y_periodic_bool, x_periodic_bool)

        for j in range(len(parameters)):
            D, epsilon = parameters[j,0], parameters[j,1]
            r_max, rep_adh_len, F_adh_max = 3*sigma, (2**(1/6))*sigma, 24*epsilon*(1/sigma)*(2*((7/26)**(13/6))-(7/26)**(7/6))
            sub_path_tau = sub_path + f'Diffusion{D}_Epsilon{epsilon}/'
            if not os.path.exists(sub_path_tau):
                os.makedirs(sub_path_tau)

            cell_coords = cell_coords_init.copy()
            Omega = Omega_init.copy()

            with multiprocessing.Pool(processes=numCPUs) as pool:
                arguments = [(cell_coords, fibre_coords, Omega, tend_iternum, delta_t, x_min, x_max, x_len, y_min, y_max, y_len, grid_x, grid_y, eta, D, sigma, r_max, epsilon, rep_adh_len, F_adh_max, beta, \
                        omega_0, s, d, tau, xi, Delta_0, rho_0, y_periodic_bool, x_periodic_bool, CG_bool, prolif_bool, int(seeds[0]), sub_path_tau, rep, shift, scale, n) for rep in range(Numrep)]
                pool.starmap(one_run, arguments)

    
    if opt == 9: # different D and epsilon values with slow diffusion and small shift

        Numrep, tend_iternum = 1, 3600
        lambda_12_const = 0.005
        lambda_sum = lambda_12_const * 2
        d = 0.0005
        s = 0.05

        sub_path = f'DiffusionEpsilon_sec{s}_deg{d}_normalprolif/'
        #sub_path = 'DiffusionEpsilon_slowProlif_/'
        if not os.path.exists(sub_path):
            os.makedirs(sub_path)

        # parameters:
        x_min, x_max, y_min, y_max, y_periodic_bool, x_periodic_bool = 0.0, 360.0, 0.0, 360.0, True, True
        x_len, y_len = x_max-x_min, y_max-y_min
        delta_t = 1.0
        eta, sigma, beta = 1.0, 12.0, 1.0
        rho_0 = 6.0
        CG_bool, gridsize_f = True, 2.0
        omega_0, tau = 1.0, 300.0
        prolif_bool, Delta_0, xi = True, 1440.0, sigma/2  
        #Delta_0 = 1440.0 * 2.5
        shift, scale, n = 0.005, 10.0, 1.0  

        D_s = [sigma/40, sigma/160]
        epsilons = [0.05]
        parameters = np.array([np.array([d, e]) for d, e in itertools.product(D_s, epsilons)])
        seeds = np.arange(1, len(parameters)*Numrep+1) 
        seeds = np.reshape(seeds, (len(parameters), Numrep)) 


        # init cell coordinates in the domain centre
        centre = np.array([x_min+x_len/2, y_min+y_len/2])
        dist = 5 * sigma
        cell_coords_init = cell_invasion_disc_init(centre, dist, sigma)

        # init little fibres: 
        #d = 0.0005
        #s = d * (lambda_sum / (1-lambda_sum))
        Omega_init, fibre_coords, grid_x, grid_y = isotropic_fibre_constDensity_init(lambda_12_const, x_min, x_max, y_min, y_max, gridsize_f, y_periodic_bool, x_periodic_bool)

        for j in range(len(parameters)):
            D, epsilon = parameters[j,0], parameters[j,1]
            r_max, rep_adh_len, F_adh_max = 3*sigma, (2**(1/6))*sigma, 24*epsilon*(1/sigma)*(2*((7/26)**(13/6))-(7/26)**(7/6))
            
            sub_path_tau = sub_path + f'Diffusion{D}_Epsilon{epsilon}/'
            if not os.path.exists(sub_path_tau):
                os.makedirs(sub_path_tau)

            cell_coords = cell_coords_init.copy()
            Omega = Omega_init.copy()

            with multiprocessing.Pool(processes=numCPUs) as pool:
                arguments = [(cell_coords, fibre_coords, Omega, tend_iternum, delta_t, x_min, x_max, x_len, y_min, y_max, y_len, grid_x, grid_y, eta, D, sigma, r_max, epsilon, rep_adh_len, F_adh_max, beta, \
                        omega_0, s, d, tau, xi, Delta_0, rho_0, y_periodic_bool, x_periodic_bool, CG_bool, prolif_bool, int(seeds[0]), sub_path_tau, rep, shift, scale, n) for rep in range(Numrep)]
                pool.starmap(one_run, arguments)

        
    if opt == 10: # different memory length and small shift

        Numrep, tend_iternum = 1, 3600

        #d = 0.0005
        #s = 0.05
        lambda_12_const = 0.005
        lambda_sum = lambda_12_const * 2
        d = 0.5
        s = d * (lambda_sum / (1-lambda_sum))

        #sub_path = 'ShiftTau_/'
        sub_path = f'ShiftTau_sec{s}_deg{d}/'
        if not os.path.exists(sub_path):
            os.makedirs(sub_path)

        # parameters:
        x_min, x_max, y_min, y_max, y_periodic_bool, x_periodic_bool = 0.0, 360.0, 0.0, 360.0, True, True
        x_len, y_len = x_max-x_min, y_max-y_min
        delta_t = 1.0
        eta, sigma, epsilon, beta = 1.0, 12.0, 0.1, 1.0
        D = sigma/40
        r_max, rep_adh_len, F_adh_max = 3*sigma, (2**(1/6))*sigma, 24*epsilon*(1/sigma)*(2*((7/26)**(13/6))-(7/26)**(7/6))
        rho_0 = 6.0
        CG_bool, gridsize_f = True, 2.0
        omega_0 = 1.0
        prolif_bool, Delta_0, xi = True, 1440.0, sigma/2  
        scale, n = 10.0, 1.0  

        shift_s = [0.005, 0.5, 0.9]
        tau_s = [1, 300, 1200]
        parameters = np.array([np.array([s, t]) for s, t in itertools.product(shift_s, tau_s)])
        seeds = np.arange(1, len(parameters)*Numrep+1) 
        seeds = np.reshape(seeds, (len(parameters), Numrep)) 


        # init cell coordinates in the domain centre
        centre = np.array([x_min+x_len/2, y_min+y_len/2])
        dist = 5 * sigma
        cell_coords_init = cell_invasion_disc_init(centre, dist, sigma)

        # init little fibres: 
        lambda_12_const = 0.005
        lambda_sum = lambda_12_const * 2
        #d = 0.0005
        #s = d * (lambda_sum / (1-lambda_sum))
        #s = 0.05
        Omega_init, fibre_coords, grid_x, grid_y = isotropic_fibre_constDensity_init(lambda_12_const, x_min, x_max, y_min, y_max, gridsize_f, y_periodic_bool, x_periodic_bool)

        for j in range(len(parameters)):
            shift, tau = parameters[j,0], parameters[j,1]
            sub_path_tau = sub_path + f'shift{shift}_tau{tau}/'
            if not os.path.exists(sub_path_tau):
                os.makedirs(sub_path_tau)

            cell_coords = cell_coords_init.copy()
            Omega = Omega_init.copy()

            with multiprocessing.Pool(processes=numCPUs) as pool:
                arguments = [(cell_coords, fibre_coords, Omega, tend_iternum, delta_t, x_min, x_max, x_len, y_min, y_max, y_len, grid_x, grid_y, eta, D, sigma, r_max, epsilon, rep_adh_len, F_adh_max, beta, \
                        omega_0, s, d, tau, xi, Delta_0, rho_0, y_periodic_bool, x_periodic_bool, CG_bool, prolif_bool, int(seeds[0]), sub_path_tau, rep, shift, scale, n) for rep in range(Numrep)]
                pool.starmap(one_run, arguments)


    pass


if __name__ == "__main__":

    opt = int(sys.argv[1]) 
    numCPUs = int(sys.argv[2]) 
    #Numrep, tend_iternum = 40, 3600 # for options 1, 2, 3, 4, 5, 6
    #Numrep, tend_iternum = 1, 5400 # for options 7 only
    Numrep, tend_iternum = 1, 3600 # for options 8, 9, 10

    cell_invasion(Numrep, tend_iternum, opt, numCPUs)