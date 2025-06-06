import pickle
import numpy as np
import sys
from numpy import linalg as LA
from numpy.linalg import norm 
import multiprocessing
import itertools
from scipy.spatial import distance



def total_cellnum(overall_folder, j, input_path, Numrep, tend_iternum):

    total_cellnum_times_allreps = np.zeros((Numrep, tend_iternum), dtype=float)

    for rep in range(Numrep):
        load_filename = input_path + f'rep_{rep}.pickle' 
        with open(load_filename, 'rb') as file:
            cell_coords_T = pickle.load(file)
        for t in range(tend_iternum):
            total_cellnum_times_allreps[rep, t] = len(cell_coords_T[t])

    output_filename = overall_folder + j + 'totalCellNum_alltimes.txt'
    np.savetxt(output_filename, total_cellnum_times_allreps, delimiter=' ', fmt='%.4f')  
    pass


def x_density_span(overall_folder, j, input_path, Numrep, x_middist, t_extract_times):
    for t in t_extract_times:
        cellnum_avex_allreps = np.zeros((Numrep, len(x_middist)), dtype=float)
        for rep in range(Numrep):
            load_filename = input_path + f'rep_{rep}.pickle' 
            with open(load_filename, 'rb') as file:
                cell_coords_T = pickle.load(file)
            cell_xcoords_t = cell_coords_T[t][:, 0]
            cellnum_avex_allreps[rep, :], _ = np.histogram(cell_xcoords_t, bins=bin_x, density=False)

        output_filename = overall_folder + j + f'cellnum_avex_allreps_t{t}.txt'
        np.savetxt(output_filename, cellnum_avex_allreps, delimiter=' ', fmt='%.4f')  
    pass


def extract_fibre_info(Omega_reshape_t, init_density):
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
                aisotropy.append(1-lambda_2/lambda_1)
            else:
                aisotropy.append(np.nan)
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


def fibre_summarystats_LR(overall_folder, index_j, j, input_path, fibre_x_2D, fibre_y_2D, x_mid, y_mid, LR_mask, Numrep, init_density, t_extract_times):
    # extract fibre summary info in the left and the right regions of the domain:

    if len(init_density) > 1: # this is for CG with different initial densities:
        init_density_ = init_density[index_j]
    else: # all the other simulations:
        init_density_ = init_density[0]

    for t in t_extract_times:
        # init:
        total_density_t_allreps_LR = []
        aisotropy_t_allreps_LR = []
        v1_degree_t_allreps_LR = []
        frac_changed_t_allreps_LR = []

        for rep in range(Numrep):
            load_filename = input_path + f'rep_{rep}.pickle' 
            with open(load_filename, 'rb') as file:
                _ = pickle.load(file)
                Omega_T = pickle.load(file)
            Omega_t = Omega_T[t]

            Omega_t_LR = Omega_t[LR_mask]
            total_elements_LR = Omega_t_LR.size
            nm_LR = int(total_elements_LR/4)
            Omega_t_reshape_LR = Omega_t_LR.reshape((nm_LR, 2, 2))
            total_density_LR, aisotropy_LR, v1_degree_LR, frac_changed_LR = extract_fibre_info(Omega_t_reshape_LR, init_density_)

            # store for LR: 
            total_density_t_allreps_LR.append(total_density_LR)
            aisotropy_t_allreps_LR.append(aisotropy_LR)
            v1_degree_t_allreps_LR.append(v1_degree_LR)
            frac_changed_t_allreps_LR.append(frac_changed_LR)

        output_filename = overall_folder + j + f'fibreInfo_LR_t{t}.pickle'
        with open(output_filename, 'wb') as file:
            pickle.dump(total_density_t_allreps_LR, file)
            pickle.dump(aisotropy_t_allreps_LR, file)
            pickle.dump(v1_degree_t_allreps_LR, file)
            pickle.dump(frac_changed_t_allreps_LR, file)
    pass


def fibre_summarystats_TB(overall_folder, j, input_path, fibre_x_2D, fibre_y_2D, x_mid, y_mid, TB_mask, Numrep, init_density, t_extract_times):
    # extract fibre summary info in the left and the right regions of the domain:

    if len(init_density) > 1: # this is for CG with different initial densities:
        init_density_ = init_density[j]
    else: # all the other simulations:
        init_density_ = init_density[0]

    for t in t_extract_times:
        # init:
        total_density_t_allreps_TB = []
        aisotropy_t_allreps_TB = []
        v1_degree_t_allreps_TB = []
        frac_changed_t_allreps_TB = []

        for rep in range(Numrep):
            load_filename = input_path + f'rep_{rep}.pickle' 
            with open(load_filename, 'rb') as file:
                _ = pickle.load(file)
                Omega_T = pickle.load(file)
            Omega_t = Omega_T[t]

            Omega_t_TB = Omega_t[TB_mask]
            total_elements_TB = Omega_t_TB.size
            nm_TB = int(total_elements_TB/4)
            Omega_t_reshape_TB = Omega_t_TB.reshape((nm_TB, 2, 2))
            total_density_TB, aisotropy_TB, v1_degree_TB, frac_changed_TB = extract_fibre_info(Omega_t_reshape_TB, init_density_)

            # store for LR: 
            total_density_t_allreps_TB.append(total_density_TB)
            aisotropy_t_allreps_TB.append(aisotropy_TB)
            v1_degree_t_allreps_TB.append(v1_degree_TB)
            frac_changed_t_allreps_TB.append(frac_changed_TB)

        output_filename = overall_folder + j + f'fibreInfo_TB_t{t}.pickle'
        with open(output_filename, 'wb') as file:
            pickle.dump(total_density_t_allreps_TB, file)
            pickle.dump(aisotropy_t_allreps_TB, file)
            pickle.dump(v1_degree_t_allreps_TB, file)
            pickle.dump(frac_changed_t_allreps_TB, file)
    pass


def fibre_summarystats_ALL(overall_folder, j, input_path, fibre_x_2D, fibre_y_2D, x_mid, y_mid, Numrep, init_density, t_extract_times):
    # extract fibre summary info in the left and the right regions of the domain:

    distances = np.sqrt((fibre_x_2D-x_mid)**2 + (fibre_y_2D-y_mid)**2)

    if len(init_density) > 1: # this is for CG with different initial densities:
        init_density_ = init_density[j]
    else: # all the other simulations:
        init_density_ = init_density[0]

    for t in t_extract_times:
        # init:
        total_density_t_allreps = []
        aisotropy_t_allreps = []
        v1_degree_t_allreps = []
        frac_changed_t_allreps = []

        for rep in range(Numrep):
            load_filename = input_path + f'rep_{rep}.pickle' 
            with open(load_filename, 'rb') as file:
                _ = pickle.load(file)
                Omega_T = pickle.load(file)
            Omega_t = Omega_T[t]

            total_elements_all = Omega_t.size
            nm_all = int(total_elements_all/4)
            Omega_reshape_t = Omega_t.reshape((nm_all, 2, 2))
            distances_reshape = distances.reshape(nm_all)

            total_density, aisotropy, v1_degree, frac_changed = extract_fibre_info_withdistance(Omega_reshape_t, init_density_, distances_reshape)

            # store for LR: 
            total_density_t_allreps.append(total_density)
            aisotropy_t_allreps.append(aisotropy)
            v1_degree_t_allreps.append(v1_degree)
            frac_changed_t_allreps.append(frac_changed)

        output_filename = overall_folder + j + f'fibreInfo_t{t}.pickle'
        with open(output_filename, 'wb') as file:
            pickle.dump(total_density_t_allreps, file)
            pickle.dump(aisotropy_t_allreps, file)
            pickle.dump(v1_degree_t_allreps, file)
            pickle.dump(frac_changed_t_allreps, file)
    pass


def KL_xavespan(overall_folder, secdeg_parametersweep_subfolders2d):

    KLmatrix_xavespan = np.zeros((11, 11), dtype=float)

    t_extract_time = 21

    extract_index = 0

    for sd_index1 in range(11):
        # check the progress on ARC
        print(extract_index)
        extract_index = extract_index + 1

        sd = secdeg_parametersweep_subfolders2d[sd_index1]

        # for each col of the KL matrix, compare with the top row, i.e. d=0 case:
        inputpath_d0 = overall_folder + sd[-1] + 'cellnum_avex_allreps_t21.txt'
        xavespan_d0 = np.loadtxt(inputpath_d0, delimiter=' ')
        xavespan_d0 = np.mean(xavespan_d0, axis=0)

        for sd_index2 in range(11):
            inputpath_d = overall_folder + sd[sd_index2] + 'cellnum_avex_allreps_t21.txt'
            xavespan_d = np.loadtxt(inputpath_d, delimiter=' ')
            xavespan_d = np.mean(xavespan_d, axis=0)

            # compute KL divergence:
            KLmatrix_xavespan[sd_index2, sd_index1] = distance.jensenshannon(xavespan_d0, xavespan_d)

    output_filename = overall_folder + f'KL_xavespan_t{t_extract_time}.pickle'
    with open(output_filename, 'wb') as file:
        pickle.dump(KLmatrix_xavespan, file)
    pass


if __name__ == "__main__": # FOR CELL INVASION SHOWCASE:

    # For Figures 3 and onwards:
    Numrep = 40
    overall_folders = ['fibredensity_/', 
                       'fibreisotropy_/', 
                       'degradation_/', 
                       'secretion_/', 
                       'SI_degradation_secretion_/', 
                       'secdeg_parametersweep_/']
    subfolders = [['VerticalFibre_totaldensity0.0_iso0.1/', 
                   'VerticalFibre_totaldensity0.1_iso0.1/', 
                   'VerticalFibre_totaldensity0.3_iso0.1/', 
                   'VerticalFibre_totaldensity0.4_iso0.1/', 
                   'VerticalFibre_totaldensity0.5_iso0.1/', 
                   'VerticalFibre_totaldensity1.0_iso0.1/'], 

                  ['VerticalFibre_totaldensity0.8_iso1.0/', 
                   'VerticalFibre_totaldensity0.8_iso0.75/', 
                   'VerticalFibre_totaldensity0.8_iso0.5/', 
                   'VerticalFibre_totaldensity0.8_iso0.25/', 
                   'VerticalFibre_totaldensity0.8_iso0.1/', 
                   'VerticalFibre_totaldensity0.8_iso0.05/'], 

                  ['deg0.0025_ICdensity0.9_ICiso0.1/', 
                   'deg0.25_ICdensity0.9_ICiso0.1/'], 

                  ['sec0.0025_ICdensity0.1_ICiso0.1/', 
                   'sec0.005_ICdensity0.1_ICiso0.1/', 
                   'sec0.025_ICdensity0.1_ICiso0.1/', 
                   'sec0.25_ICdensity0.1_ICiso0.1/'], 

                  ['deg0.0025_sec0.0_ICdensity0.5_ICiso0.1/', 
                   'deg0.25_sec0.0_ICdensity0.5_ICiso0.1/', 
                   'deg0.0_sec0.0005_ICdensity0.5_ICiso0.1/', 
                   'deg0.0_sec0.005_ICdensity0.5_ICiso0.1/',
                   'deg0.0_sec0.05_ICdensity0.5_ICiso0.1/', 
                   'deg0.0_sec0.5_ICdensity0.5_ICiso0.1/']
                  ]
    s_values = [1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128, 1/256, 1/512, 1/1024, 0.0]
    d_values = [1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128, 1/256, 1/512, 1/1024, 0.0]
    sd_params = np.array([np.array([s, d]) for s, d in itertools.product(s_values, d_values)])
    secdeg_parametersweep_subfolders = [f'deg{d}_sec{s}_ICdensity0.5_ICiso0.1/' for s, d in sd_params]
    subfolders.append(secdeg_parametersweep_subfolders)

    init_densities = [[0.0, 0.1, 0.3, 0.4, 0.5, 1.0], 
                      [0.8], 
                      [0.9], 
                      [0.1], 
                      [0.5], 
                      [0.5]]

    y_min, y_max, sigma, x_min, x_max = 0.0, 360.0, 12.0, 0.0, 360.0
    y_periodic_bool, x_periodic_bool, gridsize_f = True, True, 2.0
    x_len, y_len = x_max-x_min, y_max-y_min
    x_mid, y_mid = x_min+x_len/2, y_min+y_len/2
    centre = np.array([x_mid, y_mid])
    bin_len = sigma * 2
    bin_x = np.arange(x_min, x_max+bin_len, bin_len)
    x_middist = (bin_x+bin_len/2)[0 : -1]
    if y_periodic_bool:
        grid_y = np.arange(y_min, y_max, gridsize_f) 
    else:
        grid_y = np.arange(y_min, y_max+gridsize_f, gridsize_f) 
    if x_periodic_bool:
        grid_x = np.arange(x_min, x_max, gridsize_f) 
    else:
        grid_x = np.arange(x_min, x_max+gridsize_f, gridsize_f) 
    _, Y = np.meshgrid(grid_x, grid_y)
    yy_1d = Y.ravel()
    fibre_x_2D, fibre_y_2D = np.meshgrid(grid_x, grid_y, indexing="xy")

    # masks for the top, middle, and bottom regions of the domain:
    x_third_len, y_third_len = x_len/3, y_len/3
    x_mid_min, x_mid_max = x_min+x_third_len, x_max-x_third_len
    y_mid_min, y_mid_max = y_min+y_third_len, y_max-y_third_len
    top_middle_mask = (fibre_x_2D>=x_mid_min) & (fibre_x_2D<=x_mid_max) & (fibre_y_2D>=y_mid_max)
    bottom_middle_mask = (fibre_x_2D>=x_mid_min) & (fibre_x_2D<=x_mid_max) & (fibre_y_2D<=y_min+y_third_len)
    left_middle_mask = (fibre_x_2D<=x_min+x_third_len) & (fibre_y_2D>=y_mid_min) & (fibre_y_2D<=y_mid_max)
    right_middle_mask = (fibre_x_2D>=x_max-x_third_len) & (fibre_y_2D>=y_mid_min) & (fibre_y_2D<=y_mid_max)
    TB_mask, LR_mask = np.logical_or(top_middle_mask, bottom_middle_mask), np.logical_or(left_middle_mask, right_middle_mask)

    tend_iternum, t_extract_times = 31, [21]

    folder_toextract = [5]
    for i in folder_toextract:
        overall_folder = overall_folders[i]
        subfolder = subfolders[i]
        init_density = init_densities[i]
        KLmatrix_xavespan = np.zeros((11, 11), dtype=float)

        index_j = 0
        for j in subfolder:
            input_path = overall_folder + j
            print(input_path)

            # average span across x direction:
            x_density_span(overall_folder, j, input_path, Numrep, x_middist, t_extract_times)

            # cell population over time:
            #total_cellnum(overall_folder, j, input_path, Numrep, tend_iternum)

            # fibre dynamics in the LR regions: 
            #fibre_summarystats_LR(overall_folder, index_j, j, input_path, fibre_x_2D, fibre_y_2D, x_mid, y_mid, LR_mask, Numrep, init_density, t_extract_times)
            index_j += 1
            
        
        #if result == 2: 
            #numCPUs = int(sys.argv[1]) 
            #CG_strength(option_folders, option_subfold
            
    # -------------------------------------------- #
    # For Figure 7 (parameter sweep given different secretion and degradation rates)
    overall_folder = 'secdeg_parametersweep_/'
    secdeg_parametersweep_subfolders2d = []
    for ss in s_values:
        ss_value = [ss]
        ss_list = sorted(
            [f'deg{d}_sec{s}_ICdensity0.5_ICiso0.1/' for d, s in itertools.product(d_values, ss_value)],
            key=lambda x: float(x.split('_')[1][3:])  # Extracting the value of d
        )
        secdeg_parametersweep_subfolders2d.append(ss_list)
    
    KL_xavespan(overall_folder, secdeg_parametersweep_subfolders2d)





    # -------------------------------------------- #
    # For Figure 2 (initial void of fibres with a bigger domain)
    Numrep = 1
    overall_folders = ['InitNOfibres_/', 
                       ]
    subfolders = [['sec0.025_deg0.0025_xlen540.0_ylen540.0/'], 
                  ]
    init_densities = [[0.0], 
                      ]

    y_min, y_max, sigma, x_min, x_max = 0.0, 540.0, 12.0, 0.0, 540.0
    y_periodic_bool, x_periodic_bool, gridsize_f = True, True, 2.0
    x_len, y_len = x_max-x_min, y_max-y_min
    x_mid, y_mid = x_min+x_len/2, y_min+y_len/2
    centre = np.array([x_mid, y_mid])
    bin_len = sigma * 2
    bin_x = np.arange(x_min, x_max+bin_len, bin_len)
    x_middist = (bin_x+bin_len/2)[0 : -1]
    if y_periodic_bool:
        grid_y = np.arange(y_min, y_max, gridsize_f) 
    else:
        grid_y = np.arange(y_min, y_max+gridsize_f, gridsize_f) 
    if x_periodic_bool:
        grid_x = np.arange(x_min, x_max, gridsize_f) 
    else:
        grid_x = np.arange(x_min, x_max+gridsize_f, gridsize_f) 
    _, Y = np.meshgrid(grid_x, grid_y)
    yy_1d = Y.ravel()
    fibre_x_2D, fibre_y_2D = np.meshgrid(grid_x, grid_y, indexing="xy")

    # masks for the top, middle, and bottom regions of the domain:
    x_third_len, y_third_len = x_len/3, y_len/3
    x_mid_min, x_mid_max = x_min+x_third_len, x_max-x_third_len
    y_mid_min, y_mid_max = y_min+y_third_len, y_max-y_third_len
    top_middle_mask = (fibre_x_2D>=x_mid_min) & (fibre_x_2D<=x_mid_max) & (fibre_y_2D>=y_mid_max)
    bottom_middle_mask = (fibre_x_2D>=x_mid_min) & (fibre_x_2D<=x_mid_max) & (fibre_y_2D<=y_min+y_third_len)
    left_middle_mask = (fibre_x_2D<=x_min+x_third_len) & (fibre_y_2D>=y_mid_min) & (fibre_y_2D<=y_mid_max)
    right_middle_mask = (fibre_x_2D>=x_max-x_third_len) & (fibre_y_2D>=y_mid_min) & (fibre_y_2D<=y_mid_max)
    TB_mask, LR_mask = np.logical_or(top_middle_mask, bottom_middle_mask), np.logical_or(left_middle_mask, right_middle_mask)

    t_extract_times = [3, 21, 36]

    folder_toextract = [0]  
    for i in folder_toextract:
        overall_folder = overall_folders[i]
        subfolder = subfolders[i]
        init_density = init_densities[i]

        for j in subfolder:
            input_path = overall_folder + j

            # fibre dynamics in the LR regions: 
            #fibre_summarystats_LR(overall_folder, j, input_path, fibre_x_2D, fibre_y_2D, x_mid, y_mid, LR_mask, Numrep, init_density, t_extract_times)
            # fibre dynamics in the TB regions: 
            #fibre_summarystats_TB(overall_folder, j, input_path, fibre_x_2D, fibre_y_2D, x_mid, y_mid, TB_mask, Numrep, init_density, t_extract_times)
            # fibre dynamics in ALL regions: 
            #fibre_summarystats_ALL(overall_folder, j, input_path, fibre_x_2D, fibre_y_2D, x_mid, y_mid, Numrep, init_density, t_extract_times)
            
    



    




    
    

    

