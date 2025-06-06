# This script contains all subfunctions used in 'migration_main.py'
# MODEL assumptions to ask: search 'CHECK'

import numpy as np
from scipy.interpolate import RectBivariateSpline
from numpy.linalg import norm 


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


def total_F_cc(dist, orientation, sigma, epsilon, r_max, rep_adh_len, delta_t):
    ''' 
    This function calculates the inter-cellular forces for each cell based on their pairwise distances and orientations
    INPUT:
    'dist': a 2D np symmetrix array of size N*N storing pairwise cell distance
    'orientation': a 2D np array of size N*N storing pairwise cell orientation
    'sigma': the constant cell diameter uniform across the cell population
    'epsilon': depth of the Lennard Jone's potential 
    'r_max': the finate range of cell-cell interactions
    'rep_adh_len': the characteristic length representing the balance between repulsion and adhesion
    OUTPUT:
    'F_cc': a 2D np array of size N*2 storing the inter-cellular forces experienced by each cell
    'total_adh_magnitude': a 1D np array of size N storing the magnitudes of total adhesion force felt by cells
    'rep_dist_': a 2D np array of size N*N storing the distance to 'rep_adh_len' for repulsive neighbours
    'num_rep_neigh': a 1D np array of size N storing the number of repulsive neighbours
    'num_adh_neigh': a 1D np array of size N storing the number of adhesive neighbours

    DEBUGGED
    '''

    # initialise the magnitude of forces exposed by neighbouring cells for all the N cells:
    cc_magnitude = np.zeros((len(dist), len(dist)), dtype=float)
    
    # calculate the magnitude of forces based on the Lennard Jone's potential:
    mask = dist <= r_max
    np.fill_diagonal(dist, np.nan) # no inter-cellular force on a cell due to the presence of itself
    cc_magnitude[mask] = (2 * (sigma**6)/(dist[mask]**6) - 1) * 24 * epsilon * (sigma**6) / (dist[mask]**7)
    # for numerical stability, if abs(cc_magnitude) > sigma/2, then manually set abs(vel_t) = sigma/2
    too_large_mask, too_small_mask = cc_magnitude > 0.1*sigma, cc_magnitude < -0.1*sigma
    cc_magnitude[too_large_mask], cc_magnitude[too_small_mask] = 0.1*sigma, -0.1*sigma

    # obtain the orientation, thus the vector representation, of forces:
    # x-direction: 
    cc_dx_cells = np.multiply(cc_magnitude, np.cos(orientation))
    np.fill_diagonal(cc_dx_cells, np.nan) 
    cc_dxpart = np.nansum(cc_dx_cells, axis=1) # sum over all its neighbouring cells
    # y-direction:
    cc_dy_cells = np.multiply(cc_magnitude, np.sin(orientation))
    np.fill_diagonal(cc_dy_cells, np.nan)
    cc_dypart = np.nansum(cc_dy_cells, axis=1) 

    F_cc = np.column_stack([cc_dxpart, cc_dypart])

    # deal with values that are super close to zero (to avoid floating point error):
    small_mask = np.abs(F_cc) <= 10**(-10)
    F_cc[small_mask] = 0.0

    # count the number of repulsive and adhesive neighbours:
    rep_mask, adh_mask = dist<=rep_adh_len, (dist>rep_adh_len)&(dist <= r_max)
    num_rep_neigh, num_adh_neigh = np.nansum(rep_mask,axis=1), np.nansum(adh_mask,axis=1)
    # the 'np.nansum' in calculating 'num_rep_neigh' is because: a cell isn't itself's repulsive neighbour.

    # for the usage in 'CGcc_Pc': --------------------------------------------------------
    # output the total adhesion forces felt by cells:
    adh_magnitude = np.zeros((len(dist), len(dist)), dtype=float)
    adh_magnitude[adh_mask] = (2 * (sigma**6) / (dist[adh_mask]**6) - 1) * 24 * epsilon * (sigma**6) / (dist[adh_mask]**7)
    total_adh_magnitude = np.nansum(adh_magnitude, axis=1)
    # we don't need to care about the cell to itself, as the distance to itself is zero, falling outside the adhesive range.

    # output the distance to 'rep_adh_len' for repulsive neighbouring cells:
    rep_dist_ = np.zeros((len(dist), len(dist)), dtype=float)
    rep_dist_[rep_mask] = 1 - dist[rep_mask]/rep_adh_len
    np.fill_diagonal(rep_dist_, 0)
    # -------------------------------------------------------------------------------------

    return F_cc, total_adh_magnitude, rep_dist_, num_rep_neigh, num_adh_neigh


def total_F_rm(D, n, delta_t):
    '''
    This function samples from a 2D white noise and obtains random motion of with macroscopic coefficient 'D' for the cell population.
    INPUT:
    'D': the macroscopic coefficient
    'n': number of cells at time t
    'delta_t': numerical time step 
    OUTPUT:
    'F_rm': a 2D np array storing random forces experienced by each cell.

    DEBUGGED
    '''

    F_rm = np.random.multivariate_normal(mean=np.zeros(2), cov=2*D*np.eye(2)/delta_t, size=n)

    return F_rm


def fibre_cell_locs(grid_x, grid_y, Omega, cell_coords, N):
    '''
    This function interpolates the fibre field 'Omega' at cell central locations, it also returns the 
    total fibre space-filling percentages at those locations.
    INPUT:
    'grid_x': a 1D np array [x_min, x_min+Delta_x, x_min+2*Delta_x, ..., x_max] of size 'num_col'
    'grid_y': a 1D np array [y_min, y_min+Delta_y, y_min+2*Delta_y, ..., y_max] of size 'num_row'
    'Omega': a 4D np array of size (num_rows, num_cols, 2, 2) storing fibre tensor at each grid point
    'cell_coords': a 2D np array of size N*2 storing N cell coordinates
    'N': cell population at time t

    OUTPUT:
    'Omega_cell_loc': a 3D np array of size (N, 2, 2) storing fibre tensor at each cell location
    'total_fibre_cell_loc': a 1D np array of size N storing total fibre space-filling percentages at cell locations

    DEBUGGED
    '''
    # create spline interpolation objects for the 00, 01, 10, and 11 entries of Omega on the grid defined:
    # (we set kx=ky=1 for bilinear interpolation, so that there is no impact on sign of the value interpolated)
    interpolator_00 = RectBivariateSpline(grid_x, grid_y, Omega[:, :, 0, 0].T, kx=1, ky=1)
    interpolator_01 = RectBivariateSpline(grid_x, grid_y, Omega[:, :, 0, 1].T, kx=1, ky=1)
    interpolator_10 = RectBivariateSpline(grid_x, grid_y, Omega[:, :, 1, 0].T, kx=1, ky=1)
    interpolator_11 = RectBivariateSpline(grid_x, grid_y, Omega[:, :, 1, 1].T, kx=1, ky=1)

    # interpolate the 00, 01, 10, and 11 entries of Omega at the cell locations:
    # (we set 'grid=False' to interpolate values at these points, rather than on the grid defined by these points)
    Omega_00_interp = interpolator_00(cell_coords[:, 0], cell_coords[:, 1], grid=False)
    Omega_01_interp = interpolator_01(cell_coords[:, 0], cell_coords[:, 1], grid=False)
    Omega_10_interp = interpolator_10(cell_coords[:, 0], cell_coords[:, 1], grid=False)
    Omega_11_interp = interpolator_11(cell_coords[:, 0], cell_coords[:, 1], grid=False)

    # stack and resize:
    Omega_cell_loc = np.stack((Omega_00_interp, Omega_01_interp, Omega_10_interp, Omega_11_interp), axis=-1)
    Omega_cell_loc = Omega_cell_loc.reshape(N, 2, 2)
    
    total_fibre_cell_loc = np.trace(Omega_cell_loc, axis1=1, axis2=2)

    return Omega_cell_loc, total_fibre_cell_loc


# Define the custom hyperbolic tanh function
def scaled_tanh(x, shift, scale):
    # Hyperbolic tanh, shifted and scaled
    return 0.5 * (np.tanh(scale * (x - shift)) + 1)
# Adjust the function to ensure it starts at 0 and ends at 1
def adjusted_tanh(x, shift, scale):
    tanh_value = scaled_tanh(x, shift, scale)
    # Adjusting so that the curve starts at 0 at x=0 and reaches 1 at x=1
    return (tanh_value - scaled_tanh(0, shift, scale)) / (scaled_tanh(1, shift, scale) - scaled_tanh(0, shift, scale))

def CG_rand(total_fibre_cell_loc, F_rm, Omega_cell_loc, N, shift, scale, n):
    '''
    This function calculates the fibre contact guidance matrix associated with random motion for each cell. 
    INPUT:
    'total_fibre_cell_loc': a 1D np array of size N storing total fibre space-filling percentages at cell locations
    'F_rm': a 2D np array storing random forces experienced by each cell
    'Omega_cell_loc': a 3D np array of size (N, 2, 2) storing fibre tensor at each cell location
    'N': cell population at time t
    OUTPUT:
    'M_rand_hat': a 3D np array of size (N, 2, 2) storing the normalised contact guidance matrix at cell locations
    
    DEBUGGED
    '''

    # find the length-preserving Omega at cell locations:
    Omega_hat = np.zeros((N, 2, 2), dtype = float) 
    l_rand = norm(np.squeeze(Omega_cell_loc@F_rm[:,:,np.newaxis], axis=2), axis=1) # matrix vector multiplication length
    lnonzero_mask = np.abs(l_rand) > 10**(-15) # we only care about non-zero motility vector after contact guidance
    Omega_hat[lnonzero_mask] = (Omega_cell_loc[lnonzero_mask]) * ((norm(F_rm,axis=1)[lnonzero_mask]/l_rand[lnonzero_mask])[:,np.newaxis,np.newaxis]) # normalise
    # When F_rm is in the direction of the v2 but lambda2=0 (thus CG will modulate velocity vectors into points)
    zero_mask = np.logical_and(np.abs(l_rand)<=10**(-15), norm(F_rm,axis=1)>10**(-15)) 
    zero_mask = np.logical_and(zero_mask, np.any(np.abs(Omega_cell_loc)>10**(-15), axis=(1, 2))) # there is fibres
    zero_indices = np.where(zero_mask)[0]
    # the strength of contact guidance on a cell's random motion is weighted by the space-filling degree of fibres at the cell location:
    # CG's strength is linearly dependent on total area density (NOT OBVIOUS RESULTS)
    #M_rand = total_fibre_cell_loc[:,np.newaxis,np.newaxis] * Omega_hat + \
    #    (1-total_fibre_cell_loc[:,np.newaxis,np.newaxis]) * np.eye(2, dtype=float)
    # TRY NONLINEAR DEPENDENCE (Tanh):
    Lambda = adjusted_tanh(total_fibre_cell_loc, shift, scale)
    zero_CG_strength = Lambda[zero_indices]
    M_rand = Lambda[:,np.newaxis,np.newaxis]*Omega_hat + (1-Lambda[:,np.newaxis,np.newaxis])*np.eye(2, dtype=float)

    # normalise 'M_rand':
    M_rand_hat = np.zeros((N, 2, 2), dtype = float) 
    l_M = norm(np.squeeze(M_rand@F_rm[:,:,np.newaxis], axis=2), axis=1)
    lMnonzero_mask = np.abs(l_M) > 10**(-15)
    M_rand_hat[lMnonzero_mask] = M_rand[lMnonzero_mask] * ((norm(F_rm,axis=1)[lMnonzero_mask]/l_M[lMnonzero_mask])[:,np.newaxis,np.newaxis])
    # for 'zero_indices': set M_rand_hat = zero matrix as we are going to tackle this scenario in 'migrate_t'
    M_rand_hat[zero_indices] = np.zeros(2, dtype=float)

    return M_rand_hat, zero_indices, zero_CG_strength


def CGcc_Pf(cell_coords, N, r_max, y_len, x_len, y_periodic_bool, x_periodic_bool, fibre_coords, Omega, grid_x, grid_y):
    '''
    This function calculates the relative fibre distributions P_f for all the cells 
    INPUT:
    'cell_coords': a 2D np array of size N*2 storing N cell coordinates
    'N': cell population at time t
    'y_len': height of the rectangular domain
    'x_len': width of the rectangular domain
    'y_periodic_bool': a boolean variable specifying whether we have periodic boundary condition in y direction 
    'x_periodic_bool': a boolean variable specifying whether we have periodic boundary condition in x direction 
    'Omega': a 4D np array of size (num_rows, num_cols, 2, 2) storing fibre tensor at each grid point
    'grid_x': a 1D np array [x_min, x_min+Delta_x, x_min+2*Delta_x, ..., x_max] of size 'num_col'
    'grid_y': a 1D np array [y_min, y_min+Delta_y, y_min+2*Delta_y, ..., y_max] of size 'num_row'
    'fibre_coords': a 2D np array of size (num_col*num_row, 2) storing coordinates of all the fibre grid points
    OUTPUT:
    'cf_dist': a 2D np array of size N*(num_col*num_row), where row i stores cell i's distances to all the fibre grid points 
    'Omega_reshape': a 3D np array of size (num_col*num_row, 2, 2) storing the fibre tensorial field information on the grid points defined
    'total_fibre_density': a 1D np array of length num_col*num_row storing lambda_1+lambda_2 values on the grid points defined
    'P_f': a 1D np array of length N storing the relative fibre distributions P_f for all N cells  

    DEBUGGED
    '''

    # for each cell, find the pairwise distances to all the fibre grid points:
    dx = cell_coords[:, np.newaxis][:, :, 0]-fibre_coords[:, 0]
    dy = cell_coords[:, np.newaxis][:, :, 1]-fibre_coords[:, 1]

    # y-periodicity if specified:
    if y_periodic_bool: 
        dy_abs = abs(dy)
        dy_H = y_len - dy_abs
        mask1 = (dy_abs >= dy_H) & (fibre_coords[:, 1] < y_len/2)
        dy[mask1] = - dy_H[mask1]
        mask2 = (dy_abs >= dy_H) & (fibre_coords[:, 1] >= y_len/2)
        dy[mask2] = dy_H[mask2]

    # x-periodicity if specified:
    if x_periodic_bool: 
        dx_abs = abs(dx)
        dx_H = x_len - dx_abs
        mask1 = (dx_abs >= dx_H) & (fibre_coords[:, 0] < x_len/2)
        dx[mask1] = - dx_H[mask1]
        mask2 = (dx_abs >= dx_H) & (fibre_coords[:, 0] >= x_len/2)
        dx[mask2] = dx_H[mask2]

    cf_dist = np.sqrt(dx**2 + dy**2) # row i stores cell i's distances to all the grid points 
    #contrib_gridpts_mask = cf_dist <= r_max # all the grid points lying within 'r_max' are included when calculating the fibre CG contribution to inter-cellular interactions

    # caclulate the total fibre space-filling percentage (i.e. lambda_1+lambda_1) on the grid points:
    Omega_reshape = np.reshape(Omega, (len(grid_x)*len(grid_y), 2, 2))
    total_fibre_density = np.trace(Omega_reshape, axis1=1, axis2=2)
    #contrib_fibre_density = [total_fibre_density[row] for row in contrib_gridpts_mask] # fibre densities contributing to P_f

    # carrying capacity of fibres in a cell's neighbourhood:
    #carrying_f = np.pi * (r_max**2)

    # calculate the P_f for all the cells:
    #P_f = np.zeros(N, dtype=float)
    #for i in range(N): 
    #    # we approximate the integration in the numerator by area*(mean of values defined in the region of integration)
    #    P_f[i] = (np.mean(contrib_fibre_density[i])*np.pi*(r_max**2)) / carrying_f 
   
    return cf_dist, Omega_reshape, total_fibre_density #, P_f


def CGcc_Pc(F_adh_max, total_adh_magnitude, rep_dist_):
    '''
    This function calculates the relative inter-cellular force-weighted cell distribution.
    INPUT:
    'F_adh_max': a parameter representing the maximum adhesive (shall be the most negative) forces based on the Lennard Jone's potential
    'total_adh_magnitude': a 1D np array of length N storing the magnitudes of total adhesion force felt by cells
    'rep_dist_': a 2D np array of size N*N storing the distance to 'rep_adh_len' for repulsive neighbours
    OUTPUT:
    'P_c': a 1D np array of length N storing the relative inter-cellular force-weighted cell distribution P_c for all N cells  

    DEBUGGED
    '''

    # calculate the repulsion-weighted and the adhesion-weighted cell distributions:
    rep_distri = (1/6) * np.nansum(rep_dist_, axis=1)
    adh_distri = (1/30) * total_adh_magnitude / F_adh_max

    P_c = rep_distri + adh_distri

    return P_c


def CG_cc(beta, F_cc, P_c, P_f, N, Omega_cell_loc):
    '''
    This function calculates the fibre contact guidance matrix associated with inter-cellulat interactions for each cell.
    INPUT:
    'beta': Hill's coefficient controlling the switch-like behaviour of contact guidance
    'F_cc': a 2D np array of size N*2 storing the inter-cellular forces experienced by each cell
    'P_c': a 1D np array of length N storing the relative inter-cellular force-weighted cell distribution P_c for all N cells 
    'P_f': a 1D np array of length N storing the relative fibre distributions P_f for all N cells 
    'N': cell population at time t 
    'Omega_cell_loc': a 3D np array of size (N, 2, 2) storing fibre tensor at each cell location
    OUTPUT:
    'M_cc_hat': a 3D np array of size (N, 2, 2) storing the normalised contact guidance matrix at cell locations

    DEBUGGED
    '''

    # find the length-preserving Omega at cell locations:
    Omega_hat = np.zeros((N, 2, 2), dtype = float) 
    l_rand = norm(np.squeeze(Omega_cell_loc@F_cc[:,:,np.newaxis], axis=2), axis=1) # matrix vector multiplication length
    lnonzero_mask = l_rand > 0.0 # we only care about non-zero motility vector after contact guidance
    Omega_hat[lnonzero_mask] = Omega_cell_loc[lnonzero_mask] * ((norm(F_cc,axis=1)[lnonzero_mask]/l_rand[lnonzero_mask])[:,np.newaxis,np.newaxis]) # normalise

    # the strength of contact guidance on a cell's inter-cellular motion is weighted by the balance between the fibre distribuiton and the cell distribuiton in its vicinity
    M_cc = (P_f**beta/(P_c**beta+P_f**beta))[:,np.newaxis,np.newaxis] * Omega_hat + \
        (P_c**beta/(P_c**beta+P_f**beta))[:,np.newaxis,np.newaxis] * np.full((N,2,2), np.array([[1.0,0.0],[0.0,1.0]]))
    
    # normalise 'M_rand':
    M_cc_hat = np.zeros((N, 2, 2), dtype = float) 
    l_M = norm(np.squeeze(M_cc@F_cc[:,:,np.newaxis], axis=2), axis=1)
    lMnonzero_mask = l_M > 0.0
    M_cc_hat[lMnonzero_mask] = M_cc[lMnonzero_mask] * ((norm(F_cc,axis=1)[lMnonzero_mask]/l_M[lMnonzero_mask])[:,np.newaxis,np.newaxis])

    return M_cc_hat


def cellpool_confluency(pool_top, cell_coords, cell_pool_width, sigma, hstripe_N, y_max, x_min, x_max, y_min):
    # hstripe_N: number cells in conflency in a horizontal stripe of length x_max-x_min and width cell_pool_width

    num_hstripe = int(cell_pool_width/sigma)

    added_cellcoords = []
    remove_cellindices = []
    for i in range(num_hstripe): # loop through each horizontal stripe
        # for top pool:
        cells_top_i = np.logical_and(cell_coords[:,1]>=pool_top+i*sigma, cell_coords[:,1]<pool_top+(i+1)*sigma)
        cells_top_i_index = np.where(cells_top_i == True)[0]
        cellnum_top_i = np.sum(cells_top_i)
        if cellnum_top_i < hstripe_N: # need to add more cells for confluency state
            addnum_top_i = int(hstripe_N - cellnum_top_i)
            add_xcoord_top_i = np.random.uniform(x_min, x_max, addnum_top_i)
            add_ycoord_top_i = np.random.uniform(pool_top+i*sigma, pool_top+(i+1)*sigma, addnum_top_i)
            added_cellcoords.extend(list(np.column_stack((add_xcoord_top_i, add_ycoord_top_i))))
        if cellnum_top_i > hstripe_N: # need to remove cells!
            removenum_top_i = int(cellnum_top_i - hstripe_N)
            remove_indice_top_i = np.random.choice(cells_top_i_index, removenum_top_i, replace=False)
            remove_cellindices.extend(list(remove_indice_top_i))

        # for bottom pool:
        cells_bottom_i = np.logical_and(cell_coords[:,1]>=y_min+i*sigma, cell_coords[:,1]<y_min+(i+1)*sigma)
        cells_bottom_i_index = np.where(cells_bottom_i == True)[0]
        cellnum_bottom_i = np.sum(cells_bottom_i)
        if cellnum_bottom_i < hstripe_N: # need to add more cells for confluency state
            addnum_bottom_i = int(hstripe_N - cellnum_bottom_i)
            add_xcoord_bottom_i = np.random.uniform(x_min, x_max, addnum_bottom_i)
            add_ycoord_bottom_i = np.random.uniform(y_min+i*sigma, y_min+(i+1)*sigma, addnum_bottom_i)
            added_cellcoords.extend(list(np.column_stack((add_xcoord_bottom_i, add_ycoord_bottom_i))))
        if cellnum_bottom_i > hstripe_N: # need to remove cells!
            removenum_bottom_i = int(cellnum_bottom_i - hstripe_N)
            remove_indice_bottom_i = np.random.choice(cells_bottom_i_index, removenum_bottom_i, replace=False)
            remove_cellindices.extend(list(remove_indice_bottom_i))

    added_cellcoords = np.array(added_cellcoords)

    return added_cellcoords, remove_cellindices


def cell_proliferation(Delta_0, rho_0, xi, prolif_bool, cell_coords, num_rep_neigh, num_adh_neigh, N, y_periodic_bool, x_periodic_bool, \
                       y_len, x_len, x_min, x_max, y_min, y_max, delta_t): 
    '''
    This function allows a simple density-dependent proliferation.
    INPUT:
    'Delta_0': a parameter denoting the division time when cell i is by itself
    'rho_0': an integer representing the carrying number of cells in a cell's neighbourhood area
    'xi': a parameter denoting the distance a child cell that will be placed from its mother cell
    'prolif_bool': a boolean variable determining whether to allow proliferation in our model
    'cell_coords': a 2D np array of size N*2 storing N cell coordinates
    'num_rep_neigh': a 1D np array of size N storing the number of repulsive neighbours
    'num_adh_neigh': a 1D np array of size N storing the number of adhesive neighbours
    'N': the cell population at time N
    'y_periodic_bool': a boolean variable specifying whether we have periodic boundary condition in y direction 
    'x_periodic_bool': a boolean variable specifying whether we have periodic boundary condition in x direction 
    'y_len': height of the rectangular domain, = 'y_max' - 'y_min'
    'x_len': width of the rectangular domain, ='x_max' - 'x_min'
    OUTPUT:
    'cell_coords': an updated 2D np array of size >=N*2 for >= N cells

    DEBUGGED
    '''
    
    num_rep_neigh = num_rep_neigh.astype(float) # make sure that the data type is float as required.

    if prolif_bool: # if proliferation is allowed
        # calculate the mean division times for all the cells:
        #divi_T =  Delta_0 * (1 + ((num_rep_neigh+num_adh_neigh)/rho_0)**4) 
        #divi_probs = delta_t / divi_T # this uses linear approximation for small delta_t
        # FOR ACCURATE PROB:
        #divi_probs = 1 - np.exp(-delta_t / (Delta_0 * (1.0 + num_rep_neigh/rho_0)))
        # FOR Simpler PROB:
        divi_probs = (delta_t/Delta_0) * (1-num_rep_neigh/rho_0)
        # FOR SIMPLEST EXP GROWTH MODEL: 
        #divi_probs = growth_rate * delta_t

        # sample random numbers from U[0, 1] to determine whether the cell is able to prolifer  ate:
        u = np.random.rand(N)
        prolif_mask = np.logical_and(u<=divi_probs, num_rep_neigh<rho_0) # avoid over-crowdedness
        mother_coord = cell_coords[prolif_mask]

        # place child cells at a distance 'xi' to the mother cells at random angles
        child_angles = np.random.uniform(low=0.0, high=2*np.pi, size=np.sum(prolif_mask))
        child_dx, child_dy = np.cos(child_angles), np.sin(child_angles)
        child_coords = mother_coord + xi*np.column_stack((child_dx, child_dy))
        
        # update child_coords if periodic boundary conditions:
        # (note we do not include the children cells if falling outside the domain of interest given no periodic boundary conditions)
        # in x direction:
        outside_x = np.logical_or(child_coords[:, 0]<x_min, child_coords[:, 0]>x_max)
        if x_periodic_bool:
            child_coords[outside_x, 0] = child_coords[outside_x, 0] % x_len
        else:
            child_coords = child_coords[~outside_x]
        # in y direction:
        outside_y = np.logical_or(child_coords[:, 1]<y_min, child_coords[:, 1]>y_max)
        if y_periodic_bool:
            child_coords[outside_y, 1] = child_coords[outside_y, 1] % y_len
        else:
            child_coords = child_coords[~outside_y]
    else:
        mother_coord = np.empty(0)

    return child_coords, mother_coord


def cf_weight(omega_0, sigma, cf_dist_i):
    '''
    This function calculates the cell-fibre weight function 'omega_cf' to encapsulate the local non-uniform feedback from cells to the surrounding fibres.
    INPUT:
    'omega_0': a parameter determining the maximum cell-fibre effect
    'sigma': the constant cell diameter uniform across the cell population
    'cf_dist_i': a 1D np array of length m storing the pairwise distance between cell i and the grid points lying within its fibre-cell impact radius
    OUTPUT:
    'weight_i': a 1D np array of length m storing the weight cell i imposed on all the grid points within its fibre-cell impact radius

    DEBUGGED
    '''

    weight_i = omega_0 * (1 - cf_dist_i / (sigma/2))

    # no contributions on fibres if the fibre grid points are outside cell i's cell-fibre impact radius:
    outside_mask = cf_dist_i > sigma/2
    weight_i[outside_mask] = 0.0

    return weight_i


def fibre_degradation(Omega_reshape, cf_dist, sigma, d, omega_0, N, grid_x, grid_y, delta_t):
    '''
    This function updates the fibre field Omega due to cell degradations.
    INPUT:
    'Omega_reshape': a 3D np array of size (num_col*num_row, 2, 2) storing the fibre tensorial field information on the grid points defined
    'cf_dist': a 2D np array of size N*(num_col*num_row), where row i stores cell i's distances to all the fibre grid points 
    'sigma': the constant cell diameter uniform across the cell population
    'd': the parameter denoting the constant degradation rate
    'omega_0': a parameter determining the maximum cell-fibre effect
    'N': cell population at time t
    'grid_x': a 1D np array [x_min, x_min+Delta_x, x_min+2*Delta_x, ..., x_max] of size 'num_col'
    'grid_y': a 1D np array [y_min, y_min+Delta_y, y_min+2*Delta_y, ..., y_max] of size 'num_row'
    'delta_t': a fixed time stepping size
    OUPTUT:
    'Omega': an updated (due to fibre degradation by cells) 4D np array of size (num_rows, num_cols, 2, 2) storing fibre tensor at each grid point

    DEBUGGED
    '''

    # find the fibre grid points lying in a cell's impact area on fibres:
    contrib_gridpts_mask = cf_dist <= sigma/2
    contrib_fibre_dist = [row[mask_row] for row, mask_row in zip(cf_dist, contrib_gridpts_mask)] # row i: contributing fibre grid points for cell i

    # loop through each cell and degrate fibres:
    for i in range(N):
        weight_i = cf_weight(omega_0, sigma, contrib_fibre_dist[i])
        deg = weight_i[:,np.newaxis,np.newaxis] * d * Omega_reshape[contrib_gridpts_mask[i]] * delta_t
        Omega_reshape[contrib_gridpts_mask[i]] = Omega_reshape[contrib_gridpts_mask[i]] - deg
    
    # shape back: 
    Omega = np.reshape(Omega_reshape, (len(grid_y), len(grid_x), 2, 2))

    return Omega, Omega_reshape


def fibre_secretion(Vel_cells, total_fibre_density, Omega_reshape, cf_dist, sigma, s, omega_0, N, grid_x, grid_y, delta_t, tau):
    '''
    This function updates the fibre field Omega due to cell secretions.
    INPUT:
    'Vel_cells': a list of N lists, where the ith list correspondonds to the ith cell and it stores cell i's current and all the previous velocities
    'total_fibre_density': a 1D np array of length num_col*num_row storing lambda_1+lambda_2 values on the grid points defined
    'Omega_reshape': a 3D np array of size (num_col*num_row, 2, 2) storing the fibre tensorial field information on the grid points defined
    'cf_dist': a 2D np array of size N*(num_col*num_row), where row i stores cell i's distances to all the fibre grid points 
    'sigma': the constant cell diameter uniform across the cell population
    's': the parameter denoting the constant secretion rate
    'omega_0': a parameter determining the maximum cell-fibre effect
    'N': cell population at time t
    'grid_x': a 1D np array [x_min, x_min+Delta_x, x_min+2*Delta_x, ..., x_max] of size 'num_col'
    'grid_y': a 1D np array [y_min, y_min+Delta_y, y_min+2*Delta_y, ..., y_max] of size 'num_row'
    'delta_t': a fixed time stepping size
    'tau': the memory time length when calculating a cell's average migratory direction (this is the direction of the newly laid down fibres)
    OUPTUT:
    'Omega': an updated (due to fibre secretion by cells) 4D np array of size (num_rows, num_cols, 2, 2) storing fibre tensor at each grid point
    
    DEBUGGED
    '''

    # find the fibre grid points lying in a cell's impact area on fibres:
    contrib_gridpts_mask = cf_dist <= sigma/2
    contrib_fibre_dist = [row[mask_row] for row, mask_row in zip(cf_dist, contrib_gridpts_mask)] # row i: contributing fibre grid points for cell i
    contrib_fibre_density = [total_fibre_density[row] for row in contrib_gridpts_mask] # fibre densities contributing to cell secretion

    # loop through each cell and degrate fibres:
    for i in range(N):
        # find the average velocity (including the current step) in the past tau times, thus 'omega_sec':
        m = int(tau / delta_t)
        prev_vel_i = Vel_cells[i]
        if len(prev_vel_i) < m: # if the existing time for cell i has not reached tau: average over the cell's current age (which is smaller than m)
            vel_ave_i = np.mean(np.vstack(prev_vel_i), axis=0)
        else:
            vel_ave_i = np.mean(np.vstack(prev_vel_i[-m:]), axis=0)
        vel_ave_l = norm(vel_ave_i)
        if vel_ave_l > 0:  # make this unit length
            unit_vel_ave_i = vel_ave_i / vel_ave_l
            omega_sec = np.outer(unit_vel_ave_i, unit_vel_ave_i)
        else: # if the average velocity in the past tau times is zero, then produce isotropic fibres
            omega_sec = np.array([[1.0, 0.0], [0.0, 1.0]])

        weight_i = cf_weight(omega_0, sigma, contrib_fibre_dist[i])
        sec = (weight_i*(1-contrib_fibre_density[i]))[:,np.newaxis,np.newaxis] * s * omega_sec * delta_t
        Omega_reshape[contrib_gridpts_mask[i]] = Omega_reshape[contrib_gridpts_mask[i]] + sec
    
    # shape back: 
    Omega = np.reshape(Omega_reshape, (len(grid_y), len(grid_x), 2, 2))

    return Omega, Omega_reshape


def migrate_t(cell_coords, Vel_cells, M_cc_hat, M_rand_hat, F_rm, F_cc, eta, delta_t, y_periodic_bool, x_periodic_bool, x_min, x_max, x_len, y_min, y_max, y_len, N, 
              zero_indices_cc, zero_CG_strengt_cc, zero_indices_rm, zero_CG_strength_rm, Omega_cell_loc): 
    '''
    This function migrates based on force, CG, and the boundary information. However, we are not removing cells if they migrate outside the domain in this function (will be done later). 
    INPUT:
    'cell_coords': a 2D np array of size N*2 storing N cell coordinates
    'Vel_cells': a list of N lists, where the ith list correspondonds to the ith cell and it stores cell i's current and all the previous velocities
    'M_cc_hat': a 3D np array of size (N, 2, 2) storing the normalised contact guidance matrix at cell locations
    'M_rand_hat': a 3D np array of size (N, 2, 2) storing the normalised contact guidance matrix at cell locations
    'F_rm': a 2D np array storing random forces experienced by each cell
    'F_cc': a 2D np array of size N*2 storing the inter-cellular forces experienced by each cell
    'eta': cell's effective friction coefficient
    'delta_t': a fixed time stepping size
    'y_periodic_bool': a boolean variable specifying whether we have periodic boundary condition in y direction 
    'x_periodic_bool': a boolean variable specifying whether we have periodic boundary condition in x direction
    'y_len': height of the rectangular domain, = 'y_max' - 'y_min'
    'x_len': width of the rectangular domain, ='x_max' - 'x_min'
    'N': cell population at time t
    OUTPUT:
    'cell_coords': a 2D np array of size N*2 storing N cell coordinates after migration
    'Vel_cells': a list of N lists, where the ith list correspondonds to the ith cell and we append cell i's current velocity to it
    'vel_t': a 2D np array of N*2 storing instantaneous velocities of N cells 

    DEBUGGED
    '''

    # calculates the velocities then migrate;
    vel_t = (np.squeeze(M_rand_hat@F_rm[:,:,np.newaxis], axis=2) + np.squeeze(M_cc_hat@F_cc[:,:,np.newaxis],axis=2)) / eta

    # What if F_cc is in the direction of the v2 but lambda2=0? randomly choose from v1 and -v1 as a result of CG. 
    if len(zero_indices_cc) > 0:
        for i in range(len(zero_indices_cc)):
            # find v1 (thus -v1):
            omega_i = Omega_cell_loc[zero_indices_cc[i]]
            _, eigenvectors = np.linalg.eig(omega_i)
            v1 = eigenvectors[:, np.argmax(np.abs(eigenvectors[0]))] 
            v1 = v1 / np.linalg.norm(v1)
            v1_choices = np.array([v1, -v1])
            random_selectioni = v1_choices[np.random.choice([0, 1])]
            vel_i = (zero_CG_strengt_cc[i] * random_selectioni + \
                (1-zero_CG_strengt_cc[i]) * F_cc[i]) / eta
            vel_i = (vel_i/np.linalg.norm(vel_i)) * np.linalg.norm(F_cc[zero_indices_cc[i]]) # keep the magnitude of F_cc
            vel_t[zero_indices_cc[i]] += vel_i
    # similar for F_rm:
    if len(zero_indices_rm) > 0:
        for j in range(len(zero_indices_rm)):
            # find v1 (thus -v1):
            omega_j = Omega_cell_loc[zero_indices_rm[j]]
            _, eigenvectors = np.linalg.eig(omega_j)
            v1 = eigenvectors[:, np.argmax(np.abs(eigenvectors[0]))] 
            v1 = v1 / np.linalg.norm(v1)
            v1_choices = np.array([v1, -v1])
            random_selectionj = v1_choices[np.random.choice([0, 1])]
            vel_j = (zero_CG_strength_rm[j] * random_selectionj + \
                (1-zero_CG_strength_rm[j]) * F_rm[j]) / eta
            vel_j = (vel_j/np.linalg.norm(vel_j)) * np.linalg.norm(F_rm[zero_indices_rm[j]]) # keep the magnitude of F_cc
            vel_t[zero_indices_rm[j]] += vel_j

    # migrate cells based on velocities calculated:
    cell_coords = cell_coords + vel_t * delta_t

    # if periodic boundary conditions, update those cells falling outside the domain; otherwise, we don't remove those cells 
    # at this stage due to their later contributions to fibre dynamics. But after fibre dynamics, we will remove those cells.
    outside_x = np.logical_or(cell_coords[:, 0]<x_min, cell_coords[:, 0]>x_max) # in x direction
    if x_periodic_bool:
        cell_coords[outside_x, 0] = cell_coords[outside_x, 0] % x_len
    outside_y = np.logical_or(cell_coords[:, 1]<y_min, cell_coords[:, 1]>y_max) # in y-direction
    if y_periodic_bool:
        cell_coords[outside_y, 1] = cell_coords[outside_y, 1] % y_len

    # record velocity information for all cells in the domain into 'Vel_cells' for later fibre secretion purpose:
    for cell_i in range(N):
        Vel_cells[cell_i].append(vel_t[cell_i])

    return cell_coords, Vel_cells, vel_t


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Below subfunctions describe different initialisations of the cell and fibre distribution at t = 0.

def single_cell_atmiddle_init(x_min, x_len, y_min, y_len):
    # This function distributes a single cell in the middle of our rectangular domain.
    # TESTED

    cell_coords = np.array([[x_min+x_len/2, y_min+y_len/2]])

    return cell_coords


def pair_cell_atmiddle_init(x_min, x_len, y_min, y_len, d, theta):
    # This function distributes a pair of cells in the middle of our rectangular domain.
    # These two cells are 'd' distance away, and they form an angle 'theta' to the positive x axis.
    # TESTED

    middle_pt = np.array([x_min+x_len/2, y_min+y_len/2])
    cell1_coord = middle_pt + (d/2) * np.array([np.cos(theta), np.sin(theta)])
    cell2_coord = middle_pt - (d/2) * np.array([np.cos(theta), np.sin(theta)])
    cell_coords = np.array([cell1_coord, cell2_coord])

    return cell_coords


def cell_invasion_middle_init(x_min, x_len, y_min, y_len, d, sigma):
    # This function distributes cells of diameter 'sigma' in confluency in the middle area of the rectangular domain of size 'x_len'*'d'
    # TESTED

    # roughly the number of rows and columns of cells:
    row_cells = int(d/sigma)
    col_cells = int(x_len/sigma)
    N = row_cells * col_cells

    # sample x coordinates of cells:
    coord_x = np.random.uniform(x_min, x_min+x_len, size=N)

    # sample x coordinates of cells in each col:
    coord_y = np.array([])
    cell_y_min = y_min + y_len/2 - d/2
    for i in range(row_cells): # for the ith col
        coord_y_i = np.random.uniform(cell_y_min, cell_y_min+sigma, size=col_cells)
        coord_y = np.concatenate((coord_y, coord_y_i))
        cell_y_min = cell_y_min + sigma
        
    cell_coords = np.column_stack((coord_x, coord_y))
    return cell_coords


def nofibre_init(x_min, x_max, y_min, y_max, gridsize_f, y_periodic_bool, x_periodic_bool):
    # This function defines zero fibres on the fibre grid of grid length 'gridsize_f' in the rectangular domain.

    # take whether periodic BC into account:
    if y_periodic_bool:
        grid_y = np.arange(y_min, y_max, gridsize_f) 
    else:
        grid_y = np.arange(y_min, y_max+gridsize_f, gridsize_f) 
    if x_periodic_bool:
        grid_x = np.arange(x_min, x_max, gridsize_f) 
    else:
        grid_x = np.arange(x_min, x_max+gridsize_f, gridsize_f) 

    # obtain an array that contains coordinates of the Omega grid points (order: bottom to top; left to right)
    X, Y = np.meshgrid(grid_x, grid_y)
    num_rows, num_cols = X.shape
    xx_1d, yy_1d = X.ravel(), Y.ravel()
    fibre_coords = np.column_stack((xx_1d, yy_1d))

    Omega = np.zeros((num_rows, num_cols, 2, 2), dtype=float)

    return Omega, fibre_coords, grid_x, grid_y


def vAligned_fibre_constDensity_init(lambda_1_const, lambda_2_const, x_min, x_max, y_min, y_max, gridsize_f, y_periodic_bool, x_periodic_bool):
    # This function initialises the fibre field on the domain grid of resolution 'gridsize_f'. 
    # The fibre field has a consant vertical v1 direction and constant lambda_1 and lambda_2 across the domain.

    # take whether periodic BC into account:
    if y_periodic_bool:
        grid_y = np.arange(y_min, y_max, gridsize_f) 
    else:
        grid_y = np.arange(y_min, y_max+gridsize_f, gridsize_f) 
    if x_periodic_bool:
        grid_x = np.arange(x_min, x_max, gridsize_f) 
    else:
        grid_x = np.arange(x_min, x_max+gridsize_f, gridsize_f) 

    # obtain an array that contains coordinates of the Omega grid points (order: bottom to top; left to right)
    X, Y = np.meshgrid(grid_x, grid_y)
    num_rows, num_cols = X.shape
    xx_1d, yy_1d = X.ravel(), Y.ravel()
    fibre_coords = np.column_stack((xx_1d, yy_1d))

    # contant vertical v1 direction across the domain:
    v1, v2 = np.array([0.0, 1.0]), np.array([1.0, 0.0])
    omega_gridpt = lambda_1_const*np.outer(v1, v1) + lambda_2_const*np.outer(v2, v2)
    Omega = np.full((num_rows, num_cols, 2, 2), omega_gridpt)

    return Omega, fibre_coords, grid_x, grid_y


def vAlignedNoise_fibre_constDensity_init(Delta_theta, lambda_1_const, lambda_2_const, x_min, x_max, y_min, y_max, gridsize_f, y_periodic_bool, x_periodic_bool):
    # This function initialises the fibre field on the domain grid of resolution 'gridsize_f'. 
    # The fibre field has v1 direction in the range (-Delta_theta+Pi/2, Delta_theta+Pi/2) and constant lambda_1 and lambda_2 across the domain.

    # take whether periodic BC into account:
    if y_periodic_bool:
        grid_y = np.arange(y_min, y_max, gridsize_f) 
    else:
        grid_y = np.arange(y_min, y_max+gridsize_f, gridsize_f) 
    if x_periodic_bool:
        grid_x = np.arange(x_min, x_max, gridsize_f) 
    else:
        grid_x = np.arange(x_min, x_max+gridsize_f, gridsize_f) 

    # obtain an array that contains coordinates of the Omega grid points (order: bottom to top; left to right)
    X, Y = np.meshgrid(grid_x, grid_y)
    num_rows, num_cols = X.shape
    xx_1d, yy_1d = X.ravel(), Y.ravel()
    fibre_coords = np.column_stack((xx_1d, yy_1d))
    N_f = len(fibre_coords)

    # sample random directions for v1 thus determine v2:
    v1_theta = np.random.uniform(np.pi/2-Delta_theta, np.pi/2+Delta_theta, size=N_f)
    v2_theta = v1_theta + np.pi/2
    largerPi_mask = v2_theta > np.pi
    v2_theta[largerPi_mask] = v2_theta[largerPi_mask] - 2*np.pi
    v1_x, v1_y, v2_x, v2_y = np.cos(v1_theta), np.sin(v1_theta), np.cos(v2_theta), np.sin(v2_theta)
    v1, v2 = np.column_stack((v1_x, v1_y)), np.column_stack((v2_x, v2_y))

    Omega = np.zeros((N_f, 2, 2), dtype=float)
    for i in range(N_f):
        omega_i = lambda_1_const*np.outer(v1[i], v1[i]) + lambda_2_const*np.outer(v2[i], v2[i])
        Omega[i] = omega_i
    Omega = np.reshape(Omega, (num_rows, num_cols, 2, 2))

    return Omega, fibre_coords, grid_x, grid_y


def hAlignedNoise_fibre_constDensity_init(Delta_theta, lambda_1_const, lambda_2_const, x_min, x_max, y_min, y_max, gridsize_f, y_periodic_bool, x_periodic_bool):
    # This function initialises the fibre field on the domain grid of resolution 'gridsize_f'. 
    # The fibre field has v1 direction in the range (-Delta_theta, Delta_theta) and constant lambda_1 and lambda_2 across the domain.

    # take whether periodic BC into account:
    if y_periodic_bool:
        grid_y = np.arange(y_min, y_max, gridsize_f) 
    else:
        grid_y = np.arange(y_min, y_max+gridsize_f, gridsize_f) 
    if x_periodic_bool:
        grid_x = np.arange(x_min, x_max, gridsize_f) 
    else:
        grid_x = np.arange(x_min, x_max+gridsize_f, gridsize_f) 

    # obtain an array that contains coordinates of the Omega grid points (order: bottom to top; left to right)
    X, Y = np.meshgrid(grid_x, grid_y)
    num_rows, num_cols = X.shape
    xx_1d, yy_1d = X.ravel(), Y.ravel()
    fibre_coords = np.column_stack((xx_1d, yy_1d))
    N_f = len(fibre_coords)

    # sample random directions for v1 thus determine v2:
    v1_theta = np.random.uniform(-Delta_theta, Delta_theta, size=N_f)
    v2_theta = v1_theta + np.pi/2
    largerPi_mask = v2_theta > np.pi
    v2_theta[largerPi_mask] = v2_theta[largerPi_mask] - 2*np.pi
    v1_x, v1_y, v2_x, v2_y = np.cos(v1_theta), np.sin(v1_theta), np.cos(v2_theta), np.sin(v2_theta)
    v1, v2 = np.column_stack((v1_x, v1_y)), np.column_stack((v2_x, v2_y))

    Omega = np.zeros((N_f, 2, 2), dtype=float)
    for i in range(N_f):
        omega_i = lambda_1_const*np.outer(v1[i], v1[i]) + lambda_2_const*np.outer(v2[i], v2[i])
        Omega[i] = omega_i
    Omega = np.reshape(Omega, (num_rows, num_cols, 2, 2))

    return Omega, fibre_coords, grid_x, grid_y


def hAligned_fibre_constDensity_init(lambda_1_const, lambda_2_const, x_min, x_max, y_min, y_max, gridsize_f, y_periodic_bool, x_periodic_bool):
    # This function initialises the fibre field on the domain grid of resolution 'gridsize_f'. 
    # The fibre field has a consant horizontal v1 direction and constant lambda_1 and lambda_2 across the domain.

    # take whether periodic BC into account:
    if y_periodic_bool:
        grid_y = np.arange(y_min, y_max, gridsize_f) 
    else:
        grid_y = np.arange(y_min, y_max+gridsize_f, gridsize_f) 
    if x_periodic_bool:
        grid_x = np.arange(x_min, x_max, gridsize_f) 
    else:
        grid_x = np.arange(x_min, x_max+gridsize_f, gridsize_f) 

    # obtain an array that contains coordinates of the Omega grid points (order: bottom to top; left to right)
    X, Y = np.meshgrid(grid_x, grid_y)
    num_rows, num_cols = X.shape
    xx_1d, yy_1d = X.ravel(), Y.ravel()
    fibre_coords = np.column_stack((xx_1d, yy_1d))

    # contant vertical v1 direction across the domain:
    v1, v2 = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    omega_gridpt = lambda_1_const*np.outer(v1, v1) + lambda_2_const*np.outer(v2, v2)
    Omega = np.full((num_rows, num_cols, 2, 2), omega_gridpt)

    return Omega, fibre_coords, grid_x, grid_y


def isotropic_fibre_constDensity_init(lambda_12_const, x_min, x_max, y_min, y_max, gridsize_f, y_periodic_bool, x_periodic_bool):
    # This function initialises the fibre field on the domain grid of resolution 'gridsize_f'. 
    # The fibre field is isotropic (lambda_1 = lambda_2 = 'lambda_12_const') and has random v1 (thus v2) orientations.

    # take whether periodic BC into account:
    if y_periodic_bool:
        grid_y = np.arange(y_min, y_max, gridsize_f) 
    else:
        grid_y = np.arange(y_min, y_max+gridsize_f, gridsize_f) 
    if x_periodic_bool:
        grid_x = np.arange(x_min, x_max, gridsize_f) 
    else:
        grid_x = np.arange(x_min, x_max+gridsize_f, gridsize_f) 

    # obtain an array that contains coordinates of the Omega grid points (order: bottom to top; left to right)
    X, Y = np.meshgrid(grid_x, grid_y)
    num_rows, num_cols = X.shape
    xx_1d, yy_1d = X.ravel(), Y.ravel()
    fibre_coords = np.column_stack((xx_1d, yy_1d))
    N_f = len(fibre_coords)

    # sample random directions for v1 thus determine v2:
    v1_theta = np.random.uniform(0.0, np.pi, size=N_f)
    v2_theta = v1_theta + np.pi/2
    largerPi_mask = v2_theta > np.pi
    v2_theta[largerPi_mask] = v2_theta[largerPi_mask] - 2*np.pi
    v1_x, v1_y, v2_x, v2_y = np.cos(v1_theta), np.sin(v1_theta), np.cos(v2_theta), np.sin(v2_theta)
    v1, v2 = np.column_stack((v1_x, v1_y)), np.column_stack((v2_x, v2_y))

    Omega = np.zeros((N_f, 2, 2), dtype=float)
    for i in range(N_f):
        omega_i = lambda_12_const*np.outer(v1[i], v1[i]) + lambda_12_const*np.outer(v2[i], v2[i])
        Omega[i] = omega_i
    Omega = np.reshape(Omega, (num_rows, num_cols, 2, 2))

    return Omega, fibre_coords, grid_x, grid_y


def nofibreMid_vAlignedBottomTop_init(lambda_1_const, lambda_2_const, d, x_min, x_max, y_min, y_max, y_len, gridsize_f, y_periodic_bool, x_periodic_bool):
    # This function initialises the fibre field on the domain grid with resolution 'gridsize_f'. 
    # There are not fibres in the middle area 'x_len'*'d', while outside this area, v1 is in the direction Pi/2. 
    # TESTED

    # take whether periodic BC into account:
    if y_periodic_bool:
        grid_y = np.arange(y_min, y_max, gridsize_f) 
    else:
        grid_y = np.arange(y_min, y_max+gridsize_f, gridsize_f) 
    if x_periodic_bool:
        grid_x = np.arange(x_min, x_max, gridsize_f) 
    else:
        grid_x = np.arange(x_min, x_max+gridsize_f, gridsize_f) 

    # obtain an array that contains coordinates of the Omega grid points (order: bottom to top; left to right)
    X, Y = np.meshgrid(grid_x, grid_y)
    num_rows, num_cols = X.shape
    xx_1d, yy_1d = X.ravel(), Y.ravel()
    fibre_coords = np.column_stack((xx_1d, yy_1d))

    # find the number of rows that do not have fibres:
    row_fibre_b = np.floor((y_min+y_len/2-d/2)/gridsize_f)
    row_fibre_t = np.floor((y_min+y_len/2+d/2)/gridsize_f)
    no_fibre_num = int((row_fibre_t - row_fibre_b + 1) * num_cols)
    Omega_nofibre = np.full((no_fibre_num, 2, 2), np.array([[0.0, 0.0], [0.0, 0.0]]))

    # sample the direction of v1 (thus v2) for the bottom grid points with fibres
    fibre_num_b = int(row_fibre_b * num_cols)
    v1, v2 = np.tile([0.0, 1.0], (int(fibre_num_b), 1)), np.tile([1.0, 0.0], (int(fibre_num_b), 1))
    # init Omega for those grid points with fibres 
    Omega_b = np.zeros((fibre_num_b, 2, 2), dtype=float)
    for i in range(fibre_num_b):
        omega_i = lambda_1_const*np.outer(v1[i], v1[i]) + lambda_2_const*np.outer(v2[i], v2[i])
        Omega_b[i] = omega_i

    # sample the direction of v1 (thus v2) for the top grid points with fibres
    fibre_num_t = int((len(grid_y)-1-row_fibre_t)*num_cols)
    v1, v2 = np.tile([0.0, 1.0], (int(fibre_num_t), 1)), np.tile([1.0, 0.0], (int(fibre_num_t), 1))
    # init Omega for those grid points with fibres 
    Omega_t = np.zeros((fibre_num_t, 2, 2), dtype=float)
    for i in range(fibre_num_t):
        omega_i = lambda_1_const*np.outer(v1[i], v1[i]) + lambda_2_const*np.outer(v2[i], v2[i])
        Omega_t[i] = omega_i

    Omega = np.concatenate((np.concatenate((Omega_b, Omega_nofibre)), Omega_t))
    Omega = np.reshape(Omega, (num_rows, num_cols, 2, 2))

    return Omega, fibre_coords, grid_x, grid_y


def nofibreMid_vAlignedNoiseBottomTop_init(Delta_theta, lambda_1_const, lambda_2_const, d, x_min, x_max, y_min, y_max, y_len, gridsize_f, y_periodic_bool, x_periodic_bool):
    # This function initialises the fibre field on the domain grid with resolution 'gridsize_f'. 
    # There are not fibres in the middle area 'x_len'*'d', while outside this area, v1 is in the direction (Pi/2-Delta_theta, Pi/2+Delta_theta). 
    # TESTED

    # take whether periodic BC into account:
    if y_periodic_bool:
        grid_y = np.arange(y_min, y_max, gridsize_f) 
    else:
        grid_y = np.arange(y_min, y_max+gridsize_f, gridsize_f) 
    if x_periodic_bool:
        grid_x = np.arange(x_min, x_max, gridsize_f) 
    else:
        grid_x = np.arange(x_min, x_max+gridsize_f, gridsize_f) 

    # obtain an array that contains coordinates of the Omega grid points (order: bottom to top; left to right)
    X, Y = np.meshgrid(grid_x, grid_y)
    num_rows, num_cols = X.shape
    xx_1d, yy_1d = X.ravel(), Y.ravel()
    fibre_coords = np.column_stack((xx_1d, yy_1d))

    # find the number of rows that do not have fibres:
    row_fibre_b = np.floor((y_min+y_len/2-d/2)/gridsize_f)
    row_fibre_t = np.floor((y_min+y_len/2+d/2)/gridsize_f)
    no_fibre_num = int((row_fibre_t - row_fibre_b + 1) * num_cols)
    Omega_nofibre = np.full((no_fibre_num, 2, 2), np.array([[0.0, 0.0], [0.0, 0.0]]))

    # sample the direction of v1 (thus v2) for the bottom grid points with fibres
    fibre_num_b = int(row_fibre_b * num_cols)
    v1_theta = np.random.uniform(np.pi/2-Delta_theta, np.pi/2+Delta_theta, size=int(fibre_num_b))
    v2_theta = v1_theta + np.pi/2
    largerPi_mask = v2_theta > np.pi
    v2_theta[largerPi_mask] = v2_theta[largerPi_mask] - 2*np.pi
    v1_x, v1_y, v2_x, v2_y = np.cos(v1_theta), np.sin(v1_theta), np.cos(v2_theta), np.sin(v2_theta)
    v1, v2 = np.column_stack((v1_x, v1_y)), np.column_stack((v2_x, v2_y))
    # init Omega for those grid points with fibres 
    Omega_b = np.zeros((fibre_num_b, 2, 2), dtype=float)
    for i in range(fibre_num_b):
        omega_i = lambda_1_const*np.outer(v1[i], v1[i]) + lambda_2_const*np.outer(v2[i], v2[i])
        Omega_b[i] = omega_i

    # sample the direction of v1 (thus v2) for the top grid points with fibres
    fibre_num_t = int((len(grid_y)-1-row_fibre_t)*num_cols)
    v1_theta = np.random.uniform(np.pi/2-Delta_theta, np.pi/2+Delta_theta, size=int(fibre_num_t))
    v2_theta = v1_theta + np.pi/2
    largerPi_mask = v2_theta > np.pi
    v2_theta[largerPi_mask] = v2_theta[largerPi_mask] - 2*np.pi
    v1_x, v1_y, v2_x, v2_y = np.cos(v1_theta), np.sin(v1_theta), np.cos(v2_theta), np.sin(v2_theta)
    v1, v2 = np.column_stack((v1_x, v1_y)), np.column_stack((v2_x, v2_y))
    # init Omega for those grid points with fibres 
    Omega_t = np.zeros((fibre_num_t, 2, 2), dtype=float)
    for i in range(fibre_num_t):
        omega_i = lambda_1_const*np.outer(v1[i], v1[i]) + lambda_2_const*np.outer(v2[i], v2[i])
        Omega_t[i] = omega_i

    Omega = np.concatenate((np.concatenate((Omega_b, Omega_nofibre)), Omega_t))
    Omega = np.reshape(Omega, (num_rows, num_cols, 2, 2))

    return Omega, fibre_coords, grid_x, grid_y


def nofibreMid_hAlignedNoiseBottomTop_init(Delta_theta, lambda_1_const, lambda_2_const, d, x_min, x_max, y_min, y_max, y_len, gridsize_f, y_periodic_bool, x_periodic_bool):
    # This function initialises the fibre field on the domain grid with resolution 'gridsize_f'. 
    # There are not fibres in the middle area 'x_len'*'d', while outside this area, v1 is in the direction (-Delta_theta, Delta_theta). 
    # TESTED

    # take whether periodic BC into account:
    if y_periodic_bool:
        grid_y = np.arange(y_min, y_max, gridsize_f) 
    else:
        grid_y = np.arange(y_min, y_max+gridsize_f, gridsize_f) 
    if x_periodic_bool:
        grid_x = np.arange(x_min, x_max, gridsize_f) 
    else:
        grid_x = np.arange(x_min, x_max+gridsize_f, gridsize_f) 

    # obtain an array that contains coordinates of the Omega grid points (order: bottom to top; left to right)
    X, Y = np.meshgrid(grid_x, grid_y)
    num_rows, num_cols = X.shape
    xx_1d, yy_1d = X.ravel(), Y.ravel()
    fibre_coords = np.column_stack((xx_1d, yy_1d))

    # find the number of rows that do not have fibres:
    row_fibre_b = np.floor((y_min+y_len/2-d/2)/gridsize_f)
    row_fibre_t = np.floor((y_min+y_len/2+d/2)/gridsize_f)
    no_fibre_num = int((row_fibre_t - row_fibre_b + 1) * num_cols)
    Omega_nofibre = np.full((no_fibre_num, 2, 2), np.array([[0.0, 0.0], [0.0, 0.0]]))

    # sample the direction of v1 (thus v2) for the bottom grid points with fibres
    fibre_num_b = int(row_fibre_b * num_cols)
    v1_theta = np.random.uniform(-Delta_theta, Delta_theta, size=int(fibre_num_b))
    v2_theta = v1_theta + np.pi/2
    largerPi_mask = v2_theta > np.pi
    v2_theta[largerPi_mask] = v2_theta[largerPi_mask] - 2*np.pi
    v1_x, v1_y, v2_x, v2_y = np.cos(v1_theta), np.sin(v1_theta), np.cos(v2_theta), np.sin(v2_theta)
    v1, v2 = np.column_stack((v1_x, v1_y)), np.column_stack((v2_x, v2_y))
    # init Omega for those grid points with fibres 
    Omega_b = np.zeros((fibre_num_b, 2, 2), dtype=float)
    for i in range(fibre_num_b):
        omega_i = lambda_1_const*np.outer(v1[i], v1[i]) + lambda_2_const*np.outer(v2[i], v2[i])
        Omega_b[i] = omega_i

    # sample the direction of v1 (thus v2) for the top grid points with fibres
    fibre_num_t = int((len(grid_y)-1-row_fibre_t)*num_cols)
    v1_theta = np.random.uniform(-Delta_theta, Delta_theta, size=int(fibre_num_t))
    v2_theta = v1_theta + np.pi/2
    largerPi_mask = v2_theta > np.pi
    v2_theta[largerPi_mask] = v2_theta[largerPi_mask] - 2*np.pi
    v1_x, v1_y, v2_x, v2_y = np.cos(v1_theta), np.sin(v1_theta), np.cos(v2_theta), np.sin(v2_theta)
    v1, v2 = np.column_stack((v1_x, v1_y)), np.column_stack((v2_x, v2_y))
    # init Omega for those grid points with fibres 
    Omega_t = np.zeros((fibre_num_t, 2, 2), dtype=float)
    for i in range(fibre_num_t):
        omega_i = lambda_1_const*np.outer(v1[i], v1[i]) + lambda_2_const*np.outer(v2[i], v2[i])
        Omega_t[i] = omega_i

    Omega = np.concatenate((np.concatenate((Omega_b, Omega_nofibre)), Omega_t))
    Omega = np.reshape(Omega, (num_rows, num_cols, 2, 2))

    return Omega, fibre_coords, grid_x, grid_y


def increaseIsotropy_vAlignedNoise_constDensity_BottomTop_init(d, Delta_theta, total_f_density, iso_max, iso_min, x_min, x_max, y_min, y_max, y_len, gridsize_f, y_periodic_bool, x_periodic_bool):
    # This function initialises the fibre field with total fibre density total_f_density (=lambda_1+lambda_2) fixed on the domain grid of resolution 'gridsize_f'. 
    # There are not fibres in the middle area 'x_len'*'d', while outside this area, v1 is in the direction (Pi/2-Delta_theta, Pi/2+Delta_theta). 
    # For places with fibres, the isotropy decrease from 'iso_max' to 'iso_min' when getting close to the top or the bottom of the domain.

    # TESTED
    # take whether periodic BC into account:
    if y_periodic_bool:
        grid_y = np.arange(y_min, y_max, gridsize_f) 
    else:
        grid_y = np.arange(y_min, y_max+gridsize_f, gridsize_f) 
    if x_periodic_bool:
        grid_x = np.arange(x_min, x_max, gridsize_f) 
    else:
        grid_x = np.arange(x_min, x_max+gridsize_f, gridsize_f) 

    # obtain an array that contains coordinates of the Omega grid points (order: bottom to top; left to right)
    X, Y = np.meshgrid(grid_x, grid_y)
    num_rows, num_cols = X.shape
    xx_1d, yy_1d = X.ravel(), Y.ravel()
    fibre_coords = np.column_stack((xx_1d, yy_1d))

    # find the number of rows that do not have fibres:
    row_fibre_b = np.floor((y_min+y_len/2-d/2)/gridsize_f)
    row_fibre_t = np.floor((y_min+y_len/2+d/2)/gridsize_f)
    no_fibre_num = int((row_fibre_t - row_fibre_b + 1) * num_cols)
    Omega_nofibre = np.full((no_fibre_num, 2, 2), np.array([[0.0, 0.0], [0.0, 0.0]]))

    # sample the direction of v1 (thus v2) for the bottom grid points with fibres
    iso_gradient_b = np.linspace(iso_max, iso_min, int(row_fibre_b))
    iso_gradient_b = np.repeat(iso_gradient_b, num_cols)
    fibre_num_b = int(row_fibre_b * num_cols)
    v1_theta = np.random.uniform(np.pi/2-Delta_theta, np.pi/2+Delta_theta, size=int(fibre_num_b))
    v2_theta = v1_theta + np.pi/2
    largerPi_mask = v2_theta > np.pi
    v2_theta[largerPi_mask] = v2_theta[largerPi_mask] - 2*np.pi
    v1_x, v1_y, v2_x, v2_y = np.cos(v1_theta), np.sin(v1_theta), np.cos(v2_theta), np.sin(v2_theta)
    v1, v2 = np.column_stack((v1_x, v1_y)), np.column_stack((v2_x, v2_y))
    # init Omega for those grid points with fibres 
    Omega_b = np.zeros((fibre_num_b, 2, 2), dtype=float)
    for i in range(fibre_num_b):
        iso = iso_gradient_b[i]
        lambda_1_i, lambda_2_i = total_f_density/(1+iso), iso*total_f_density/(1+iso)
        omega_i = lambda_1_i*np.outer(v1[i], v1[i]) + lambda_2_i*np.outer(v2[i], v2[i])
        Omega_b[i] = omega_i

    # sample the direction of v1 (thus v2) for the top grid points with fibres
    iso_gradient_t = np.linspace(iso_min, iso_max, int((len(grid_y)-1-row_fibre_t)))
    iso_gradient_t = np.repeat(iso_gradient_t, num_cols)
    fibre_num_t = int((len(grid_y)-1-row_fibre_t)*num_cols)
    v1_theta = np.random.uniform(np.pi/2-Delta_theta, np.pi/2+Delta_theta, size=int(fibre_num_t))
    v2_theta = v1_theta + np.pi/2
    largerPi_mask = v2_theta > np.pi
    v2_theta[largerPi_mask] = v2_theta[largerPi_mask] - 2*np.pi
    v1_x, v1_y, v2_x, v2_y = np.cos(v1_theta), np.sin(v1_theta), np.cos(v2_theta), np.sin(v2_theta)
    v1, v2 = np.column_stack((v1_x, v1_y)), np.column_stack((v2_x, v2_y))
    # init Omega for those grid points with fibres 
    Omega_t = np.zeros((fibre_num_t, 2, 2), dtype=float)
    for i in range(fibre_num_t):
        iso = iso_gradient_t[i]
        lambda_1_i, lambda_2_i = total_f_density/(1+iso), iso*total_f_density/(1+iso)
        omega_i = lambda_1_i*np.outer(v1[i], v1[i]) + lambda_2_i*np.outer(v2[i], v2[i])
        Omega_t[i] = omega_i
    
    Omega = np.concatenate((np.concatenate((Omega_b, Omega_nofibre)), Omega_t))
    Omega = np.reshape(Omega, (num_rows, num_cols, 2, 2))

    return Omega, fibre_coords, grid_x, grid_y



def nofibreMid_isorandomBottomTop_init(lambda_12_const, d, x_min, x_max, y_min, y_max, y_len, gridsize_f, y_periodic_bool, x_periodic_bool):
    # This function initialises the fibre field on the domain grid with resolution 'gridsize_f'. 
    # There are not fibres in the middle area 'x_len'*'d', while outside this area, fibres are perfectly random and isotropically distributed. 
    # TESTED

    # take whether periodic BC into account:
    if y_periodic_bool:
        grid_y = np.arange(y_min, y_max, gridsize_f) 
    else:
        grid_y = np.arange(y_min, y_max+gridsize_f, gridsize_f) 
    if x_periodic_bool:
        grid_x = np.arange(x_min, x_max, gridsize_f) 
    else:
        grid_x = np.arange(x_min, x_max+gridsize_f, gridsize_f) 

    # obtain an array that contains coordinates of the Omega grid points (order: bottom to top; left to right)
    X, Y = np.meshgrid(grid_x, grid_y)
    num_rows, num_cols = X.shape
    xx_1d, yy_1d = X.ravel(), Y.ravel()
    fibre_coords = np.column_stack((xx_1d, yy_1d))

    # find the number of rows that do not have fibres:
    row_fibre_b = np.floor((y_min+y_len/2-d/2)/gridsize_f)
    row_fibre_t = np.floor((y_min+y_len/2+d/2)/gridsize_f)
    no_fibre_num = int((row_fibre_t - row_fibre_b + 1) * num_cols)
    Omega_nofibre = np.full((no_fibre_num, 2, 2), np.array([[0.0, 0.0], [0.0, 0.0]]))

    # sample the direction of v1 (thus v2) for the bottom grid points with fibres
    fibre_num_b = int(row_fibre_b * num_cols)
    v1_theta = np.random.uniform(0.0, np.pi, size=int(fibre_num_b))
    v2_theta = v1_theta + np.pi/2
    largerPi_mask = v2_theta > np.pi
    v2_theta[largerPi_mask] = v2_theta[largerPi_mask] - 2*np.pi
    v1_x, v1_y, v2_x, v2_y = np.cos(v1_theta), np.sin(v1_theta), np.cos(v2_theta), np.sin(v2_theta)
    v1, v2 = np.column_stack((v1_x, v1_y)), np.column_stack((v2_x, v2_y))
    # init Omega for those grid points with fibres 
    Omega_b = np.zeros((fibre_num_b, 2, 2), dtype=float)
    for i in range(fibre_num_b):
        omega_i = lambda_12_const*np.outer(v1[i], v1[i]) + lambda_12_const*np.outer(v2[i], v2[i])
        Omega_b[i] = omega_i

    # sample the direction of v1 (thus v2) for the top grid points with fibres
    fibre_num_t = int((len(grid_y)-1-row_fibre_t)*num_cols)
    v1_theta = np.random.uniform(0.0, np.pi, size=int(fibre_num_t))
    v2_theta = v1_theta + np.pi/2
    largerPi_mask = v2_theta > np.pi
    v2_theta[largerPi_mask] = v2_theta[largerPi_mask] - 2*np.pi
    v1_x, v1_y, v2_x, v2_y = np.cos(v1_theta), np.sin(v1_theta), np.cos(v2_theta), np.sin(v2_theta)
    v1, v2 = np.column_stack((v1_x, v1_y)), np.column_stack((v2_x, v2_y))
    # init Omega for those grid points with fibres 
    Omega_t = np.zeros((fibre_num_t, 2, 2), dtype=float)
    for i in range(fibre_num_t):
        omega_i = lambda_12_const*np.outer(v1[i], v1[i]) + lambda_12_const*np.outer(v2[i], v2[i])
        Omega_t[i] = omega_i

    Omega = np.concatenate((np.concatenate((Omega_b, Omega_nofibre)), Omega_t))
    Omega = np.reshape(Omega, (num_rows, num_cols, 2, 2))

    return Omega, fibre_coords, grid_x, grid_y



def yesfibreMid_vAlignedNoiseBottomTop_init(Delta_theta, lambda_1_const, lambda_2_const, d, x_min, x_max, y_min, y_max, y_len, gridsize_f, y_periodic_bool, x_periodic_bool):
    # This function initialises the fibre field on the domain grid with resolution 'gridsize_f'. 
    # There are only fibres in the middle area 'x_len'*'d'; v1 is in the direction (Pi/2-Delta_theta, Pi/2+Delta_theta). 
    # TESTED

    # take whether periodic BC into account:
    if y_periodic_bool:
        grid_y = np.arange(y_min, y_max, gridsize_f) 
    else:
        grid_y = np.arange(y_min, y_max+gridsize_f, gridsize_f) 
    if x_periodic_bool:
        grid_x = np.arange(x_min, x_max, gridsize_f) 
    else:
        grid_x = np.arange(x_min, x_max+gridsize_f, gridsize_f) 

    # obtain an array that contains coordinates of the Omega grid points (order: bottom to top; left to right)
    X, Y = np.meshgrid(grid_x, grid_y)
    num_rows, num_cols = X.shape
    xx_1d, yy_1d = X.ravel(), Y.ravel()
    fibre_coords = np.column_stack((xx_1d, yy_1d))

    # find the number of rows that has fibres:
    row_fibre_b = np.floor((y_min+y_len/2-d/2)/gridsize_f)
    row_fibre_t = np.floor((y_min+y_len/2+d/2)/gridsize_f)
    yes_fibre_num = int((row_fibre_t - row_fibre_b + 1) * num_cols)
    Omega_mid = np.zeros((yes_fibre_num, 2, 2), dtype=float)
    v1_theta = np.random.uniform(np.pi/2-Delta_theta, np.pi/2+Delta_theta, size=int(yes_fibre_num))
    v2_theta = v1_theta + np.pi/2
    largerPi_mask = v2_theta > np.pi
    v2_theta[largerPi_mask] = v2_theta[largerPi_mask] - 2*np.pi
    v1_x, v1_y, v2_x, v2_y = np.cos(v1_theta), np.sin(v1_theta), np.cos(v2_theta), np.sin(v2_theta)
    v1, v2 = np.column_stack((v1_x, v1_y)), np.column_stack((v2_x, v2_y))
    for i in range(yes_fibre_num):
        omega_i = lambda_1_const*np.outer(v1[i], v1[i]) + lambda_2_const*np.outer(v2[i], v2[i])
        Omega_mid[i] = omega_i
    
    # bottom: no fibres
    fibre_num_b = int(row_fibre_b * num_cols)
    Omega_b = np.full((fibre_num_b, 2, 2), np.array([[0.0, 0.0], [0.0, 0.0]]))

    # top: no fibres
    fibre_num_t = int((len(grid_y)-1-row_fibre_t)*num_cols)
    Omega_t = np.full((fibre_num_t, 2, 2), np.array([[0.0, 0.0], [0.0, 0.0]]))

    Omega = np.concatenate((np.concatenate((Omega_b, Omega_mid)), Omega_t))
    Omega = np.reshape(Omega, (num_rows, num_cols, 2, 2))

    return Omega, fibre_coords, grid_x, grid_y



def cell_invasion_disc_init(centre, dist, sigma):
    # This function init cells in a disc with centre 'centre' and radius 'centre' with confluency.

    N = int((np.pi * (dist**2)) / (np.pi * ((sigma/2)**2)))

    # randomly init raidus and angel
    R = np.sqrt(np.random.uniform(0, dist**2, N))
    theta = np.random.uniform(0, 2*np.pi, N)

    # obtain their x and y coords:
    x = centre[0] + R * np.cos(theta)
    y = centre[1] + R * np.sin(theta)
    cell_coords = np.column_stack((x, y))

    return cell_coords


def fibre_enclose_celldisc_init(centre, dist, lambda_1_const, lambda_2_const, x_min, x_max, y_min, y_max, gridsize_f, epsion):
    # Fibres inside the circular disc are randomly distributed while near the boundary are aligned along the circlar arcs
    # assume NO periodic boundary conditions and boundaries are assumed to be infinitely away

    grid_y = np.arange(y_min, y_max+gridsize_f, gridsize_f) 
    grid_x = np.arange(x_min, x_max+gridsize_f, gridsize_f) 
    # obtain an array that contains coordinates of the Omega grid points 
    X, Y = np.meshgrid(grid_x, grid_y)
    num_rows, num_cols = X.shape
    xx_1d, yy_1d = X.ravel(), Y.ravel()
    fibre_coords = np.column_stack((xx_1d, yy_1d))

    Omega = np.zeros((num_rows, num_cols, 2, 2), dtype=float)
    lambda_12_const = (lambda_1_const+lambda_2_const)/2

    # loop through grid points:
    for i in range(len(grid_x)):
        for j in range(len(grid_y)):
            d = np.sqrt((X[i, j]-centre[0])**2+(Y[i, j]-centre[1])**2)
            if d <= dist+epsion: # fibres exist
                if d <= dist-epsion: # random fibres
                    v1_theta = np.random.uniform(0.0, np.pi, size=1)
                    v2_theta = v1_theta + np.pi/2
                    if v2_theta > np.pi:
                        v2_theta = v2_theta - 2*np.pi
                    v1_x, v1_y, v2_x, v2_y = np.cos(v1_theta), np.sin(v1_theta), np.cos(v2_theta), np.sin(v2_theta)
                    v1, v2 = np.column_stack((v1_x, v1_y)), np.column_stack((v2_x, v2_y))
                    #Omega[i, j] = lambda_1_const*np.outer(v1, v1) + lambda_2_const*np.outer(v2, v2)
                    Omega[i, j] = lambda_12_const*np.outer(v1, v1) + lambda_12_const*np.outer(v2, v2)
                else: # following the direction of the circular arc
                    v1_x, v1_y = -(Y[i, j]-centre[1]), (X[i, j]-centre[0])
                    v2_x, v2_y = (X[i, j]-centre[0]), (Y[i, j]-centre[1])
                    v1, v2 = np.column_stack((v1_x, v1_y)), np.column_stack((v2_x, v2_y))
                    v1, v2 = v1/norm(v1), v2/norm(v2)
                    Omega[i, j] = lambda_1_const*np.outer(v1, v1) + lambda_2_const*np.outer(v2, v2)

    return Omega, fibre_coords, grid_x, grid_y