import numpy as np
import sys

from scipy.integrate import lebedev_rule

from Propagator import propagate_closed
from AAHydrophobic import gather_charges_and_lambdas
from AndersonMixing import SmallAnderson

from collections import Counter


def lebedev_grid(N_ang):
    try:
        from scipy.integrate import lebedev_rule
        pts, w = lebedev_rule(N_ang)
        u = np.array(pts)
        w = np.array(w)
        return u, w
    except Exception:
        i = np.arange(0, N_ang)
        phi = 2.0 * np.pi * i / ((1 + np.sqrt(5)) / 2.0)
        z = 1.0 - (2.0*i + 1.0)/N_ang
        r = np.sqrt(1.0 - z*z)
        u = np.stack((r * np.cos(phi), r * np.sin(phi), z), axis=1)
        w = np.ones(N_ang) * (4*np.pi / N_ang)
        return u, w


def precompute_diffusion(k2, b2_over_6):
    return np.exp(- b2_over_6 * k2)

def F_ev(rho_p3d, u_kernel_k, vchi_pp):
    return 0.5*vchi_pp* np.fft.ifftn(np.fft.fftn(rho_p3d) * u_kernel_k).real

def F_rs_a(h_a, lambda_a, V_kernel_k):
    return 0.5*lambda_a*np.fft.ifftn(np.fft.fftn(h_a) * V_kernel_k).real

def F_hb_pb(fourier_kernel, ang_kernel, ang_weights, rho_ru):
    Nx, Ny, Nz, Nang = rho_ru.shape
    rho_flat = rho_ru.reshape(-1, Nang)
    M = ang_weights[:, None] * ang_kernel.T
    I_flat = rho_flat @ M
    I = I_flat.reshape(Nx, Ny, Nz, Nang)
    out = np.fft.ifftn(np.fft.fftn(I, axes=(0,1,2)) * fourier_kernel[..., None], axes=(0,1,2)).real
    return out

#Compute the chemical potentials given explicit density fields
def Nchempot(rhoC, rhoP, epsilon_hb, hb_kernel_k,hb_kernel_ang, ang_weights, vchi_pp, u_ev_k, alpha_pb, f_pb_k, ang_pb):
    return F_ev(rhoP, u_ev_k, vchi_pp)[..., None]*np.ones_like(rhoC) -epsilon_hb*0.5* F_hb_pb(hb_kernel_k, hb_kernel_ang, ang_weights, rhoC) +alpha_pb* F_hb_pb(f_pb_k, ang_pb, ang_weights, rhoC)

def Cchempot(rhoN, rhoP, epsilon_hb, hb_kernel_k,hb_kernel_ang, ang_weights, vchi_pp, u_ev_k, alpha_pb, f_pb_k, ang_pb):
    return F_ev(rhoP, u_ev_k, vchi_pp)[..., None]*np.ones_like(rhoN) -epsilon_hb*0.5* F_hb_pb(hb_kernel_k, hb_kernel_ang, ang_weights, rhoN) +alpha_pb* F_hb_pb(f_pb_k, ang_pb, ang_weights, rhoN)
                                                                                                                                                     
def chempot_per_resclass(s, sequence, steps_per_residue, rhoP, u_ev_k, vchi_pp, lambdas, charges, h_as, V_kernel_k):
    res_spec_contrib = 0
    res_idx = int(s / steps_per_residue)
    if res_idx >= len(sequence):
        res_idx = len(sequence) - 1
    for a_key in charges:
        res_spec_contrib += charges[a_key][res_idx] * F_rs_a(h_as[a_key], lambdas[a_key], V_kernel_k)
    return res_spec_contrib + F_ev(rhoP, u_ev_k, vchi_pp)

def update_eta_mu(rho_class, rho0_per_class, eta, mu, dV, ang_weights, relax_eta, relax_mu):
    copy_dict = dict(rho_class)
    if 'Nsc' in copy_dict and copy_dict['Nsc'].ndim == 4:
        copy_dict['Nsc'] = np.sum(copy_dict['Nsc'] * ang_weights[None, None, None, :], axis=-1)
    if 'Csc' in copy_dict and copy_dict['Csc'].ndim == 4:
        copy_dict['Csc'] = np.sum(copy_dict['Csc'] * ang_weights[None, None, None, :], axis=-1)

    Deltarho =0 
    for c_key in copy_dict:
        Deltarho += np.sum(copy_dict[c_key]*dV) 
    eta += relax_eta * (Deltarho- sum(rho0_per_class.values()))
    
    for c_key in copy_dict:
        if c_key in mu and c_key in rho0_per_class:
            total_rho_c = np.sum(copy_dict[c_key]*dV)
            mu[c_key] += relax_mu * (total_rho_c - rho0_per_class[c_key])
    return eta, mu

def ft_gaussian(k, sigma, l=0): # added l=0 default for non-shifted kernels
    return np.exp(-0.5*(k*sigma)**2)*np.exp(-1j*k*l)

def ft_yukawa(k, eps, decay):
    return eps / (1.0 + (k*decay)**2)

def make_3D_kernel(ft_func, kx3d, ky3d, kz3d, *params):
    k2 = kx3d**2 + ky3d**2 + kz3d**2
    return ft_func(np.sqrt(k2), *params)

def build_angular_kernel(N_ang, u_vectors, theta_0):
    kernel = np.zeros((N_ang, N_ang))
    for idxu, u in enumerate(u_vectors): 
        for idxv, v in enumerate(u_vectors): 
            kernel[idxu, idxv] = (np.tensordot(u, v, axes=([0], [0])) - np.cos(theta_0))**2
    return kernel

def main(sequence, rhop0, gridshape, box_lengths, b2_over_6, dx, u_vectors, ang_weights, epsilon_hb, vchi_pp, alpha_pb, relax_eta, relax_mu):
    # Parameters
    Nx, Ny, Nz, Nang = gridshape
    spatial_weights = np.ones((Nx, Ny, Nz)) * (dx**3) / np.prod(box_lengths)
    kx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, d=dx)
    kz = 2 * np.pi * np.fft.fftfreq(Nz, d=dx)
    kx3d, ky3d, kz3d = np.meshgrid(kx, ky, kz, indexing='ij')
    Ks = (kx, ky, kz)

    anderson_eta = SmallAnderson(m=15, beta=-0.6)
    anderson_mu  = SmallAnderson(m=15, beta=-0.6)

    steps_per_residue = 3
    Nsteps_bb = int((len(sequence)-1/3) * steps_per_residue)
    l_chain = len(sequence) * (3.8 + 2)#Angstrom 
    V = np.prod(box_lengths)
    spat_weights = V/(Nx*Ny*Nz)*np.ones((Nx, Ny, Nz))
    print(f'Check on spatial weights: {np.sum(spat_weights)}, {V}')
    #spat_weights /= V

    ds =  len(sequence)/ Nsteps_bb # Contour step size
    diff_factor = precompute_diffusion(kx**2 + ky**2 + kz**2, b2_over_6)

    sigma_pb = 0.2   # Pept-bond Gaussian kernel width
    sigma_ev = 0.2   # Excluded volume Gaussian kernel width
    sigma_hb = 0.5   # Hydrogen bond Gaussian kernel width (fixed median)
    length_hb = 0.5  # Hydrogen bond length (median)
    eps_yukawa = 1.0 # Yukawa kernel strength
    decay_yukawa = 0.5 # Yukawa kernel decay
    
    K_pb = make_3D_kernel(ft_gaussian, kx3d, ky3d, kz3d, sigma_pb, 0)
    K_ev = make_3D_kernel(ft_gaussian, kx3d, ky3d, kz3d, sigma_ev, 0)
    K_hb = make_3D_kernel(ft_gaussian, kx3d, ky3d, kz3d, sigma_hb, length_hb)
    V_k = make_3D_kernel(ft_yukawa, kx3d, ky3d, kz3d, eps_yukawa, decay_yukawa)
    hb_kernel_ang = build_angular_kernel(Nang, u_vectors, 0)
    pb_kernel_ang = build_angular_kernel(Nang, u_vectors, np.pi)
   

    # Count occurrences of each residue in the sequence
    residue_counts = Counter(sequence)
    residue_classes = list(set(sequence))  # Unique and sorted residue names
    
    residue_classes_per_s = []
    for s in range(len(sequence)):
        for i in range(steps_per_residue):
            residue_classes_per_s.append(sequence[s])

    rho0_per_class = {}
    for res in residue_classes:
        n_occ = residue_counts[res]
        rho0_per_class[res] = (3.8 / l_chain) * rhop0 * (n_occ)
    rho_class = {res_key: np.ones((Nx, Ny, Nz)) * rho0_per_class[res_key] for res_key in residue_classes}
    rho0_per_class['Nsc'] = (1.5 / l_chain) * len(sequence) * rhop0
    rho0_per_class['Csc'] = (1.5 / l_chain) * len(sequence) * rhop0
    rho_class['Nsc'] = np.ones((Nx, Ny, Nz, Nang)) * rho0_per_class['Nsc']
    rho_class['Csc'] = np.ones((Nx, Ny, Nz, Nang)) * rho0_per_class['Csc']
    charges, lambdas = gather_charges_and_lambdas(sequence)
    
    residue_classes.append('Nsc')
    residue_classes.append('Csc')
    total_current_rho = 0.0
    max_diff = 0.0
    for c_key in rho0_per_class:
            if c_key == 'Nsc':
                mean_rho = (1/V)*np.sum(np.sum(rho_class['Nsc'] * ang_weights, axis=-1) * spat_weights)
                print(f'Current NSc total {mean_rho}')
            elif c_key == 'Csc':
                mean_rho = (1/V)* np.sum(np.sum(rho_class['Csc'] * ang_weights, axis=-1) * spat_weights)
                print(f'Current CSc total {mean_rho}')
            else:
                mean_rho = (1/V)* np.sum(rho_class[c_key] * spat_weights)
                print(f'Current {c_key} total {mean_rho}')
            total_current_rho += mean_rho
            diff = abs(mean_rho - rho0_per_class[c_key])
            if diff > max_diff:
                max_diff = diff
    total_target_rho = sum(rho0_per_class.values())
    total_diff = abs(total_current_rho - total_target_rho)
    print(f"Iter {-1}: Max per-class diff={max_diff:.5g}, Total diff={total_diff:.5g}")

    
    # --- Initialize fields ---
    eta = np.zeros((Nx, Ny, Nz))  # global Lagrange multiplier for total density
    mu = {res_key: np.array(0.0) for res_key in rho0_per_class}  # per-class Lagrange multipliers
    
    def make_chempot_fields(eta_field, mu_dict, current_rho_class, current_alpha_pb, current_vchi_pp, current_epsilon_hb, h_as_fields, V_k_kernel, K_ev_kernel, K_hb_kernel, K_pb_kernel, hb_kernel_ang_param, ang_pb_param):
        w_sidechains = []
        # Compute total polymer density, summing over Nx, Ny, Nz for all classes,
        # but for 'N' and 'C' classes, first integrate over angles with ang_weights
        rhoP= 0.0
        for key, val in current_rho_class.items():
            if key in ['Nsc', 'Csc']:
                print(f'Key:{key}, val.shape:{val.shape}')
                rhoP +=  np.sum(val * ang_weights[None, None, None, :], axis=-1)
            else:
                rhoP += val

        for terminal_type in ['Nsc', 'Csc']:
            if terminal_type == 'Nsc':
                w = Nchempot(
                    current_rho_class['Csc'] , 
                    rhoP,  
                    current_epsilon_hb,
                    K_hb_kernel, hb_kernel_ang_param, ang_weights, current_vchi_pp, K_ev_kernel, current_alpha_pb, K_pb_kernel, ang_pb_param
                )
            else: 
                w = Cchempot(
                    current_rho_class['Nsc'] ,  # rhoN (angular)
                    rhoP,  # rhoP (integrated)
                    current_epsilon_hb,
                    K_hb_kernel, hb_kernel_ang_param, ang_weights, current_vchi_pp, K_ev_kernel, current_alpha_pb, K_pb_kernel, ang_pb_param
                )
            w_sidechains.append(eta_field[..., None]*np.ones((Nx, Ny, Nz, Nang)) + mu_dict.get(terminal_type, 0.0)[..., None]*np.ones((Nx, Ny, Nz, Nang))) # Add eta and mu contributions

        def w_backbone(s):
            return  eta_field + mu_dict[sequence[int(s/steps_per_residue)]] #+ chempot_per_resclass(s, sequence, steps_per_residue, rhoP, K_ev_kernel, current_vchi_pp,lambdas, charges, h_as_fields, V_k_kernel) 
        return w_sidechains, w_backbone


    print("Starting SCFT iteration...")
    print(f"Target per-class fractions: {rho0_per_class}")

    current_alpha_pb = 0.0
    current_vchi_pp = 0.0
    max_iter = 150
    tol = 1e-4
    for it in range(max_iter):
        
        for res_name in residue_classes:
            print(f'{res_name}; {rho_class[res_name].shape}')
        print(f"\nIteration {it + 1}")
        h_as = {}
        aa_to_idx = {aa:i for i,aa in enumerate(sequence)} # Map AA to its index for charges array
        for a_key in charges: # Loop through modes a0, a1, ...
            h_a_val = np.zeros((Nx, Ny, Nz))
            for res_name in residue_classes: # Loop through actual amino acid names
                if res_name in rho_class: # Ensure we have density for this residue
                    res_idx_in_charges = aa_to_idx.get(res_name, -1) # Get index in charges array
                    if res_idx_in_charges != -1: # Only add if it's a known amino acid
                        # If N or C, integrate over angles
                        if res_name == 'Nsc' or res_name == 'Csc':
                            h_a_val += np.sum(rho_class[res_name] * ang_weights, axis=-1) * charges[a_key][res_idx_in_charges]
                        else:
                            h_a_val += rho_class[res_name] * charges[a_key][res_idx_in_charges]
            h_as[a_key] = h_a_val
        
        current_alpha_pb = min(alpha_pb, (it)/100 * alpha_pb)
        current_vchi_pp = min(vchi_pp, (it)/100 * vchi_pp)
        print(f'Residue classes per s {residue_classes_per_s}')
        w_sidechains, w_backbone = make_chempot_fields(
            eta, mu, rho_class, current_alpha_pb, current_vchi_pp, epsilon_hb, h_as, V_k, K_ev, K_hb, K_pb, hb_kernel_ang, pb_kernel_ang)
        rho_class_new, rho_sc_output, Q = propagate_closed(
            rho0_per_class, w_sidechains, w_backbone, diff_factor, ds, box_lengths, residue_classes_per_s, u_vectors, ang_weights, Ks, Nsteps_bb, steps_per_residue, sidechains_length=1.5)
        # Update backbone densities
        dV = V / (Nx * Ny * Nz)

        for res_key in rho_class_new:
            rho_class[res_key] = rho_class_new[res_key]
        # Update sidechain densities (Nsc and Csc) with the full Nx,Ny,Nz,Nang tensors
        rho_class['Nsc'] = rho_sc_output[0]
        rho_class['Csc'] = rho_sc_output[1]

        total_current_rho = 0
        max_diff = 0
        res_vecs = {}  # collect residuals per class

        for c_key in rho0_per_class:
            if c_key == 'Nsc':
                mean_rho = np.sum(np.sum(rho_sc_output[0] * ang_weights, axis=-1) * spat_weights)
                print(f'Current NSc total {mean_rho}')
            elif c_key == 'Csc':
                mean_rho = np.sum(np.sum(rho_sc_output[1] * ang_weights, axis=-1) * spat_weights)
                print(f'Current CSc total {mean_rho}')
            else:
                mean_rho = np.sum(rho_class[c_key] * spat_weights)
                print(f'Current {c_key} total {mean_rho}')
            total_current_rho += mean_rho

            # residual per-class
            res_vecs[c_key] = mean_rho - rho0_per_class[c_key]
            diff = abs(res_vecs[c_key])
            if diff > max_diff:
                max_diff = diff

        total_diff = total_current_rho - rhop0

        # residuals as arrays
        res_eta = np.array([total_diff])  # scalar constraint
        res_mu  = np.array([res_vecs[c] for c in rho0_per_class if c in mu])  # vector constraint

        # store history and mix
        if it < 15:
            if np.abs(total_diff) < 5e-3:
                relax_eta = 0.05
                relax_mu = 0.05
            elif it > 5:
                eta, mu = update_eta_mu(rho_class, rho0_per_class, eta, mu, dV, ang_weights, 0.2+(it-5)*0.2, 0.2+(it-5)*0.2)
            else:
                eta, mu = update_eta_mu(rho_class, rho0_per_class, eta, mu, dV, ang_weights, relax_eta, 0.00)
        else:
            # Anderson mixing stage
            eta_vec = np.array([eta])
            mu_vec  = np.array([mu[c] for c in rho0_per_class if c in mu])

            anderson_eta.push(eta_vec, res_eta)
            anderson_mu.push(mu_vec, res_mu)

            eta_new = anderson_eta.mix()[0]
            mu_new_vec = anderson_mu.mix()

            # unpack back into dict
            for i, c in enumerate([c for c in rho0_per_class if c in mu]):
                mu[c] = mu_new_vec[i]
            eta = eta_new


        print(f"Iter {it+1}: Max per-class diff={max_diff:.5g}, Total diff={total_diff:.5g}")
        print(f"Current alpha_pb: {current_alpha_pb:.3f}, Current vchi_pp: {current_vchi_pp:.3f}\n")

# ----------------------------------
# Parameters and Launch
# ----------------------------------
sequence = "ADE"
Nx, Ny, Nz = 30, 30, 30
Nang = 50
lx_grid, ly_grid, lz_grid = Nx*0.75, Nx*0.75, Nx*0.75 
box_lengths =(lx_grid, ly_grid, lz_grid)
dx =lx_grid/Nx #its 0.75 A
b2_over_6 = 2.25 
rhop0 = 0.5

steps_per_residue = 3
N_steps_sidechain = 1

epsilon_hb = 0
vchi_pp = 0
alpha_pb = 0

u_vectors, ang_weights = lebedev_grid(Nang)
ang_weights /= np.sum(ang_weights)
u_vectors = u_vectors / np.linalg.norm(u_vectors, axis=1)[:, None]
print(np.sum(ang_weights))

main(sequence, rhop0, (Nx, Ny, Nz, Nang), (lx_grid, ly_grid, lz_grid), b2_over_6, dx, u_vectors, ang_weights, epsilon_hb, vchi_pp, alpha_pb, 0.5, 0.0)

