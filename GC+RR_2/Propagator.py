import numpy as np
from collections import Counter

def propagate_forward_directional(iw_field, q_init, u_vecs, s, length, Ks, N_steps):
    Nx, Ny, Nz, N_ang = iw_field.shape
    Delta_s = s / N_steps
    kx, ky, kz = Ks

    q_p = np.ones((Nx, Ny, Nz, N_ang), dtype=q_init.dtype) * q_init[..., None]

    for idx in range(N_ang):
        iw_fft = np.fft.fftn(iw_field[..., idx])
        for m in range(N_steps):
            s_m = m * Delta_s
            shift = (s - s_m) * u_vecs[idx]
            phase = np.exp(-1j*(kx*shift[0] + ky*shift[1] + kz*shift[2]))
            w_shifted = np.fft.ifftn(iw_fft * phase).real
            q_p[..., idx] *= np.exp(-w_shifted * Delta_s)

    return q_p


def propagate_backward_directional(iw_field, q_init, u_vecs, s, length, Ks, N_steps):
    Nx, Ny, Nz, N_ang = iw_field.shape
    Delta_s = (length - s) / N_steps
    kx, ky, kz = Ks

    q_init_fft = np.fft.fftn(q_init)
    q_p = np.ones((Nx, Ny, Nz, N_ang), dtype=q_init.dtype)
    q_p *= np.fft.ifftn(q_init_fft * np.exp(+1j*(kx*(length-s) + ky*(length-s) + kz*(length-s)))).real[..., None].repeat(N_ang, axis=3)

    for idx in range(N_ang):
        iw_fft = np.fft.fftn(iw_field[..., idx])
        for m in range(N_steps):
            s_m = s + m * Delta_s
            shift = (s - s_m) * u_vecs[idx]
            phase = np.exp(+1j*(kx*shift[0] + ky*shift[1] + kz*shift[2]))
            w_shifted = np.fft.ifftn(iw_fft * phase).real
            q_p[..., idx] *= np.exp(-w_shifted * Delta_s)
    return q_p


def propagate_closed(rho0_per_class, w_sidechains, w, diff_factor, ds, box_lengths,
                     residue_classes, u_vectors, ang_weights, Ks,
                     Nsteps_bb, steps_per_residue, sidechains_length=1.5,
                     N_steps_sidechain=3):
    Nx, Ny, Nz, N_ang = w_sidechains[0].shape

    # Precompute scalar weight of attaching sidechains (angular quadrature applied)
    q_sidechains_finalweight_list = [
        np.tensordot(
            propagate_forward_directional(iw, np.ones((Nx, Ny, Nz)), u_vectors, sidechains_length, sidechains_length, Ks, N_steps_sidechain),
            ang_weights, axes=([3], [0])
        )
        for iw in w_sidechains
    ]

    class_labels = np.unique(residue_classes)
    rho_class = {c: np.zeros((Nx, Ny, Nz)) for c in class_labels}

    # Forward backbone propagators (scalar spatial arrays per contour point)
    q = np.zeros((Nsteps_bb+1, Nx, Ny, Nz))
    
    q[0] = q_sidechains_finalweight_list[0]
    for n in range(1, Nsteps_bb+1):
        q[n] = np.exp(-w(n))* np.fft.ifftn(diff_factor*np.fft.fftn(q[n-1]))
        if n % steps_per_residue == 0:
            q[n] *= q_sidechains_finalweight_list[0]
        elif n % steps_per_residue == int((2/3) * steps_per_residue):
            q[n] *= q_sidechains_finalweight_list[1]

    # Backward
    qdag = np.zeros_like(q)
    V = np.prod(box_lengths)
    dV = V / (Nx * Ny * Nz)
    spat_weights = dV * np.ones((Nx, Ny, Nz))
    Q = np.sum(q[Nsteps_bb] * spat_weights)

    q_sidechains_backward_initialdatum_list = np.zeros((2, Nx, Ny, Nz))
    qdag[Nsteps_bb] = q_sidechains_finalweight_list[1]
    q_sidechains_backward_initialdatum_list[1] += q[Nsteps_bb]

    for n in range(Nsteps_bb - 1, -1, -1):

        qdag[n] = np.exp(-w(n))* np.fft.ifftn(diff_factor*np.fft.fftn(qdag[n+1]))
        if n % steps_per_residue == 0:  # N_type_chain
            q_sidechains_backward_initialdatum_list[0] += qdag[n] * q[n] 
            qdag[n] *= q_sidechains_finalweight_list[0]
        elif n % steps_per_residue == int((2 / 3) * steps_per_residue):  # C_type_chain
            q_sidechains_backward_initialdatum_list[1] += qdag[n] * q[n] 
            qdag[n] *= q_sidechains_finalweight_list[1]

        # accumulate backbone density (include ds and divide only by Q)
        rho_class[residue_classes[n]] += (rho0_per_class[residue_classes[n]]/(Counter(residue_classes)[residue_classes[n]] / steps_per_residue)) * q[n] * qdag[n] * ds / Q
    rho_class[residue_classes[Nsteps_bb]] += (rho0_per_class[residue_classes[n]]/(Counter(residue_classes)[residue_classes[n]] / steps_per_residue)) * q[Nsteps_bb] * qdag[Nsteps_bb] * ds / Q
  
    # Sidechains: accumulate with angular quadrature and sidechain contour ds
    rho_sc = np.zeros((2, Nx, Ny, Nz, N_ang), dtype=np.float64)
    s_vals = np.linspace(0, sidechains_length, N_steps_sidechain)
    delta_s_side = s_vals[1] - s_vals[0] if len(s_vals) > 1 else sidechains_length

    for s in s_vals:
        # N-side
        fwd0 = propagate_forward_directional(w_sidechains[0], np.ones((Nx, Ny, Nz)), u_vectors, s, sidechains_length, Ks, N_steps_sidechain)
        bwd0 = propagate_backward_directional(w_sidechains[0], q_sidechains_backward_initialdatum_list[0], u_vectors, s, sidechains_length, Ks, N_steps_sidechain)
        rho_sc[0] += ((rho0_per_class['Nsc']/(len(residue_classes)/steps_per_residue)) * delta_s_side / Q)* fwd0 * bwd0 

        # C-side
        fwd1 = propagate_forward_directional(w_sidechains[1], np.ones((Nx, Ny, Nz)), u_vectors, s, sidechains_length, Ks, N_steps_sidechain)
        bwd1 = propagate_backward_directional(w_sidechains[1], q_sidechains_backward_initialdatum_list[1], u_vectors, s, sidechains_length, Ks, N_steps_sidechain)
        rho_sc[1] += ((rho0_per_class['Csc']/(len(residue_classes)/steps_per_residue)) * delta_s_side / Q) * fwd1 * bwd1 
    
    return rho_class, rho_sc, Q


if __name__ == '__main__':
    Nx = Ny = Nz = int(30/1)
    N_ang = 20
    box_lengths = np.array([30.0, 30.0, 30.0])
    dx = box_lengths[0] / Nx

    # backbone: 3 residues total, steps_per_residue fixed to 6 => Nsteps_bb = 18
    num_residues = 6
    steps_per_residue = 3
    Nsteps_bb = int((num_residues-1/3) * steps_per_residue)
    print(Nsteps_bb)

    # contour spacing in monomer units (so sum(ds over contour) = M_backbone)
    ds = num_residues / float(Nsteps_bb+1)

    # sidechains
    sidechains_length = 1
    N_steps_sidechain = 1

    # directions and weights
    u_vectors = np.random.randn(N_ang, 3)
    u_vectors /= np.linalg.norm(u_vectors, axis=1)[:, None]
    ang_weights = np.ones(N_ang) / N_ang

    # zero chemical potentials
    w_sidechains = [np.zeros((Nx, Ny, Nz, N_ang)), np.zeros((Nx, Ny, Nz, N_ang))]

    def w(s):
        return np.zeros((Nx, Ny, Nz))

    # trivial diffusion factor as before
    b2_over_6 = 2.25
    kx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, d=dx)
    kz = 2 * np.pi * np.fft.fftfreq(Nz, d=dx)
    kx3d, ky3d, kz3d = np.meshgrid(kx, ky, kz, indexing='ij')
    Ks = (kx, ky, kz)

    def precompute_diffusion(k2, b2_over_6):
        return np.exp(- b2_over_6 * k2)

    diff_factor = precompute_diffusion(kx3d**2 + ky3d**2 + kz3d**2, b2_over_6)

    # classes
    residue_classes = ['E']*steps_per_residue + ['A']*steps_per_residue +['E']*steps_per_residue + ['A']*steps_per_residue +['E']*steps_per_residue + ['A']*steps_per_residue 
    rho0_per_class = {c: 0.3 for c in ['E','A','Nsc','Csc']}
    print(rho0_per_class)
    # run
    rho_class, rho_sc, Q = propagate_closed(
        rho0_per_class, w_sidechains, w, diff_factor, ds,
        box_lengths, np.array(residue_classes), u_vectors, ang_weights, Ks,
        Nsteps_bb, steps_per_residue, sidechains_length=sidechains_length,
        N_steps_sidechain=N_steps_sidechain
    )

    # integrals
    V = np.prod(box_lengths)
    dV = V / (Nx * Ny * Nz)
    print("Partition function Q =", Q)
    print(f'Steps, per: residue {steps_per_residue}, sidechain {N_steps_sidechain}')

    print("Backbone totals:")
    for k, v in rho_class.items():
        total = np.sum(v) * dV
        print(f"{k}: {total}")

    print("Sidechain totals:")
    for lbl, sc in zip(['Nsc', 'Csc'], rho_sc):
        total = np.sum(np.sum(sc*ang_weights, axis = -1) * dV)  # ang_weights already applied
        print(f"{lbl}: {total}")
