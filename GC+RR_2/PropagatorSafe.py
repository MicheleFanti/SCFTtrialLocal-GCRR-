import numpy as np

def forward_step(q, w, diff_factor, ds, buf=None):
    if buf is None: buf = np.empty_like(q)
    tw = np.exp(-0.5*ds*w)
    buf[:] = tw * q
    buf[:] = np.fft.ifftn(diff_factor * np.fft.fftn(buf)).real
    buf *= tw
    return buf

def adjoint_step(qdag, w, diff_factor, ds, buf=None):
    if buf is None: buf = np.empty_like(qdag)
    tw = np.exp(+0.5*ds*w)
    buf[:] = tw * qdag
    buf[:] = np.fft.ifftn(diff_factor * np.fft.fftn(buf)).real
    buf *= tw
    return buf

def propagate_forward_directional(iw_field, q_init, u_vecs, s, length, Ks, N_steps):
    Nx, Ny, Nz, N_ang = iw_field.shape
    Delta_s = s / N_steps
    kx, ky, kz = Ks
    k_stack = np.stack((kx, ky, kz), axis=-1)

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
    q_p = np.fft.ifftn(q_init_fft * np.exp(+1j*(kx*(length-s) + ky*(length-s) + kz*(length-s)))).real[..., None].repeat(N_ang, axis=3)

    for idx in range(N_ang):
        iw_fft = np.fft.fftn(iw_field[..., idx])
        for m in range(N_steps):
            s_m = s + m * Delta_s
            shift = (s - s_m) * u_vecs[idx]
            phase = np.exp(+1j*(kx*shift[0] + ky*shift[1] + kz*shift[2]))
            w_shifted = np.fft.ifftn(iw_fft * phase).real
            q_p[..., idx] *= np.exp(-w_shifted * Delta_s)
    return q_p

def propagate_closed(rho0_per_class, w_sidechains, w, diff_factor, ds, box_lengths, residue_classes, u_vectors, ang_weights, Ks, Nsteps_bb, steps_per_residue, sidechains_length=1.5):
    Nx, Ny, Nz, N_ang = w_sidechains[0].shape
    q_sidechains_finalweight_list = [np.tensordot(propagate_forward_directional(iw, np.ones((Nx, Ny, Nz)),u_vectors, sidechains_length, sidechains_length, Ks, 3), ang_weights, axes =([3], [0])) for iw in w_sidechains]
    class_labels = np.unique(residue_classes)
    rho_class = {c: np.zeros((Nx,Ny,Nz)) for c in class_labels}
    q = np.zeros((Nsteps_bb, Nx, Ny, Nz))
    q[0] = q_sidechains_finalweight_list[0]
    for n in range(1, Nsteps_bb):
        q[n] = forward_step(q[n-1], w((n-1)*ds), diff_factor, ds, None)
        if n%steps_per_residue == 0:  #N_type_chain
            q[n] *= q_sidechains_finalweight_list[0]
        elif n%steps_per_residue == int((2/3)*steps_per_residue):
            q[n] *= q_sidechains_finalweight_list[1]
        #print(f'Forward:{q[n]}')
    qdag = np.zeros_like(q)
    V = np.prod(box_lengths)
    spat_weights = V/(Nx*Ny*Nz)*np.ones((Nx, Ny, Nz))
    #spat_weights /= V
    Q = np.sum(q[Nsteps_bb-1]*spat_weights)
    q_sidechains_backward_initialdatum_list = np.zeros((2, Nx, Ny, Nz))
    qdag[Nsteps_bb-1] = q_sidechains_finalweight_list[1]
    for n in range(Nsteps_bb-2, 0, -1):
        qdag[n] = adjoint_step(qdag[n+1], w((n+1)*ds), diff_factor, ds, None)
        if n%steps_per_residue == 0:  #C_type_chain
            q_sidechains_backward_initialdatum_list[1] += qdag[n]*q[n]
            qdag[n] *= q_sidechains_finalweight_list[1]
        elif n%steps_per_residue == int((2/3)*steps_per_residue): #N_type_chain
            q_sidechains_backward_initialdatum_list[0] += qdag[n]*q[n]
            qdag[n] *= q_sidechains_finalweight_list[0]
        rho_class[residue_classes[n]] += 3*(ds)*rho0_per_class[residue_classes[n]]*q[n]*qdag[n]/(Q)
        

    rho_sc = np.zeros((2, Nx, Ny, Nz, N_ang), dtype = np.float32)

    N_steps_sidechain = 3
    s_vals = np.linspace(0, sidechains_length, N_steps_sidechain)
    for s in s_vals:
        rho_sc[0] += rho0_per_class['Nsc']*(
            propagate_forward_directional(
                w_sidechains[0],
                np.ones((Nx, Ny, Nz)),
                u_vectors,
                s,
                sidechains_length,
                Ks,
                N_steps=N_steps_sidechain
            )
            * propagate_backward_directional(
                w_sidechains[0],
                q_sidechains_backward_initialdatum_list[0],
                u_vectors,
                s,
                sidechains_length,
                Ks,
                N_steps=N_steps_sidechain
            )
        )/(Q)
        rho_sc[1] += rho0_per_class['Csc']* (
            propagate_forward_directional(
                w_sidechains[1],
                np.ones((Nx, Ny, Nz)),
                u_vectors,
                s,
                sidechains_length,
                Ks,
                N_steps=N_steps_sidechain
            )
            * propagate_backward_directional(
                w_sidechains[1],
                q_sidechains_backward_initialdatum_list[1],
                u_vectors,
                s,
                sidechains_length,
                Ks,
                N_steps=N_steps_sidechain
            )
        )/(Q)
    return rho_class, rho_sc, Q

# -------------------------------
# Parameters
Nx = Ny = Nz = 10
N_ang = 20
box_lengths = np.array([30.0, 30.0, 30.0])
dx = box_lengths[0]/Nx
# backbone
Nsteps_bb = 36
steps_per_residue = 6
b2_over_6 = 2.25 

# sidechains
sidechains_length = 1.5
u_vectors = np.random.randn(N_ang, 3)
u_vectors /= np.linalg.norm(u_vectors, axis=1)[:,None]  # normalize
ang_weights = np.ones(N_ang) / N_ang  # normalized weights

# zero chemical potentials
w_sidechains = [np.zeros((Nx,Ny,Nz,N_ang)),
                np.zeros((Nx,Ny,Nz,N_ang))]

def w(s): return np.zeros((Nx,Ny,Nz))
ds = 1.0/Nsteps_bb

# classes (3 backbone + 2 sidechains)
residue_classes = ['E']*steps_per_residue + ['A']*steps_per_residue + ['D']*steps_per_residue \
                  + ['E']*steps_per_residue + ['A']*steps_per_residue + ['D']*steps_per_residue
rho0_per_class = {c: 0.2 for c in ['E','A','D','Nsc','Csc']}

kx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
ky = 2 * np.pi * np.fft.fftfreq(Ny, d=dx)
kz = 2 * np.pi * np.fft.fftfreq(Nz, d=dx)
kx3d, ky3d, kz3d = np.meshgrid(kx, ky, kz, indexing='ij')
Ks = (kx, ky, kz)
def precompute_diffusion(k2, b2_over_6, ds):
    return np.exp(- ds * b2_over_6 * k2)

# -------------------------------
# Run
rho_class, rho_sc, Q = propagate_closed(
    rho0_per_class, w_sidechains, w, precompute_diffusion(kx**2+ky**2+kz**2, b2_over_6, ds), ds,
    box_lengths, np.array(residue_classes),
    u_vectors, ang_weights, Ks,
    Nsteps_bb, steps_per_residue,
    sidechains_length=sidechains_length
)

# -------------------------------
# Integrals
V = np.prod(box_lengths)
dV = V/(Nx*Ny*Nz)
print("Partition function Q =", Q)

print("Backbone totals:")
for k,v in rho_class.items():
    total = np.sum(v)*dV
    print(f"{k}: {total}")

print("Sidechain totals:")
for lbl,sc in zip(['Nsc','Csc'], rho_sc):
    total = np.sum(sc)*dV*np.sum(ang_weights)
    print(f"{lbl}: {total}")