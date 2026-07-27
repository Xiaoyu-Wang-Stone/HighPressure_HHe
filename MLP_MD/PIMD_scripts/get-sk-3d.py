import os, sys
import numpy as np 
from math import pi
import time
# Replace ovito with ASE
from ase.io import read

def read_trj(file_path):
    # Read all frames from the trajectory file
    frames = read(file_path, index=':')  # ':' means all frames
    
    # Get cell information from the first frame
    cell = np.array([frames[0].cell[0, 0], frames[0].cell[1, 1], frames[0].cell[2, 2]])
    
    sq_time = []
    
    # Get atom types from the first frame
    natoms = len(frames[0])
    names = np.zeros(natoms, dtype='U3')
    for i, atom in enumerate(frames[0]):
        symbol = atom.symbol
        # Handle potential numerical symbols from the original format
        if symbol == '1':
            names[i] = 'H'
        elif symbol == '2':
            names[i] = 'He'
        else:
            names[i] = symbol
    
    # Process each frame
    for frame in frames:
        # Get positions
        positions = frame.get_positions()
        
        # Convert to fractional coordinates (divide by cell lengths)
        sq = np.divide(positions, cell)
        sq_time.append(sq)
    
    return [cell, names, sq_time]

def FT_density(q, kgrid):
    # This is the un-normalized FT for density fluctuations
    ng = len(kgrid)
    ak = np.zeros(ng,dtype=complex)
    for n,k in enumerate(kgrid):
        ak[n] = np.sum(np.exp(-1j*(q[:,0]*k[0]+q[:,1]*k[1]+q[:,2]*k[2])))
    return ak


def Sk(names, q, kgrid, e_A, e_B):
    # This is the un-normalized FT for the density fluctuations
    q_A = np.asarray([ q_now for i, q_now in enumerate(q) if names[i] in e_A ])
    n_A = len(q_A)
    print("Number of element A: ", n_A)
    if n_A > 0:
        FTrho_A = FT_density(q_A, kgrid)
    else:
        FTrho_A = np.empty(len(kgrid))
        FTrho_A[:] = np.NaN
    if e_A != e_B:
        q_B = np.asarray([ q_now for i,q_now in enumerate(q) if names[i] in e_B ])
        n_B = len(q_B)
        print("Number of element B: ", n_B)
        if n_B > 0:
            FTrho_B = FT_density(q_B, kgrid)
        else:
            FTrho_B = np.empty(len(kgrid))
            FTrho_B[:] = np.NaN
    else:
        FTrho_B = FTrho_A


    return np.multiply(FTrho_A, np.conjugate(FTrho_A))/n_A, \
                   np.multiply(FTrho_A, np.conjugate(FTrho_B))/(n_A*n_B)**0.5, \
                   np.multiply(FTrho_B, np.conjugate(FTrho_B))/n_B



def main(sprefix="Sk", straj="HHe_PIMD_2000K_centroid_converted.xyz", sbins=4):
    # number of k grids
    bins = int(sbins)
    # get total number of bins and initialize the grid
    print("Use number of bins:", bins)
    straj = 'HHe_PIMD_2000K_centroid_converted.xyz'  # Hard-coded trajectory filename
    
    # Outputs
    ofile_AA = open(sprefix+'-HeHe-real.dat',"ab")
    ofile_AB = open(sprefix+'-HeH-real.dat',"ab")
    ofile_BB = open(sprefix+'-HH-real.dat',"ab")
    
    # Read the trajectory file using ASE
    [cell, names, sq_data] = read_trj(straj)
    
    # Calculate start index for last 60% of frames
    total_frames = len(sq_data)
    start_idx = int(total_frames * 0.1)  # Start from 40% to get last 60%
    
    # Process each frame
    nframe = 0
    for frame_idx in range(start_idx, total_frames):
        start_time = time.time()
        nframe += 1
        print("Frame No:", nframe)
    
        if (nframe == 1):
            # normalization
            volume = np.prod(cell[:])
            kgrid = np.zeros((bins*bins*bins,3),float)
            kgridbase = np.zeros((bins*bins*bins,3),float)
            # initialize k grid
            [ dkx, dky, dkz ] = [ 1./cell[0], 1./cell[1], 1./cell[2] ]
            n=0
            for i in range(bins):
                for j in range(bins):
                    for k in range(bins):
                        if i+j+k == 0: pass
                        # initialize k grid
                        kgridbase[n,:] = (2.*pi)*np.array([i, j, k])
                        kgrid[n,:] = [ dkx*i, dky*j, dkz*k ]
                        n+=1
            np.savetxt(sprefix+'-kgrid.dat',kgrid)
        
        # FT analysis of density fluctuations
        sk_AA, sk_AB, sk_BB = Sk(names, sq_data[frame_idx], kgridbase, ['He'], ['H'])
        print("--- %s seconds after FFT density ---" % (time.time() - start_time))
        
        # Outputs
        np.savetxt(ofile_AA,sk_AA[None].real, fmt='%4.4e', delimiter=' ',header="Frame No: "+str(nframe))
        np.savetxt(ofile_AB,sk_AB[None].real, fmt='%4.4e', delimiter=' ',header="Frame No: "+str(nframe))
        np.savetxt(ofile_BB,sk_BB[None].real, fmt='%4.4e', delimiter=' ',header="Frame No: "+str(nframe))


if __name__ == "__main__":
    main()

