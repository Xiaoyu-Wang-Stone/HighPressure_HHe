#!/bin/bash
#SBATCH --account=co_12monkeys
#SBATCH --time=24:00:00
#SBATCH --job-name=HHe_8Nodes
#SBATCH --partition=savio4_htc
#SBATCH --qos=savio_lowprio
#SBATCH -N 8                        # request 8 nodes
#SBATCH --ntasks-per-node=56        # 56 cores per node
#SBATCH --exclusive                 # exclusive node allocation
#SBATCH --exclude=n0120.savio3,n0113.savio3

source ~/.bashrc
module load cmake gcc/10 openmpi/4.1.6 fftw cuda/12.2.1 openblas/0.3.24 gsl/2.7.1 ffmpeg/6.0

# Activate environment
mamba activate /global/home/groups/co_12monkeys/software/conda_envs/mace-stable

# Set LAMMPS path
LMP_EXE="/global/scratch/users/xiaoyuwang/lammps/build/lmp"
export PATH=/global/scratch/users/xiaoyuwang/lammps/build/:$PATH

# Prevent OpenMP from interfering with MPI
export OMP_NUM_THREADS=1

# ======================================================
# 1. Auto-detect master node and update input.xml
# ======================================================
# Get the first allocated node name (Master Node)
MASTER_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
echo "Running on 8 nodes. Master node (i-PI Server): $MASTER_NODE"

# Back up original input.xml (avoid cumulative edits)
if [ ! -f input.xml.orig ]; then
    cp input.xml input.xml.orig
fi

# Replace localhost in input.xml with the real hostname
# Assumes input.xml contains <address>localhost</address>
cp input.xml.orig input.xml
sed -i "s|<address>.*</address>|<address> $MASTER_NODE </address>|g" input.xml

# ======================================================
# 2. Auto-convert Restart -> XYZ (if needed)
# ======================================================
if [ ! -f start.xyz ]; then
    echo "Converting restart file to start.xyz..."
    $LMP_EXE -e both << EOF
    units metal
    atom_style atomic
    read_restart restart_file.restart
    mass 1 1.008
    mass 2 4.0026
    write_dump all xyz start.xyz modify element H He sort id
EOF
fi

# ======================================================
# 3. Start i-PI (Server)
# ======================================================
echo "Starting i-PI server..."
# Use -u so Python logs are unbuffered and real-time
python -u $(which i-pi) input.xml > log.ipi &

# Give i-PI 20 seconds to start (multi-node can be slower)
sleep 20

# ======================================================
# 4. Start LAMMPS (Client) - 8-node mode
# ======================================================
# Total cores: 8 nodes * 56 cores = 448 cores
# Partitioning: -p 8x56 (8 partitions, 56 cores each)
# This allows each bead to run on a full node

echo "Starting LAMMPS: 8 partitions x 56 cores/partition = 448 cores total"

# Note: after input.xml address changes, LAMMPS input must match
# The next line also replaces localhost in npt_PIMD.lmp
sed -i "s|fix ipi all ipi .* 31415|fix ipi all ipi $MASTER_NODE 31415|g" npt_PIMD.lmp

mpirun -np 448 $LMP_EXE -p 8x56 -in npt_PIMD.lmp -screen none

wait
