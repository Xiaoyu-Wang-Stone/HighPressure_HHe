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


mamba activate /global/home/groups/co_12monkeys/software/conda_envs/mace-stable


LMP_EXE="/global/scratch/users/xiaoyuwang/lammps/build/lmp"
export PATH=/global/scratch/users/xiaoyuwang/lammps/build/:$PATH


export OMP_NUM_THREADS=1


RESTART_FILE="HHe_PIMD_2000K.restart"


MASTER_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
echo "Running on 8 nodes. Master node (i-PI Server): $MASTER_NODE"


cp $RESTART_FILE ${RESTART_FILE}.bak_$(date +%s)


sed -i "s|<address>.*</address>|<address> $MASTER_NODE </address>|g" $RESTART_FILE

echo "Updated address in $RESTART_FILE to $MASTER_NODE"


echo "Resuming i-PI server from checkpoint..."



python -u $(which i-pi) $RESTART_FILE > log.ipi &

sleep 20


echo "Starting LAMMPS clients..."


sed -i "s|fix ipi all ipi .* 31415|fix ipi all ipi $MASTER_NODE 31415|g" npt_PIMD.lmp


mpirun -np 448 $LMP_EXE -p 8x56 -in npt_PIMD.lmp -screen none

