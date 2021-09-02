#!/bin/sh
#SBATCH --job-name=tf_job_test
# Job name 
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Run on a single CPU 
#SBATCH --time=00:00:10
# Time limit hrs:min:sec
#SBATCH --output=tf_test_%j.out
# Standard output and error log
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0
#SBATCH --mem=1GB
#SBATCH --partition=dgx

echo $CUDA_VISIBLE_DEVICES
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo ""
echo "Number of Nodes Allucated      = $SLURM_JOB_NUM_NODES"
echo "Number of Tasks Allocated      = $SLURM_NTASKS"
echo "Number of Cores/Task Allocated = $SLURM_CPUS_PER_TASK"
echo "Working Directory = $(pwd)"
echo "working directory = "$SLURM_SUBMIT_DIR


pwd; hostname; date |tee result

echo $CUDA_VISIBLE_DEVICES


docker container ls
nvidia-smi
docker stats


docker container ls
nvidia-smi



