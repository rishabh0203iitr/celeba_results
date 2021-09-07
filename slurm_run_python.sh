#!/bin/sh
#SBATCH --job-name=tf_job_test
# Job name
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Run on a single CPU
#SBATCH --time=100:00:00
# Time limit hrs:min:sec
#SBATCH --output=tf_test_%j.out
# Standard output and error log
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24GB
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
echo "working directory = $SLURM_SUBMIT_DIR"


pwd; hostname; date |tee result

echo $CUDA_VISIBLE_DEVICES

NV_GPU=2 nvidia-docker run --network common --rm  --ipc=host -t ${USER_TTY} --name $SLURM_JOB_ID --user $(id -u):$(id -g) -v /raid/ysharma_me/fair_lr/LfF/:/raid/ysharma_me/fair_lr/LfF  pytorchlightning-mod-ysharma/pytorch-lightning:base-conda-try-1 python /raid/ysharma_me/fair_lr/LfF/train.py with server_user celeba

















#docker images

#docker container ls
