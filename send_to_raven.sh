#!/bin/bash -l
#
# log files (-o output, -e errors), not created automatically.
#SBATCH -o logs/%j_gpu1_undistr.out.undistr
#SBATCH -e logs/%j_gpu1_undistr.err.undistr
# -D is working directory.
#SBATCH -D ./
# -J is the name of the job. when running multiple jobs, helps check how long it takes, where it is in the queue, etc.
#SBATCH -J lorenz_transformer
#
# 
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=0
#
# only gpu nodes
#SBATCH --constraint="gpu"
# kind of gpu (a100 is the name, 1 is the number of gpus you request, in this case we don't use more)
#SBATCH --gres=gpu:a100:1
#
# if I want to receive notification. "end" means sends notification _to_ the email. you can also set it to "none".
#SBATCH --mail-type=end
#SBATCH --mail-user=niconico.leng@googlemail.com
#
# Wall clock limit (max. is 24 hours), if you need more time you would need to find a way to resume the job:
#SBATCH --time=02:00:00


source /etc/profile.d/modules.sh
# remove all modules that are already loaded
module purge

module load gcc/10 anaconda/3/2020.02 impi/2021.2 pytorch/gpu-cuda-11.2/1.8.0 h5py-mpi/2.10
python -m venv --system-site-packages ${HOME}/venvs/project_env
source ${HOME}/venvs/project_env/bin/activate
pip install --user -e .
pip install --user -r ./docs/requirements.txt
pip install --user gdown

# --- TF related flags (provided by Tim) ---

# Avoid CUPTI warning message
export LD_LIBRARY_PATH=${CUDA_HOME}/extras/CUPTI/lib64:${LD_LIBRARY_PATH}

# Avoid OOM
export TF_FORCE_GPU_ALLOW_GROWTH=true

## XLA
# cuda aware
export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_HOME}"
# enable autoclustering for CPU and GPU
export TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"

srun python examples/lorenz/train_lorenz_transformer.py