#!/bin/bash

# the very first line is a special directive for Bash shell, do not remove
# lines that start with "#SBATCH" are special directives for Slurm
# other lines that start with "#" are comments

#SBATCH --job-name=grad-comp  ## job name
#SBATCH -A jallard_lab     ## account to charge
#SBATCH --partition=standard  
#SBATCH --cpus-per-task=4 ## (-c) number of cpus to allocate (PER ARRAY JOB)
#SBATCH --mem=16G  # Use all available memory on the node
#SBATCH --array=1-675 ## number of array tasks #TODO: replace with 1-45 after initial testing
#SBATCH --ntasks=1 ## (-n) number of tasks to launch (PER ARRAY JOB)
#SBATCH --constraint=intel  # Use Intel nodes hopefully prevents some of the cache issues
#SBATCH --error=/pub/stilese/2026/FlexiSpaceLocal/logs/slurm-%A.err    ## Slurm error  file, %x - job name, %A job id 
#SBATCH --output=/pub/stilese/2026/FlexiSpaceLocal/logs/slurm-%A.out      ## Slurm output file, %x - job name, %A job id
#SBATCH --time=0-03:00:00 # Maximum time. # is this per job or summing each job together? 

# error and out are default from the first job I ran?

# hpc3 environment
module load julia-1.12.5
export JULIA_DEPOT_PATH=/pub/stilese/opt/julia-1.12.5/depot

# Sync NFS cache and check compiled packages exist
sync
echo "Checking compiled cache exists:"
ls -la $JULIA_DEPOT_PATH/compiled/v1.11/FlexiBasicLearning/ 2>/dev/null || echo "FlexiBasicLearning not found in compiled cache!"
echo "Cache file checksums (compare with instantiate output):"
md5sum $JULIA_DEPOT_PATH/compiled/v1.11/FlexiBasicLearning/*.ji 2>/dev/null || echo "No .ji files found"

# Debug: Report environment for troubleshooting precompilation issues
echo "=== Environment Debug Info ==="
echo "Hostname: $(hostname)"
echo "Node CPU model: $(grep 'model name' /proc/cpuinfo | head -1)"
echo "JULIA_DEPOT_PATH: $JULIA_DEPOT_PATH"
echo "Julia version: $(julia --version)"
echo "Julia binary: $(which julia)"
julia -e 'println("Julia Sys.BINDIR: ", Sys.BINDIR)'
julia -e 'println("Julia DEPOT_PATH: ", DEPOT_PATH)'
julia -e 'println("Julia Base.julia_cmd(): ", Base.julia_cmd())'
echo "=== End Debug Info ==="

export JULIA_NUM_THREADS=4
julia -e 'println("Julia threads: ", Threads.nthreads()); println("CPU threads visible: ", Sys.CPU_THREADS)'
julia -e 'using LinearAlgebra; println("BLAS threads: ", BLAS.get_num_threads())'

# Set up the experiment
SOURCEPATH=/pub/stilese/2026/FlexiSpace 
OUTPUTPATH=../FlexiSpaceLocal/exp # just for logging

RUNNAME=flexi-${SLURM_ARRAY_TASK_ID}


cd $SOURCEPATH


# echo "Test1: does parallelized classicalMultInt.jl run on hpc3?"

# Test if project loads without recompilation (should be <5 seconds if precompiled)
echo "=== Testing Model8 load (precompilation check) ==="
echo "Expected: <5 seconds if precompiled, >60 seconds if recompiling"
date
/usr/bin/time -v julia +1.11.6 --project=FlexiBasicLearning -e 'using FlexiBasicLearning; println("FlexiBasicLearning loaded successfully")' 2>&1 | grep -E "(User time|Elapsed|Precompiling|FlexiBasicLearning loaded)"
date
echo "=== End precompilation check ==="

# execute script
INPUT=$SOURCEPATH/scripts/timing_grads_tasks.txt
ARGS=$(awk "NR==$SLURM_ARRAY_TASK_ID" $INPUT)

echo "Arguments: $ARGS"


echo "Running timing_grads-parallel.jl" 
date; /usr/bin/time -v stdbuf -oL julia +1.11.6 --project=FlexiBasicLearning FlexiBasicLearning/src/timing_grads_parallel.jl $ARGS \
> $OUTPUTPATH/logs/$SLURM_ARRAY_JOB_ID.$RUNNAME.$SLURM_ARRAY_TASK_ID.log 2>&1 ; date

# After job completes, move the logs
logdate=$(date +%y%m%d)
mkdir -p "$OUTPUTPATH/logs/$logdate"
# Note: slurm .out/.err files are still open during job - cannot move them here
mv "$OUTPUTPATH/logs/$SLURM_ARRAY_JOB_ID.$RUNNAME.$SLURM_ARRAY_TASK_ID.log" "$OUTPUTPATH/logs/$logdate/"
# max