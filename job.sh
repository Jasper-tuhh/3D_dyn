#!/bin/bash -l
#SBATCH --job-name=hertz3d
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4000
#SBATCH --time=12:00:00
#SBATCH --output=hertz_%x_%j.out
#SBATCH --error=hertz_%x_%j.err

# Usage:  sbatch --export=ALL,SWEEP=h job.sh        (SWEEP = h | p | hp | fine)
# Optional overrides: MATERIAL_MODEL, REG_METHOD, TARGET_LOAD

. /etc/profile.d/module.sh
module purge
module load anaconda/2023.07-1
eval "$(conda shell.bash hook)"
conda activate $WORK/envs/fenicsx

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg

export MATERIAL_MODEL=${MATERIAL_MODEL:-neo_hooke}
export REG_METHOD=${REG_METHOD:-huhu_lulu}
export REG_VALUE=${REG_VALUE:-1e-3}
export TARGET_LOAD=${TARGET_LOAD:--600}

# ---- sweep definitions: edit the values here ----
# h = element size ON THE SPHERE SURFACE at the contact pole (grows away from it).
# Size of the problem (p=2): h=0.1 -> 118k dofs, 0.07 -> 174k, 0.05 -> 245k, 0.035 -> 349k,
# 0.025 -> 459k.  p=3 is ~3.3x larger.  p must be >= 2 (Hessian regularisation).
# Rule of thumb for MUMPS in 3D: roughly 60-100 GB total for 450k dofs -> raise --ntasks / mem.
case "$SWEEP" in
  h)    export H_VALUES="0.1,0.07,0.05,0.035"; export P_VALUES="2" ;;
  p)    export H_VALUES="0.15";                export P_VALUES="2,3" ;;
  hp)   export H_VALUES="0.15,0.1";            export P_VALUES="2,3" ;;
  fine) export H_VALUES="0.025";               export P_VALUES="2" ;;
  *)  echo "SWEEP must be h, p, hp or fine (got '$SWEEP')"; exit 1 ;;
esac
echo "SWEEP=$SWEEP  H_VALUES=$H_VALUES  P_VALUES=$P_VALUES  MATERIAL_MODEL=$MATERIAL_MODEL  REG_METHOD=$REG_METHOD  REG_VALUE=$REG_VALUE  TARGET_LOAD=$TARGET_LOAD"

MYWORKDIR=$SCRATCH/$SLURM_JOBID
mkdir -p "$MYWORKDIR"
cp "$SLURM_SUBMIT_DIR"/*.py "$MYWORKDIR"/
cd "$MYWORKDIR"
export XDG_CACHE_HOME="$MYWORKDIR/cache"; export HOME="$MYWORKDIR"; mkdir -p "$XDG_CACHE_HOME"

RESULTDIR=$SLURM_SUBMIT_DIR/results3d_${SWEEP}_${MATERIAL_MODEL}_$SLURM_JOBID
save_results() {
    mkdir -p "$RESULTDIR"
    cp -r hertz3d_sweep_* "$RESULTDIR"/ 2>/dev/null
    cd "$SLURM_SUBMIT_DIR" && rm -rf "$MYWORKDIR"
}
trap save_results EXIT

mpirun -n "$SLURM_NTASKS" python hertz_3d.py