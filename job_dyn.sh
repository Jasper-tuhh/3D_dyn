#!/bin/bash -l
#SBATCH --job-name=hertz3d_dyn
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4000
#SBATCH --time=24:00:00
#SBATCH --output=hertzdyn_%x_%j.out
#SBATCH --error=hertzdyn_%x_%j.err

# Usage (run from the directory that contains hertz_3d_dyn.py and its library files):
#   sbatch job_dyn.sh                                      # SWEEP=single (h=0.05, p=2, T_END=1.5)
#   sbatch --export=ALL,SWEEP=check job_dyn.sh             # cheap smoke test (coarse mesh, short run)
#   sbatch --export=ALL,SWEEP=h     job_dyn.sh             # mesh sweep  H_VALUES=0.08,0.05,0.035
#   sbatch --export=ALL,SWEEP=p     job_dyn.sh             # degree sweep P_VALUES=2,3
#   sbatch --export=ALL,SWEEP=single,RHO_BALL=2,T_END=2.0,NEWMARK_GAMMA=0.5 job_dyn.sh
#
# SWEEP = check | single | h | p
# Optional overrides (comma-separated after ALL,):
#   H_VALUES, P_VALUES, MATERIAL_MODEL (neo_hooke), REG_METHOD (huhu_lulu|huhu), REG_VALUE (1e-3),
#   T_END (1.5), DT_FREE (4e-3), DT_CONTACT (5e-4), DT_MIN (1e-6), GAP_FINE (0.25),
#   RHO_BALL (1.0), RHO_SUB (1.0), RHO_MEDIUM (1e-3), G (9.81), NEWMARK_GAMMA (0.6),
#   ALPHA_M (0), N_SHELL (6), WRITE_EVERY (5), VERBOSE (0)

. /etc/profile.d/module.sh
module purge
module load anaconda/2023.07-1
eval "$(conda shell.bash hook)"
conda activate $WORK/envs/fenicsx

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg

SWEEP="${SWEEP:-single}"

# ---- sweep definitions: edit the values here ----
# h = element size on the sphere surface at the contact pole (smaller = finer, much more expensive:
# h=0.025 is ~450k dofs at p=2; h=0.05 is roughly 8x fewer). p must be >= 2.
case "$SWEEP" in
  check)  export H_VALUES="${H_VALUES:-0.1}";             export P_VALUES="${P_VALUES:-2}";   export T_END="${T_END:-0.6}"; export DT_FREE="${DT_FREE:-1e-2}" ;;
  single) export H_VALUES="${H_VALUES:-0.05}";            export P_VALUES="${P_VALUES:-2}" ;;
  h)      export H_VALUES="${H_VALUES:-0.08,0.05,0.035}"; export P_VALUES="${P_VALUES:-2}" ;;
  p)      export H_VALUES="${H_VALUES:-0.08}";            export P_VALUES="${P_VALUES:-2,3}" ;;
  *) echo "SWEEP must be check, single, h or p (got '$SWEEP')"; exit 1 ;;
esac

echo "node: $(hostname)  kernel: $(uname -r)  glibc: $(ldd --version | head -1)"
echo "SWEEP=$SWEEP H_VALUES=$H_VALUES P_VALUES=$P_VALUES"
echo "dynamics (unset = script default): T_END=${T_END:-} DT_FREE=${DT_FREE:-} DT_CONTACT=${DT_CONTACT:-} DT_MIN=${DT_MIN:-} GAP_FINE=${GAP_FINE:-} RHO_BALL=${RHO_BALL:-} RHO_SUB=${RHO_SUB:-} RHO_MEDIUM=${RHO_MEDIUM:-} G=${G:-} NEWMARK_GAMMA=${NEWMARK_GAMMA:-} ALPHA_M=${ALPHA_M:-}"
echo "MATERIAL_MODEL=${MATERIAL_MODEL:-neo_hooke} REG_METHOD=${REG_METHOD:-huhu_lulu} REG_VALUE=${REG_VALUE:-1e-3} N_SHELL=${N_SHELL:-6} WRITE_EVERY=${WRITE_EVERY:-5}"

# ---- fail fast if a needed file is missing from the submit directory ----
missing=""
for f in hertz_3d_dyn.py mesh3d.py methods.py functional.py functional_3d.py plotting.py; do
  [ -f "$SLURM_SUBMIT_DIR/$f" ] || missing="$missing $f"
done
if [ -n "$missing" ]; then
  echo "ERROR: missing in $SLURM_SUBMIT_DIR:$missing"
  echo "Copy them there (the job copies *.py from the submit directory) and resubmit."
  exit 1
fi

MYWORKDIR=$SCRATCH/$SLURM_JOBID
mkdir -p "$MYWORKDIR"
cp "$SLURM_SUBMIT_DIR"/*.py "$MYWORKDIR"/
cd "$MYWORKDIR"
export XDG_CACHE_HOME="$MYWORKDIR/cache"; export HOME="$MYWORKDIR"; mkdir -p "$XDG_CACHE_HOME"

RESULTDIR=$SLURM_SUBMIT_DIR/results3d_dyn_${SWEEP}_$SLURM_JOBID
save_results() {
    mkdir -p "$RESULTDIR"
    # sweep folder written by hertz_3d_dyn.py: hertz3d_dyn_<reg_method>_<material>/
    cp -r hertz3d_dyn_* "$RESULTDIR"/ 2>/dev/null
    cd "$SLURM_SUBMIT_DIR" && rm -rf "$MYWORKDIR"
}
trap save_results EXIT

mpirun -n "$SLURM_NTASKS" python hertz_3d_dyn.py