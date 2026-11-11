#!/bin/bash -l
#SBATCH --job-name=bench3d
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4000
#SBATCH --time=12:00:00
#SBATCH --output=bench_%x_%j.out
#SBATCH --error=bench_%x_%j.err

# Usage:
#   sbatch --export=ALL,CASE=cbox,SWEEP=h       job_bench3d.sh
#   sbatch --export=ALL,CASE=monolith,SWEEP=h   job_bench3d.sh
#   sbatch --export=ALL,CASE=cbox,SWEEP=check   job_bench3d.sh    # 2D-equivalence test (cheap)
#   sbatch --export=ALL,CASE=cbox_dyn,SWEEP=dyn job_bench3d.sh    # DYNAMIC cbox: force on until T_OFF, runs to T_END
#   sbatch --export=ALL,CASE=monolith_dyn,SWEEP=dyn job_bench3d.sh   # DYNAMIC monolith, runs through and beyond cavity contact
#     dynamic options: T_END=50 T_OFF=25 DT=0.25 LOAD_AMP=3e-3 FORCE_PROFILE=halfsine|sin2|triangle RHO=1 NEWMARK_GAMMA=0.5
# CASE  = cbox | monolith | cbox_dyn | monolith_dyn
# SWEEP = check | h | p | hp | dyn
# Optional overrides (comma-separated after ALL,):
#   MATERIAL_MODEL (cbox default neo_hooke, monolith default hartmann_neff), REG_VALUE (1e-5),
#   REG_METHODS (huhu_lulu,huhu), DEPTH, NZ, ZBC (free|plane_strain), CELL_TYPE, MESHES, P_VALUES

. /etc/profile.d/module.sh
module purge
module load anaconda/2023.07-1
eval "$(conda shell.bash hook)"
conda activate $WORK/envs/fenicsx

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg

case "$CASE" in
  cbox|monolith|cbox_dyn|monolith_dyn) ;;
  *) echo "CASE must be cbox, monolith, cbox_dyn or monolith_dyn (got '$CASE')"; exit 1 ;;
esac

# ---- sweep definitions: edit the values here ----
# MESHES uses the 2D definition NxxNy (cbox: Nx%10==0, Ny%5==0; monolith: Nx%20==0, Ny%5==0).
# Elements over the depth: NZ (default 8). p must be >= 2.
# dofs at p=2 (NZ=8):  cbox 20x10 48k | 40x20 186k | 60x30 414k ;  monolith 20x10 44k | 40x20 169k | 60x30 376k
# p=3 is ~3.2x larger.
case "$SWEEP" in
  check) export MESHES="${MESHES:-20x10}";             export P_VALUES="${P_VALUES:-2}";   export ZBC=plane_strain; export NZ=1 ;;
  h)     export MESHES="${MESHES:-20x10,40x20,60x30}"; export P_VALUES="${P_VALUES:-2}" ;;
  p)     export MESHES="${MESHES:-20x10}";             export P_VALUES="${P_VALUES:-2,3}" ;;
  hp)    export MESHES="${MESHES:-20x10,40x20}";       export P_VALUES="${P_VALUES:-2,3}" ;;
  dyn)   export MESHES="${MESHES:-20x10}";             export P_VALUES="${P_VALUES:-2}" ;;
  *) echo "SWEEP must be check, h, p, hp or dyn (got '$SWEEP')"; exit 1 ;;
esac
echo "node: $(hostname)  kernel: $(uname -r)  glibc: $(ldd --version | head -1)"
echo "CASE=$CASE SWEEP=$SWEEP MESHES=$MESHES P_VALUES=$P_VALUES ZBC=${ZBC:-free} NZ=${NZ:-8} DEPTH=${DEPTH:-<2D width>}"
echo "dynamic (unset = script default): T_END=${T_END:-} T_OFF=${T_OFF:-} DT=${DT:-} DT_CONTACT=${DT_CONTACT:-} LOAD_AMP=${LOAD_AMP:-} FORCE_PROFILE=${FORCE_PROFILE:-} RHO=${RHO:-} NEWMARK_GAMMA=${NEWMARK_GAMMA:-} T_AFTER_CONTACT=${T_AFTER_CONTACT:-}"
echo "MATERIAL_MODEL=${MATERIAL_MODEL:-<case default>} REG_VALUE=${REG_VALUE:-1e-5} REG_METHODS=${REG_METHODS:-huhu_lulu,huhu}"

# ---- fail fast if a needed file is missing from the submit directory ----
missing=""
for f in ${CASE}_3d.py driver3d.py functional_3d.py methods.py functional.py plotting.py; do
  [ -f "$SLURM_SUBMIT_DIR/$f" ] || missing="$missing $f"
done
if [ -n "$missing" ]; then
  echo "ERROR: missing in $SLURM_SUBMIT_DIR:$missing"
  echo "Copy them there (the job only copies *.py from the submit directory) and resubmit."
  exit 1
fi

MYWORKDIR=$SCRATCH/$SLURM_JOBID
mkdir -p "$MYWORKDIR"
cp "$SLURM_SUBMIT_DIR"/*.py "$MYWORKDIR"/
cd "$MYWORKDIR"
export XDG_CACHE_HOME="$MYWORKDIR/cache"; export HOME="$MYWORKDIR"; mkdir -p "$XDG_CACHE_HOME"

RESULTDIR=$SLURM_SUBMIT_DIR/results3d_${CASE}_${SWEEP}_$SLURM_JOBID
save_results() {
    mkdir -p "$RESULTDIR"
    cp -r ${CASE}3d_* "$RESULTDIR"/ 2>/dev/null
    cd "$SLURM_SUBMIT_DIR" && rm -rf "$MYWORKDIR"
}
trap save_results EXIT

mpirun -n "$SLURM_NTASKS" python ${CASE}_3d.py