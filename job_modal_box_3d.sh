#!/bin/bash -l
#SBATCH --job-name=modalbox3d
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=6000
#SBATCH --time=12:00:00
#SBATCH --output=modal_%x_%j.out
#SBATCH --error=modal_%x_%j.err

# Modal (free-vibration) analysis of the clamped box (modal_box_3d.py):
#   stage 1: linear buckling (P_cr, delta_cr)           -> defines the pre-load levels
#   stage 2: modal analysis about pre-compressed states -> frequencies.csv, modes_f<frac>.bp
#   stage 3: (NONLINEAR=1) displacement-controlled run with imperfection
#
# Usage:
#   sbatch --export=ALL,SWEEP=test                              job_modal_box3d.sh   # 20x10, LOAD_FRACS=0:0.5:0.9
#   sbatch --export=ALL,SWEEP=test,LOAD_FRACS=0                 job_modal_box3d.sh   # unloaded modes only
#   sbatch --export=ALL,SWEEP=h,NMODES=12,RHO=1000              job_modal_box3d.sh
# SWEEP = test | h | p | hp
#
# Optional overrides (comma-separated after ALL,); only set when given:
#   RHO (1.0), RHO_MED_RATIO (1e-8), NMODES (10), LOAD_FRACS ("0:0.5:0.9", use ":" because sbatch --export splits on commas; fractions of the linear critical
#   end shortening), MODAL_SHIFT (0.0), NEV (6 buckling modes), NONLINEAR (0), IMP_FRAC, DISP_FACTOR, DISP_MAX, FORCE,
#   MU (10), K (20), MATERIAL_MODEL, REG_VALUE (1e-5), REG_METHODS, MESHES, P_VALUES (>=2), NZ (8), DEPTH (0.5),
#   CELL_TYPE, LAYER, LAYER_Z, OUTER_BC, NEWTON_MAXIT, WRITE_EVERY

. /etc/profile.d/module.sh
module purge
module load anaconda/2023.07-1
eval "$(conda shell.bash hook)"
conda activate $WORK/envs/fenicsx

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg

CODE=modal_box_3d.py
PREFIX=modalbox3d          # = case name passed to sweep_main in modal_box_3d.py

# ---- sweep definitions: edit the values here ----
# The eigenvalue stage needs several MUMPS factorisations, hence more memory per core than the benchmark job.
case "${SWEEP:-test}" in
  test) export MESHES="${MESHES:-20x10}";             export P_VALUES="${P_VALUES:-2}" ;;
  h)    export MESHES="${MESHES:-20x10,40x20,60x30}"; export P_VALUES="${P_VALUES:-2}" ;;
  p)    export MESHES="${MESHES:-20x10}";             export P_VALUES="${P_VALUES:-2,3}" ;;
  hp)   export MESHES="${MESHES:-20x10,40x20}";       export P_VALUES="${P_VALUES:-2,3}" ;;
  *) echo "SWEEP must be test, h, p or hp (got '${SWEEP}')"; exit 1 ;;
esac
SWEEP="${SWEEP:-test}"

# the buckling problem is fully 3D
if [ "${ZBC:-free}" != "free" ]; then
  echo "modal_box_3d.py needs ZBC=free (got '$ZBC')"; exit 1
fi
export ZBC=free

echo "node: $(hostname)  kernel: $(uname -r)  glibc: $(ldd --version | head -1)"
echo "SWEEP=$SWEEP MESHES=$MESHES P_VALUES=$P_VALUES ZBC=$ZBC NZ=${NZ:-8} DEPTH=${DEPTH:-0.5}"
echo "MATERIAL_MODEL=${MATERIAL_MODEL:-hartmann_neff} MU=${MU:-10} K=${K:-20} REG_VALUE=${REG_VALUE:-1e-5} REG_METHODS=${REG_METHODS:-huhu_lulu,huhu}"
echo "RHO=${RHO:-1.0} RHO_MED_RATIO=${RHO_MED_RATIO:-1e-8} NMODES=${NMODES:-10} LOAD_FRACS=${LOAD_FRACS:-0:0.5:0.9} MODAL_SHIFT=${MODAL_SHIFT:-0.0}"
echo "NEV=${NEV:-6} NONLINEAR=${NONLINEAR:-0} IMP_FRAC=${IMP_FRAC:-1e-3} DISP_FACTOR=${DISP_FACTOR:-2.0} DISP_MAX=${DISP_MAX:-<auto>} FORCE=${FORCE:-<auto: P_cr>}"
echo "LAYER=${LAYER:-1} LAYER_Z=${LAYER_Z:-1} OUTER_BC=${OUTER_BC:-free}"

# fail early if SLEPc is missing in the environment
python -c "import slepc4py, dolfinx; print('dolfinx', dolfinx.__version__, '| slepc4py', slepc4py.__version__)" \
  || { echo "slepc4py/dolfinx not importable in the fenicsx env (conda install -c conda-forge slepc4py)"; exit 1; }

MYWORKDIR=$SCRATCH/$SLURM_JOBID
mkdir -p "$MYWORKDIR"
cp "$SLURM_SUBMIT_DIR"/*.py "$MYWORKDIR"/
cd "$MYWORKDIR"
export XDG_CACHE_HOME="$MYWORKDIR/cache"; export HOME="$MYWORKDIR"; mkdir -p "$XDG_CACHE_HOME"

RESULTDIR=$SLURM_SUBMIT_DIR/results3d_modalbox_${SWEEP}_$SLURM_JOBID
save_results() {
    mkdir -p "$RESULTDIR"
    cp -r ${PREFIX}_* "$RESULTDIR"/ 2>/dev/null
    cd "$SLURM_SUBMIT_DIR" && rm -rf "$MYWORKDIR"
}
trap save_results EXIT

mpirun -n "$SLURM_NTASKS" python $CODE