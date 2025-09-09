#!/bin/sh
#PBS -l select=1
#PBS -P <PROJECT_NAME>

PROJ_ROOT=/groups/<PROJECT_NAME>/<WORKSPACE_NAME>/RoboManipBaselines
MISC_DIR="${PROJ_ROOT}/robo_manip_baselines/misc"
EVAL_SCRIPT="${MISC_DIR}/AutoEval.py"
RESULT_DATA_DIR="${MISC_DIR}/result"
TARGET_DIR=/groups/<PROJECT_NAME>/<WORKSPACE_NAME>/RoboManipBaselines/robo_manip_baselines/misc/<TASK_DIR>
DATA_ROOT=/groups/<PROJECT_NAME>/data/RoboManipBaselines
DATA_LOC="${DATA_ROOT}/${DATA_TAG}/"
EVAL_COMMIT_DIR=<COMMIT_DIR>/RoboManipBaselines-Eval/

source ~/miniconda3/bin/activate
conda activate rmb
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=$CONDA_PREFIX/lib/libGLEW.so

PROJ_ROOT="${PROJ_ROOT:?Please set the PROJ_ROOT environment variable.}"
EVAL_SCRIPT="${EVAL_SCRIPT:?Please set the EVAL_SCRIPT environment variable.}"
DATA_TAG="${DATA_TAG:?Please set the DATA_TAG environment variable.}"
DATA_ROOT="${DATA_ROOT:?Please set the DATA_ROOT environment variable.}"
DATA_LOC="${DATA_LOC:?Please set the DATA_LOC environment variable.}"
POLICY="${POLICY:?Please set the POLICY environment variable.}"
ENV="${ENV:?Please set the ENV environment variable.}"
SEEDS="${SEEDS:?Please set the SEEDS environment variable.}"
JQUEUE="${JQUEUE:?Please set the JQUEUE environment variable.}"

ARGS_FILE_TRAIN="${ARGS_FILE_TRAIN:-}"
SETTING_REMARK="${SETTING_REMARK:-}"

if [ -n "${ARGS_FILE_TRAIN}" ]; then
  ARGS_FILE_TRAIN_PATH="${MISC_DIR}/${ARGS_FILE_TRAIN}"
else
  ARGS_FILE_TRAIN_PATH=""
fi

(
  python3 -u "${EVAL_SCRIPT}" \
      "${POLICY}" "${ENV}" \
      --upgrade_pip_setuptools \
      --world_idx_list 0 1 2 3 4 5 \
      --target_dir "${TARGET_DIR}" \
      --seeds ${SEEDS} \
      --jqueue "${JQUEUE}" \
      --dataset_location "${DATA_LOC}" \
      --args_file_train "${ARGS_FILE_TRAIN_PATH}" \
      --setting_remark "${SETTING_REMARK}"
) && (
  python3 -u "${EVAL_SCRIPT}" \
      --n_eval_trials_expected 30 \
      --result_data_dir "${RESULT_DATA_DIR}" \
      --eval_commit_dir "${EVAL_COMMIT_DIR}" \
      --jqueue "DO_REPORT_EVAL_MD" \
      --do_report_eval_md
)
