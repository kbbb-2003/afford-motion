#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
SEED="${SEED:-2023}"
RATIO="${RATIO:-0.2}"
ADM_STEPS="${ADM_STEPS:-20000}"
AMDM_STEPS="${AMDM_STEPS:-20000}"
ADM_SAVE_EVERY="${ADM_SAVE_EVERY:-5000}"
AMDM_SAVE_EVERY="${AMDM_SAVE_EVERY:-5000}"
BATCH_ADM="${BATCH_ADM:-64}"
BATCH_AMDM="${BATCH_AMDM:-32}"
RUN_BASELINE="${RUN_BASELINE:-1}"
RUN_IMPROVED="${RUN_IMPROVED:-1}"
H3D_DIR="${H3D_DIR:-${ROOT_DIR}/data/H3D}"
TRAIN_BACKUP="${TRAIN_BACKUP:-train_full.txt}"
TRAIN_SUBSET="${TRAIN_SUBSET:-train_20.txt}"
TAG="${TAG:-SUB20_S${SEED}_A${ADM_STEPS}_M${AMDM_STEPS}}"

BASELINE_ADM_EXP="BASELINE_${TAG}_ADM_H3D"
BASELINE_AMDM_EXP="BASELINE_${TAG}_AMDM_H3D"
IMPROVED_ADM_EXP="IMPROVED_${TAG}_ADM_H3D"
IMPROVED_AMDM_EXP="IMPROVED_${TAG}_AMDM_H3D"

TRAIN_FILE="${H3D_DIR}/train.txt"
BACKUP_FILE="${H3D_DIR}/${TRAIN_BACKUP}"
SUBSET_FILE="${H3D_DIR}/${TRAIN_SUBSET}"

restore_train_split() {
  if [[ -f "${BACKUP_FILE}" ]]; then
    cp "${BACKUP_FILE}" "${TRAIN_FILE}"
    echo "[restore] Restored original train split from ${BACKUP_FILE}"
  fi
}

latest_eval_dir() {
  local exp_dir="$1"
  local latest
  latest="$(ls -dt "${exp_dir}"/eval/test-* 2>/dev/null | head -n 1 || true)"
  if [[ -z "${latest}" ]]; then
    echo "[error] Cannot find eval/test-* under ${exp_dir}" >&2
    exit 1
  fi
  printf "%s" "${latest}"
}

trap restore_train_split EXIT

if [[ ! -d "${H3D_DIR}" ]]; then
  echo "[error] H3D directory not found: ${H3D_DIR}" >&2
  exit 1
fi

if [[ ! -f "${TRAIN_FILE}" ]]; then
  echo "[error] train.txt not found: ${TRAIN_FILE}" >&2
  exit 1
fi

echo "[step] Generating and activating subset split"
"${PYTHON_BIN}" "${ROOT_DIR}/make_h3d_train20.py" \
  --h3d-dir "${H3D_DIR}" \
  --backup-name "${TRAIN_BACKUP}" \
  --output-name "${TRAIN_SUBSET}" \
  --ratio "${RATIO}" \
  --seed "${SEED}" \
  --activate \
  --force

echo "[info] Current subset split: ${SUBSET_FILE}"
echo "[info] Baseline run enabled: ${RUN_BASELINE}"
echo "[info] Improved run enabled: ${RUN_IMPROVED}"

if [[ "${RUN_BASELINE}" == "1" ]]; then
  echo "[step] Baseline ADM training"
  "${PYTHON_BIN}" train.py hydra/job_logging=none hydra/hydra_logging=none \
    exp_name="${BASELINE_ADM_EXP}" \
    output_dir=outputs \
    platform=TensorBoard \
    diffusion.steps=500 \
    task=text_to_motion_contact_gen \
    task.dataset.sigma=0.8 \
    task.dataset.temporal_affordance=false \
    task.train.batch_size="${BATCH_ADM}" \
    task.train.max_steps="${ADM_STEPS}" \
    task.train.save_every_step="${ADM_SAVE_EVERY}" \
    model=cdm \
    model.arch=Perceiver \
    model.scene_model.use_scene_model=False \
    model.text_model.max_length=20

  echo "[step] Baseline ADM test / affordance generation"
  "${PYTHON_BIN}" test.py hydra/job_logging=none hydra/hydra_logging=none \
    exp_dir="outputs/${BASELINE_ADM_EXP}" \
    seed="${SEED}" \
    output_dir=outputs \
    diffusion.steps=500 \
    task=text_to_motion_contact_gen \
    task.dataset.sigma=0.8 \
    task.dataset.temporal_affordance=false \
    task.evaluator.k_samples=0 \
    task.evaluator.eval_nbatch=32 \
    task.evaluator.num_k_samples=128 \
    model=cdm \
    model.arch=Perceiver \
    model.scene_model.use_scene_model=False \
    model.text_model.max_length=20

  BASELINE_AFFORD_DIR="$(latest_eval_dir "${ROOT_DIR}/outputs/${BASELINE_ADM_EXP}")"
  echo "[info] Baseline affordance dir: ${BASELINE_AFFORD_DIR}"

  echo "[step] Baseline AMDM training"
  "${PYTHON_BIN}" train.py hydra/job_logging=none hydra/hydra_logging=none \
    exp_name="${BASELINE_AMDM_EXP}" \
    output_dir=outputs \
    platform=TensorBoard \
    diffusion.steps=1000 \
    task=text_to_motion_contact_motion_gen \
    task.dataset.sigma=0.8 \
    task.dataset.temporal_affordance=false \
    task.dataset.mix_train_ratio=0.0 \
    task.train.batch_size="${BATCH_AMDM}" \
    task.train.max_steps="${AMDM_STEPS}" \
    task.train.save_every_step="${AMDM_SAVE_EVERY}" \
    task.dataset.train_transforms="['RandomEraseLang','RandomEraseContact','NumpyToTensor']" \
    model=cmdm \
    model.arch=trans_enc \
    model.data_repr=h3d \
    model.text_model.max_length=20 \
    model.geometry_loss.enable=false

  echo "[step] Baseline AMDM test / motion generation"
  "${PYTHON_BIN}" test.py hydra/job_logging=none hydra/hydra_logging=none \
    exp_dir="outputs/${BASELINE_AMDM_EXP}" \
    seed="${SEED}" \
    output_dir=outputs \
    diffusion.steps=1000 \
    task=text_to_motion_contact_motion_gen \
    task.dataset.sigma=0.8 \
    task.dataset.temporal_affordance=false \
    task.test.contact_folder="${BASELINE_AFFORD_DIR}" \
    task.evaluator.k_samples=0 \
    task.evaluator.eval_nbatch=32 \
    task.evaluator.num_k_samples=128 \
    model=cmdm \
    model.arch=trans_enc \
    model.data_repr=h3d \
    model.text_model.max_length=20
fi

if [[ "${RUN_IMPROVED}" == "1" ]]; then
  echo "[step] Improved ADM training"
  "${PYTHON_BIN}" train.py hydra/job_logging=none hydra/hydra_logging=none \
    exp_name="${IMPROVED_ADM_EXP}" \
    output_dir=outputs \
    platform=TensorBoard \
    diffusion.steps=500 \
    task=text_to_motion_contact_gen \
    task.dataset.sigma=0.8 \
    task.dataset.temporal_affordance=true \
    task.dataset.num_phases=4 \
    task.train.batch_size="${BATCH_ADM}" \
    task.train.max_steps="${ADM_STEPS}" \
    task.train.save_every_step="${ADM_SAVE_EVERY}" \
    model=cdm \
    model.arch=Perceiver \
    model.scene_model.use_scene_model=False \
    model.text_model.max_length=20

  echo "[step] Improved ADM test / affordance generation"
  "${PYTHON_BIN}" test.py hydra/job_logging=none hydra/hydra_logging=none \
    exp_dir="outputs/${IMPROVED_ADM_EXP}" \
    seed="${SEED}" \
    output_dir=outputs \
    diffusion.steps=500 \
    task=text_to_motion_contact_gen \
    task.dataset.sigma=0.8 \
    task.dataset.temporal_affordance=true \
    task.dataset.num_phases=4 \
    task.evaluator.k_samples=0 \
    task.evaluator.eval_nbatch=32 \
    task.evaluator.num_k_samples=128 \
    model=cdm \
    model.arch=Perceiver \
    model.scene_model.use_scene_model=False \
    model.text_model.max_length=20

  IMPROVED_AFFORD_DIR="$(latest_eval_dir "${ROOT_DIR}/outputs/${IMPROVED_ADM_EXP}")"
  echo "[info] Improved affordance dir: ${IMPROVED_AFFORD_DIR}"

  echo "[step] Improved AMDM training"
  "${PYTHON_BIN}" train.py hydra/job_logging=none hydra/hydra_logging=none \
    exp_name="${IMPROVED_AMDM_EXP}" \
    output_dir=outputs \
    platform=TensorBoard \
    diffusion.steps=1000 \
    task=text_to_motion_contact_motion_gen \
    task.dataset.sigma=0.8 \
    task.dataset.temporal_affordance=true \
    task.dataset.num_phases=4 \
    task.dataset.mix_train_ratio=0.0 \
    task.train.batch_size="${BATCH_AMDM}" \
    task.train.max_steps="${AMDM_STEPS}" \
    task.train.save_every_step="${AMDM_SAVE_EVERY}" \
    task.dataset.train_transforms="['RandomEraseLang','RandomEraseContact','NumpyToTensor']" \
    model=cmdm \
    model.arch=trans_enc \
    model.data_repr=h3d \
    model.text_model.max_length=20 \
    model.geometry_loss.enable=true

  echo "[step] Improved AMDM test / motion generation"
  "${PYTHON_BIN}" test.py hydra/job_logging=none hydra/hydra_logging=none \
    exp_dir="outputs/${IMPROVED_AMDM_EXP}" \
    seed="${SEED}" \
    output_dir=outputs \
    diffusion.steps=1000 \
    task=text_to_motion_contact_motion_gen \
    task.dataset.sigma=0.8 \
    task.dataset.temporal_affordance=true \
    task.dataset.num_phases=4 \
    task.test.contact_folder="${IMPROVED_AFFORD_DIR}" \
    task.evaluator.k_samples=0 \
    task.evaluator.eval_nbatch=32 \
    task.evaluator.num_k_samples=128 \
    model=cmdm \
    model.arch=trans_enc \
    model.data_repr=h3d \
    model.text_model.max_length=20
fi

echo "[done] H3D subset experiment finished."
echo "[done] Baseline ADM exp: ${BASELINE_ADM_EXP}"
echo "[done] Baseline AMDM exp: ${BASELINE_AMDM_EXP}"
echo "[done] Improved ADM exp: ${IMPROVED_ADM_EXP}"
echo "[done] Improved AMDM exp: ${IMPROVED_AMDM_EXP}"
