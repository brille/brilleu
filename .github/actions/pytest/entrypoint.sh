#!/bin/bash -l

# Exit immediately on errors
set -e

function echo_run {
  echo "$" "$@"
  "$@"
}

if [[ ! -z "${INPUT_PYTHON_PACKAGES}" ]]; then
  echo ::group::Install extra required Python packages
  for pkg in $INPUT_PYTHON_PACKAGES; do
    echo_run python -m pip install $pkg
  done
  echo ::endgroup::
fi
if [[ ! -z "${INPUT_PYTHON_ONLY_PACKAGES}" ]]; then
  echo ::group::Install extra required Python-only packages
  for pkg in $INPUT_PYTHON_ONLY_PACKAGES; do
    echo_run python -m pip install --install-option="--python-only" $pkg
  done
  echo ::endgroup::
fi

if [[ ! -z "${INPUT_TEST_DIR}" ]]; then
  echo ::group::Move to test directory
  echo_run cd "${GITHUB_WORKSPACE}/${INPUT_TEST_DIR}"
  echo ::endgroup::
fi

echo ::group::Run Pytest
echo_run python -m pytest ${INPUT_ARGUMENTS}
echo ::endgroup::
