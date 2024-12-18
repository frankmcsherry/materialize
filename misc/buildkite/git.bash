#!/usr/bin/env bash

# Copyright Materialize, Inc. and contributors. All rights reserved.
#
# Use of this software is governed by the Business Source License
# included in the LICENSE file at the root of this repository.
#
# As of the Change Date specified in that file, in accordance with
# the Business Source License, use of this software will be governed
# by the Apache License, Version 2.0.

set -euo pipefail

. misc/shlib/shlib.bash

export BUILDKITE_REPO_REF="${BUILDKITE_REPO_REF:-origin}"
export BUILDKITE_PULL_REQUEST_BASE_BRANCH="${BUILDKITE_PULL_REQUEST_BASE_BRANCH:-main}"

configure_git_user_if_in_buildkite() {
  if [[ "${BUILDKITE:-}" == "true" ]]; then
    ci_collapsed_heading "Configure git"
    run git config --global user.email "buildkite@materialize.com"
    run git config --global user.name "Buildkite"
  fi
}

fetch_pr_target_branch() {
  ci_collapsed_heading "Fetch target branch"
  run git fetch "$BUILDKITE_REPO_REF" "$BUILDKITE_PULL_REQUEST_BASE_BRANCH"
}

merge_pr_target_branch() {
  configure_git_user_if_in_buildkite

  ci_collapsed_heading "Merge target branch"
  run git merge "$BUILDKITE_REPO_REF"/"$BUILDKITE_PULL_REQUEST_BASE_BRANCH" --message "Merge"
}

get_common_ancestor_commit_of_pr_and_target() {
  run git merge-base HEAD "$BUILDKITE_REPO_REF"/"$BUILDKITE_PULL_REQUEST_BASE_BRANCH"
}
