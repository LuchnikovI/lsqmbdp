#!/usr/bin/env bash

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

. "${script_dir}/utils.sh"

# ---------------------------------------------------------------------------

log INFO "Running typechecker..."

if . "${script_dir}/runner.sh" --typecheck; then
    log INFO Type checking OK
else
    log ERROR Type checking failed
    exit 1
fi

log INFO "Running tests..."

if . "${script_dir}/runner.sh" --test; then
    log INFO Testing OK
else
    log ERROR Testing failed
    exit 1
fi

log INFO "Running linter..."

if . "${script_dir}/runner.sh" --lint; then
    log INFO Linting OK
else
    log WARNING Linting failed
fi

log INFO "Done!"