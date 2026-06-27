#!/bin/bash

set -euo pipefail

exclude_specs=(':!:*.png' ':!:*.jpg' ':!:*.zip')

mapfile -d '' files < <(git ls-files -z -- . "${exclude_specs[@]}")

echo "Total"
cloc "${files[@]}"

mapfile -t subdirs < <(
    printf '%s\0' "${files[@]}" |
        awk -v RS='\0' -F/ 'NF > 1 { print $1 }' |
        sort -u
)

for subdir in "${subdirs[@]}"; do
    mapfile -d '' subdir_files < <(git ls-files -z -- "$subdir" "${exclude_specs[@]}")

    if ((${#subdir_files[@]} == 0)); then
        continue
    fi

    cloc_output=$(cloc "${subdir_files[@]}")

    if [[ -z "$cloc_output" ]]; then
        continue
    fi

    printf '\n%s\n%s\n' "$subdir" "$cloc_output"
done
