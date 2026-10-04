#!/usr/bin/env bash
# The layering gate for G5 (AGENTS.md rule 6): the serving layer must depend on the
# engine only through the neutral conversation seam. It may include its own headers,
# `infrastructure/session/**`, `infrastructure/text/text_generation.hpp`,
# `infrastructure/json.hpp`, and the vendored HTTP library — nothing else. A model,
# weight-format or kernel header, or any HIP reference, is a violation.
set -euo pipefail

status=0

while IFS= read -r file; do
    while IFS= read -r line; do
        include=$(sed -n 's/^[[:space:]]*#include[[:space:]]*"\([^"]*\)".*/\1/p' <<<"$line")
        [ -z "$include" ] && continue
        case "$include" in
            infrastructure/session/*|infrastructure/text/text_generation.hpp|infrastructure/json.hpp|server/*|httplib.h)
                ;;
            *)
                echo "layering: forbidden include in $file: $include" >&2
                status=1
                ;;
        esac
    done < "$file"

    if grep -nE 'amdhip|hip[A-Z_]' "$file" >/dev/null 2>&1; then
        echo "layering: forbidden HIP reference in $file" >&2
        grep -nE 'amdhip|hip[A-Z_]' "$file" >&2 || true
        status=1
    fi
done < <(find server -type f \( -name '*.hpp' -o -name '*.cpp' \) | sort)

if [ "$status" -eq 0 ]; then
    echo "layering: clean"
fi
exit "$status"
