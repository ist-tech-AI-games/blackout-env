#!/bin/sh
# Restart the policy server if it dies mid-match (e.g. a crash inside onnxruntime).
# Exit 0 (close requested) and 2 (submission failed to load) end the container.
while true; do
    python3 /app/policy_server.py "$@"
    code=$?
    if [ "$code" -eq 0 ] || [ "$code" -eq 2 ]; then
        exit "$code"
    fi
    echo "[policy_entrypoint] policy_server exited with $code; restarting"
    sleep 1
done
