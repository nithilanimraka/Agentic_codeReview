#!/bin/sh
set -e

echo "--- DIAGNOSTICS START ---"
echo ""
echo "--- Current Directory ---"
pwd
echo ""
echo "--- Files Present In /app ---"
ls -laR /app
echo ""
echo "--- Environment Variables ---"
env
echo ""
echo "--- ATTEMPTING TO START APP ---"

# Now, execute the original command from your Dockerfile
exec uvicorn main:app --host 0.0.0.0 --port 8000