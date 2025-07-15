#!/bin/bash
# start.sh

# Take the content from the PRIVATE_KEY_PEM environment variable
# and write it to the app.pem file inside the container.
echo "$PRIVATE_KEY_PEM" > /app/app.pem

# Now, execute the original command to start the FastAPI server.
uvicorn main:app --host 0.0.0.0 --port 8000