#!/bin/bash
export PYTHONPATH=$(pwd)
export DETECTION_ENGINE=model
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
#!/bin/bash
export PYTHONPATH=$(pwd)
export DETECTION_ENGINE=model
python3 -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
