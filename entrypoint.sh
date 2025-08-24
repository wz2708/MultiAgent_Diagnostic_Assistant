#!/usr/bin/env bash
set -euo pipefail

python --version
pip --version

exec streamlit run app/web_demo.py --server.port ${STREAMLIT_SERVER_PORT:-7860} --server.address ${STREAMLIT_SERVER_ADDRESS:-0.0.0.0}
