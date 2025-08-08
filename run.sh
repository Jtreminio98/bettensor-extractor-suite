#!/bin/bash
#
# Easy-to-use script to run the Bettensor Prediction Extractor.
#

# --- Script Body ---
cd "$(dirname "$0")" || exit

echo "🚀 Starting Bettensor Prediction Extractor..."
echo ""

# Run the Python extractor script
python3 get_predictions_improved.py

echo ""
echo "✅ Extractor run finished."