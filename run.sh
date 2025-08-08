#!/bin/bash
#
# Easy-to-use script to run the Bettensor Prediction Extractor.
#
# This script automatically uses the correct wallet and hotkey that we
# configured and registered on the network.
#

# --- Configuration ---
WALLET_NAME="recovery_wallet"
HOTKEY_NAME="default_hotkey"
SCRIPT_FILE="get_predictions_improved.py"

# --- Script Body ---
echo "🚀 Starting Bettensor Prediction Extractor..."
echo "   Wallet: $WALLET_NAME"
echo "   Hotkey: $HOTKEY_NAME"
echo ""

# Modify the script to use the correct wallet configuration
# This is a simple way to override the config without complex file parsing
sed -i "s/wallet_name = 'default'/wallet_name = '$WALLET_NAME'/" "$SCRIPT_FILE"
sed -i "s/hotkey_name = 'default'/hotkey_name = '$HOTKEY_NAME'/" "$SCRIPT_FILE"

# Run the Python extractor script
python3 "$SCRIPT_FILE"

# Revert the script back to its original state for cleanliness
# This makes sure the git history stays clean if you commit changes.
sed -i "s/wallet_name = '$WALLET_NAME'/wallet_name = 'default'/" "$SCRIPT_FILE"
sed -i "s/hotkey_name = '$HOTKEY_NAME'/hotkey_name = 'default'/" "$SCRIPT_FILE"

echo ""
echo "✅ Extractor run finished."
