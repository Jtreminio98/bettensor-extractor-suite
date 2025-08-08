# Bettensor Prediction Extractor Suite

This project contains a suite of tools to query the Bittensor network for sports prediction data, store it, and analyze it for consensus signals.

This repository has been pre-configured to use the `recovery_wallet` and `default_hotkey` that were set up during our session.

## Quick Start

The easiest way to run the extractor is to use the provided `run.sh` script.

```bash
./run.sh
```

This script will handle all the necessary steps to query the network and provide an analysis of the collected prediction data.

## Manual Setup and Execution

If you need to run the steps manually or are setting this up in a new environment, follow the steps below.

### 1. Install Dependencies

This project requires Python 3 and several libraries. It is recommended to use a virtual environment.

```bash
# Create and activate a virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate

# Install required libraries
pip install bittensor pydantic
```

### 2. Wallet Configuration

This project is configured to use the following wallet:

*   **Wallet Name:** `recovery_wallet`
*   **Hotkey Name:** `default_hotkey`

This wallet has already been registered on the `finney` network for `netuid 30`. If you need to use a different wallet, you can either:
a) Update the `WALLET_NAME` and `HOTKEY_NAME` variables in the `run.sh` script.
b) Manually edit the `get_predictions_improved.py` script.

### 3. Running the Extractor

You can run the extractor script directly:

```bash
python3 get_predictions_improved.py
```

The script will:
1.  Connect to the Bittensor `finney` network.
2.  Query the top 50 miners on subnet 30 for predictions.
3.  Store the collected predictions in a local SQLite database (`data/bettensor/validator/state/validator.db`).
4.  Analyze the predictions for consensus.
5.  Save the raw results to a timestamped JSON file (e.g., `predictions_extract_20250807_195132.json`).

## Understanding the Output

The script will print its progress to the console. The final output will be an analysis of the consensus signals found in the data, highlighting any strong agreements among the miners.

All collected data is stored persistently in the `validator.db` database file, so you can re-run the analysis without querying the network again if needed.
