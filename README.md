# Bettensor Prediction Extractor Suite

This project contains a suite of tools to query the Bittensor network for sports prediction data, store it, and analyze it for consensus signals.

This repository has been pre-configured to use the `recovery_wallet` and `default_hotkey` that were set up during our session.

## Quick Start

The easiest way to run the extractor is to use the provided `run.sh` script. This will query the network for all available predictions for the current day.

```bash
./run.sh
```

## Querying for Specific Games

You can now query for specific games by providing command-line arguments to the `get_predictions_improved.py` script.

**Available Arguments:**
*   `--sport`: The sport to filter by (e.g., `baseball`, `football`).
*   `--league`: The league to filter by (e.g., `MLB`, `NFL`).
*   `--date`: The date of the game in `YYYY-MM-DD` format.

**Example:**

To find predictions for an MLB game on August 8, 2025, you would run:

```bash
python3 get_predictions_improved.py --league MLB --date 2025-08-08
```

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

This project is hardcoded to use the following wallet:

*   **Wallet Name:** `recovery_wallet`
*   **Hotkey Name:** `default_hotkey`

This wallet has already been registered on the `finney` network for `netuid 30`.

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