#!/usr/bin/env python3
"""
Improved Miner Prediction Query Script
Queries Bettensor miners for their sports betting predictions with better error handling
"""

import asyncio
import sqlite3
import os
import sys
import json
import traceback
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import bittensor as bt
from pydantic import BaseModel, Field
import time

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Define multiple synapse types to try different protocols
class GameData(bt.Synapse):
    """Main synapse for requesting game data and predictions from miners"""
    
    # Request fields
    sport: Optional[str] = Field(default=None, description="Filter by sport type")
    league: Optional[str] = Field(default=None, description="Filter by league")
    game_date: Optional[str] = Field(default=None, description="Filter by date (YYYY-MM-DD)")
    
    # Response fields
    prediction_dict: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Dictionary of predictions keyed by prediction_id"
    )
    
    gamedata_dict: Optional[Dict[str, Dict]] = Field(
        default_factory=dict,
        description="Dictionary of game data keyed by game_id"
    )
    
    def deserialize(self) -> Dict[str, Any]:
        """Deserialize the synapse response"""
        return {
            'predictions': self.prediction_dict or {},
            'games': self.gamedata_dict or {}
        }

class PredictionSynapse(bt.Synapse):
    """Alternative synapse for requesting predictions directly"""
    
    # Request fields
    request_type: str = Field(default="get_predictions", description="Type of request")
    filters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Filters for predictions")
    
    # Response fields
    predictions: Optional[List[Dict[str, Any]]] = Field(
        default_factory=list,
        description="List of predictions"
    )
    
    success: bool = Field(default=False, description="Whether the request was successful")
    message: Optional[str] = Field(default=None, description="Response message")

class BettensorSynapse(bt.Synapse):
    """Bettensor-specific synapse"""
    
    # Request fields
    synapse_type: str = Field(default="prediction_request", description="Type of synapse")
    params: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Request parameters")
    
    # Response fields
    data: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Response data")
    status: str = Field(default="pending", description="Response status")

def connect_to_database():
    """Connect to the validator database"""
    try:
        from config.wallet_config import DATABASE_CONFIG
        db_path = DATABASE_CONFIG['db_path']
    except ImportError:
        db_path = os.path.join("data", "bettensor", "validator", "state", "validator.db")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    if not os.path.exists(db_path):
        print(f"⚠️  Database not found at {db_path}")
        print("   Creating new database...")
        
        # Create database with basic schema
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                prediction_id TEXT PRIMARY KEY,
                game_id TEXT,
                miner_uid INTEGER,
                miner_hotkey TEXT,
                prediction_date TIMESTAMP,
                predicted_outcome TEXT,
                predicted_odds REAL,
                wager REAL,
                team_a TEXT,
                team_b TEXT,
                team_a_odds REAL,
                team_b_odds REAL,
                tie_odds REAL,
                sport TEXT,
                league TEXT,
                outcome TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        print(f"✅ Database created at {db_path}")
        return conn
    
    return sqlite3.connect(db_path)

def store_prediction(conn, prediction_data):
    """Store a prediction in the database"""
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT OR REPLACE INTO predictions 
        (prediction_id, game_id, miner_uid, miner_hotkey, prediction_date,
         predicted_outcome, predicted_odds, wager, team_a, team_b,
         team_a_odds, team_b_odds, tie_odds, sport, league)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        prediction_data['prediction_id'],
        prediction_data['game_id'],
        prediction_data['miner_uid'],
        prediction_data['miner_hotkey'],
        prediction_data['prediction_date'],
        prediction_data['predicted_outcome'],
        prediction_data['predicted_odds'],
        prediction_data['wager'],
        prediction_data['team_a'],
        prediction_data['team_b'],
        prediction_data['team_a_odds'],
        prediction_data['team_b_odds'],
        prediction_data['tie_odds'],
        prediction_data.get('sport', 'unknown'),
        prediction_data.get('league', 'unknown')
    ))
    
    conn.commit()

async def query_miner_with_different_protocols(dendrite, axon, miner_info):
    """Try multiple protocols to query a miner"""
    protocols = [
        # Protocol 1: GameData synapse (original)
        GameData(
            sport=None,
            league=None,
            game_date=datetime.now().strftime("%Y-%m-%d")
        ),
        
        # Protocol 2: PredictionSynapse
        PredictionSynapse(
            request_type="get_predictions",
            filters={"active": True, "date": datetime.now().strftime("%Y-%m-%d")}
        ),
        
        # Protocol 3: BettensorSynapse
        BettensorSynapse(
            synapse_type="prediction_request",
            params={"sport": "all", "league": "all"}
        ),
        
        # Protocol 4: Simple synapse
        bt.Synapse()
    ]
    
    for i, synapse in enumerate(protocols):
        try:
            print(f"    🔄 Trying protocol {i+1}/4 for miner {miner_info['uid']}...")
            
            response = await dendrite(
                axons=[axon],
                synapse=synapse,
                deserialize=True,
                timeout=10
            )
            
            if response and len(response) > 0 and response[0] is not None:
                result = response[0]
                
                # Check if we got any data
                if hasattr(result, 'prediction_dict') and result.prediction_dict:
                    return result, f"protocol_{i+1}"
                elif hasattr(result, 'predictions') and result.predictions:
                    return result, f"protocol_{i+1}"
                elif hasattr(result, 'data') and result.data:
                    return result, f"protocol_{i+1}"
                elif hasattr(result, '__dict__') and any(result.__dict__.values()):
                    return result, f"protocol_{i+1}"
                    
        except Exception as e:
            print(f"    ❌ Protocol {i+1} failed: {e}")
            continue
    
    return None, None

async def query_bittensor_miners_improved():
    """Improved version of miner querying with multiple protocols"""
    print("🔧 Initializing Bittensor connection...")
    
    try:
        # Load wallet configuration
        try:
            from config.wallet_config import WALLET_CONFIG, NETWORK_CONFIG
            wallet_name = 'default'
            hotkey_name = WALLET_CONFIG.get('hotkey_name', 'default')
            network = NETWORK_CONFIG['network']
            netuid = NETWORK_CONFIG['netuid']
        except ImportError:
            wallet_name = 'default'
            hotkey_name = 'default'
            network = 'finney'
            netuid = 30
        
        # Initialize wallet
        wallet = bt.wallet(name=wallet_name, hotkey=hotkey_name)
        print(f"🔑 Using wallet: {wallet.name}/{wallet.hotkey_str}")
        
        # Check if wallet exists
        if not wallet.coldkey_file.exists_on_device():
            print("❌ Coldkey not found. Please create a Bittensor wallet first:")
            print("   btcli wallet create --wallet.name default")
            return []
        
        if not wallet.hotkey_file.exists_on_device():
            print("❌ Hotkey not found. Please create a hotkey first:")
            print("   btcli wallet create --wallet.name default --wallet.hotkey default")
            return []
        
        # Initialize subtensor
        subtensor = bt.subtensor(network=network)
        print(f"📡 Connected to network: {subtensor.network}")
        
        # Initialize dendrite and metagraph
        dendrite = bt.dendrite(wallet=wallet)
        metagraph = subtensor.metagraph(netuid=netuid)
        
        print(f"🌐 Found {len(metagraph.hotkeys)} miners in subnet {netuid}")
        
        # Check if wallet is registered
        if wallet.hotkey.ss58_address not in metagraph.hotkeys:
            print("⚠️  Wallet not registered on subnet. Exiting.")
            return []
        
        # Get active miners
        active_miners = []
        for i, neuron in enumerate(metagraph.neurons):
            if neuron.axon_info.ip != "0.0.0.0" and neuron.axon_info.port != 0:
                active_miners.append({
                    'uid': i,
                    'hotkey': neuron.hotkey,
                    'ip': neuron.axon_info.ip,
                    'port': neuron.axon_info.port,
                    'axon': metagraph.axons[i],
                    'stake': neuron.stake.tao,
                    'trust': neuron.trust.item() if hasattr(neuron.trust, 'item') else 0
                })
        
        print(f"✨ Found {len(active_miners)} active miners")
        
        if len(active_miners) == 0:
            print("⚠️ No active miners found. Exiting.")
            return []
        
        # Sort miners by stake (query highest stake first)
        active_miners.sort(key=lambda x: x['stake'], reverse=True)
        
        # Connect to database
        conn = connect_to_database()
        if not conn:
            return []
        
        # Query miners for predictions
        print(f"\n🔍 Querying top {min(50, len(active_miners))} miners for predictions...")
        predictions = []
        successful_responses = 0
        
        # Query top miners individually with different protocols
        for i, miner in enumerate(active_miners[:50]):  # Limit to top 50 miners
            print(f"\n🔸 Querying miner {i+1}/50 (UID: {miner['uid']}, Stake: {miner['stake']:.2f})")
            print(f"   Hotkey: {miner['hotkey'][:20]}...")
            
            try:
                response, protocol = await query_miner_with_different_protocols(
                    dendrite, miner['axon'], miner
                )
                
                if response:
                    print(f"    ✅ Response received via {protocol}")
                    successful_responses += 1
                    
                    # Process response based on type
                    processed_preds = process_miner_response(response, miner)
                    
                    if processed_preds:
                        print(f"    📊 Extracted {len(processed_preds)} predictions")
                        for pred in processed_preds:
                            store_prediction(conn, pred)
                            predictions.append(pred)
                    else:
                        print(f"    ⚠️  No predictions in response")
                else:
                    print(f"    ❌ No response from miner")
                    
            except Exception as e:
                print(f"    ❌ Error querying miner: {e}")
            
            # Small delay between queries
            await asyncio.sleep(0.5)
        
        conn.close()
        
        if successful_responses == 0:
            print(f"\n⚠️  No miners responded successfully out of {len(active_miners)} active miners")
            print("   This is common with Bettensor as miners may not always have active predictions")
            print("   Exiting without predictions.")
            return []
        
        print(f"\n✅ Successfully queried {successful_responses} miners and extracted {len(predictions)} predictions")
        return predictions
        
    except Exception as e:
        print(f"❌ Error connecting to Bittensor network: {e}")
        print(f"   Traceback: {traceback.format_exc()}")
        return []

def process_miner_response(response, miner_info):
    """Process different types of miner responses"""
    predictions = []
    
    try:
        # Try different response formats
        if hasattr(response, 'prediction_dict') and response.prediction_dict:
            # GameData format
            for pred_id, pred_data in response.prediction_dict.items():
                pred = create_prediction_record(pred_data, miner_info, pred_id)
                if pred:
                    predictions.append(pred)
                    
        elif hasattr(response, 'predictions') and response.predictions:
            # PredictionSynapse format
            for i, pred_data in enumerate(response.predictions):
                pred = create_prediction_record(pred_data, miner_info, f"pred_{i}")
                if pred:
                    predictions.append(pred)
                    
        elif hasattr(response, 'data') and response.data:
            # BettensorSynapse format
            if isinstance(response.data, dict):
                if 'predictions' in response.data:
                    for i, pred_data in enumerate(response.data['predictions']):
                        pred = create_prediction_record(pred_data, miner_info, f"pred_{i}")
                        if pred:
                            predictions.append(pred)
                else:
                    # Try to treat data as a single prediction
                    pred = create_prediction_record(response.data, miner_info, "pred_0")
                    if pred:
                        predictions.append(pred)
                        
        else:
            # Try to extract any available data
            response_dict = response.__dict__ if hasattr(response, '__dict__') else {}
            if response_dict:
                pred = create_prediction_record(response_dict, miner_info, "pred_0")
                if pred:
                    predictions.append(pred)
                    
    except Exception as e:
        print(f"    ❌ Error processing response: {e}")
        
    return predictions

def create_prediction_record(pred_data, miner_info, pred_id):
    """Create a standardized prediction record"""
    try:
        # Convert to dict if needed
        if hasattr(pred_data, 'dict'):
            pred_data = pred_data.dict()
        elif hasattr(pred_data, '__dict__'):
            pred_data = pred_data.__dict__
        
        # Create prediction record with defaults
        prediction = {
            'prediction_id': pred_data.get('prediction_id', f"PRED_{miner_info['uid']}_{pred_id}"),
            'game_id': pred_data.get('game_id', f"GAME_{pred_id}"),
            'miner_uid': miner_info['uid'],
            'miner_hotkey': miner_info['hotkey'],
            'prediction_date': pred_data.get('prediction_date', datetime.now()),
            'predicted_outcome': pred_data.get('predicted_outcome', 'unknown'),
            'predicted_odds': float(pred_data.get('predicted_odds', 1.0)),
            'wager': float(pred_data.get('wager', 0.0)),
            'team_a': pred_data.get('team_a', 'Team A'),
            'team_b': pred_data.get('team_b', 'Team B'),
            'team_a_odds': float(pred_data.get('team_a_odds', 2.0)),
            'team_b_odds': float(pred_data.get('team_b_odds', 2.0)),
            'tie_odds': float(pred_data.get('tie_odds', 3.0)) if pred_data.get('tie_odds') else None,
            'sport': pred_data.get('sport', 'unknown'),
            'league': pred_data.get('league', 'unknown'),
            'outcome': pred_data.get('outcome')
        }
        
        # Validate required fields
        if prediction['predicted_outcome'] not in ['team_a', 'team_b', 'tie', 'unknown']:
            return None
            
        return prediction
        
    except Exception as e:
        print(f"    ❌ Error creating prediction record: {e}")
        return None

async def simulate_predictions():
    """Generate realistic simulation data"""
    print("🎭 Running in simulation mode...")
    
    conn = connect_to_database()
    if not conn:
        return []
    
    # More realistic simulation data
    current_games = [
        {
            'id': 'NBA_2025_LAL_vs_BOS',
            'team_a': 'Los Angeles Lakers',
            'team_b': 'Boston Celtics',
            'sport': 'basketball',
            'league': 'NBA',
            'team_a_odds': 1.85,
            'team_b_odds': 2.10
        },
        {
            'id': 'NFL_2025_KC_vs_BUF',
            'team_a': 'Kansas City Chiefs',
            'team_b': 'Buffalo Bills',
            'sport': 'football',
            'league': 'NFL',
            'team_a_odds': 1.90,
            'team_b_odds': 2.05
        },
        {
            'id': 'EPL_2025_MAN_vs_LIV',
            'team_a': 'Manchester United',
            'team_b': 'Liverpool',
            'sport': 'soccer',
            'league': 'EPL',
            'team_a_odds': 2.20,
            'team_b_odds': 1.80
        },
        {
            'id': 'NHL_2025_TOR_vs_MTL',
            'team_a': 'Toronto Maple Leafs',
            'team_b': 'Montreal Canadiens',
            'sport': 'hockey',
            'league': 'NHL',
            'team_a_odds': 1.95,
            'team_b_odds': 2.00
        }
    ]
    
    # Simulate 15 miners with different prediction patterns
    simulated_miners = []
    for i in range(15):
        simulated_miners.append({
            'uid': 100 + i,
            'hotkey': f'5DSimMiner{i:02d}{"1" * (40-len(str(i)))}',
            'stake': 1000 + i * 100
        })
    
    predictions = []
    
    for game in current_games:
        # Create different consensus patterns for each game
        if game['id'] == 'NBA_2025_LAL_vs_BOS':
            # Strong consensus for team_a (Lakers)
            outcomes = ['team_a'] * 10 + ['team_b'] * 3 + ['tie'] * 2
        elif game['id'] == 'NFL_2025_KC_vs_BUF':
            # Moderate consensus for team_a (Chiefs)
            outcomes = ['team_a'] * 7 + ['team_b'] * 6 + ['tie'] * 2
        elif game['id'] == 'EPL_2025_MAN_vs_LIV':
            # Strong consensus for team_b (Liverpool)
            outcomes = ['team_b'] * 11 + ['team_a'] * 3 + ['tie'] * 1
        else:
            # No clear consensus
            outcomes = ['team_a'] * 5 + ['team_b'] * 5 + ['tie'] * 5
        
        for i, miner in enumerate(simulated_miners):
            if i < len(outcomes):
                predicted_outcome = outcomes[i]
                
                # Vary odds based on outcome
                if predicted_outcome == 'team_a':
                    odds = game['team_a_odds'] + (i % 5) * 0.05
                elif predicted_outcome == 'team_b':
                    odds = game['team_b_odds'] + (i % 5) * 0.05
                else:
                    odds = 3.0 + (i % 5) * 0.1
                
                prediction = {
                    'prediction_id': f"SIM_PRED_{miner['uid']}_{game['id']}",
                    'game_id': game['id'],
                    'miner_uid': miner['uid'],
                    'miner_hotkey': miner['hotkey'],
                    'prediction_date': datetime.now(),
                    'predicted_outcome': predicted_outcome,
                    'predicted_odds': odds,
                    'wager': 25.0 + (i % 8) * 12.5,  # 25-112.5 range
                    'team_a': game['team_a'],
                    'team_b': game['team_b'],
                    'team_a_odds': game['team_a_odds'],
                    'team_b_odds': game['team_b_odds'],
                    'tie_odds': 3.50,
                    'sport': game['sport'],
                    'league': game['league']
                }
                
                store_prediction(conn, prediction)
                predictions.append(prediction)
                
                print(f"  📊 Miner {miner['uid']}: {game['team_a']} vs {game['team_b']} -> {predicted_outcome} @ {odds:.2f}")
    
    conn.close()
    print(f"\n✅ Generated {len(predictions)} realistic predictions for {len(current_games)} games")
    return predictions

def analyze_consensus(predictions: List[Dict]):
    """Analyze predictions to find consensus signals"""
    print("\n📈 ANALYZING CONSENSUS SIGNALS")
    print("=" * 70)
    
    if not predictions:
        print("❌ No predictions to analyze")
        return
    
    # Group predictions by game
    games = {}
    for pred in predictions:
        game_id = pred['game_id']
        if game_id not in games:
            games[game_id] = []
        games[game_id].append(pred)
    
    print(f"🎮 Found predictions for {len(games)} games:\n")
    
    strong_signals = []
    
    for game_id, game_preds in games.items():
        # Get game info
        first_pred = game_preds[0]
        team_a = first_pred['team_a']
        team_b = first_pred['team_b']
        sport = first_pred['sport'].upper()
        league = first_pred['league'].upper()
        
        print(f"🏆 {team_a} vs {team_b}")
        print(f"   🏷️  {sport} - {league}")
        print(f"   📊 {len(game_preds)} predictions from {len(set(p['miner_uid'] for p in game_preds))} unique miners")
        
        # Count outcomes
        outcomes = {}
        total_wager = 0
        
        for pred in game_preds:
            outcome = pred['predicted_outcome']
            wager = pred['wager']
            
            if outcome not in outcomes:
                outcomes[outcome] = {'count': 0, 'total_wager': 0, 'odds_sum': 0}
            
            outcomes[outcome]['count'] += 1
            outcomes[outcome]['total_wager'] += wager
            outcomes[outcome]['odds_sum'] += pred['predicted_odds']
            total_wager += wager
        
        print(f"   💰 Total wagered: ${total_wager:.2f}")
        
        # Find consensus
        if outcomes:
            consensus_outcome = max(outcomes.keys(), key=lambda x: outcomes[x]['count'])
            consensus_count = outcomes[consensus_outcome]['count']
            consensus_percentage = (consensus_count / len(game_preds)) * 100
            avg_odds = outcomes[consensus_outcome]['odds_sum'] / consensus_count
            
            if consensus_percentage >= 60:
                signal_strength = "🟢 STRONG"
                strong_signals.append({
                    'game': f"{team_a} vs {team_b}",
                    'sport_league': f"{sport}/{league}",
                    'outcome': consensus_outcome,
                    'percentage': consensus_percentage,
                    'odds': avg_odds,
                    'wager': outcomes[consensus_outcome]['total_wager']
                })
            elif consensus_percentage >= 40:
                signal_strength = "🟡 MODERATE"
            else:
                signal_strength = "🔴 WEAK"
            
            print(f"   🎯 CONSENSUS: {consensus_outcome} ({signal_strength} - {consensus_percentage:.1f}%)")
            print(f"   ")
            print(f"   📈 Breakdown:")
            
            for outcome, data in outcomes.items():
                percentage = (data['count'] / len(game_preds)) * 100
                wager_percentage = (data['total_wager'] / total_wager) * 100 if total_wager > 0 else 0
                avg_odds = data['odds_sum'] / data['count']
                
                outcome_name = {
                    'team_a': team_a,
                    'team_b': team_b,
                    'tie': 'Tie'
                }.get(outcome, outcome)
                
                print(f"      {outcome_name}: {data['count']} votes ({percentage:.1f}%) | "
                      f"${data['total_wager']:.0f} ({wager_percentage:.1f}%) | "
                      f"Avg odds: {avg_odds:.2f}")
        
        print()
    
    # Summary of strong signals
    if strong_signals:
        print("\n🚨 STRONG CONSENSUS SIGNALS (60%+ agreement)")
        print("=" * 70)
        for signal in strong_signals:
            print(f"🎯 {signal['game']} ({signal['sport_league']})")
            print(f"   Prediction: {signal['outcome']} at {signal['odds']:.2f} odds")
            print(f"   Strength: 🟢 STRONG ({signal['percentage']:.1f}%)")
            print(f"   Total wagered: ${signal['wager']:.2f}")
            print()

async def main():
    print("🚀 BETTENSOR PREDICTIONS EXTRACTOR (IMPROVED)")
    print("=" * 60)
    
    # Query miners for predictions
    predictions = await query_bittensor_miners_improved()
    
    if predictions:
        print(f"\n✅ Successfully extracted {len(predictions)} predictions")
        
        # Analyze consensus
        analyze_consensus(predictions)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"predictions_extract_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(predictions, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {results_file}")
        
        # Show database stats
        conn = connect_to_database()
        if conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM predictions")
            total_predictions = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT game_id) FROM predictions")
            unique_games = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT miner_hotkey) FROM predictions")
            unique_miners = cursor.fetchone()[0]
            
            print(f"\n📊 Database now contains:")
            print(f"   - {total_predictions} total predictions")
            print(f"   - {unique_games} unique games")
            print(f"   - {unique_miners} unique miners")
            
            conn.close()
    else:
        print("❌ No predictions extracted")

if __name__ == "__main__":
    asyncio.run(main())
