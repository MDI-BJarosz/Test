"""Test prediction accuracy on a sample of trips."""
import asyncio
import sys
import time
import json
import re
import pandas as pd
from pathlib import Path
from google.genai import types
from sklearn.metrics import f1_score, confusion_matrix

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.main import runner, session_service
from src.config import config

# ============================================================================
# CONFIGURATION
# ============================================================================
DATASET_NAME = "tdsc"  # Change to "nhts" to test NHTS dataset
TEST_MODE = False       # Set to False for full 50-trip test, True for 5-trip test
SAMPLE_SIZE = 5 if TEST_MODE else 50  # Number of trips to test
RANDOM_SEED = 42       # For repeatable sampling
# ============================================================================

# Override config dataset type to match test dataset
config._config['DATASET_TYPE'] = DATASET_NAME


def get_dataset_columns(dataset_name: str) -> dict:
    """
    Get dataset-specific column names and file paths.

    Returns:
        dict with keys: trip_file, person_id_col, trip_id_col, mode_col, purpose_col
    """
    if dataset_name.lower() == 'tdsc':
        dataset_config = config._config['TDSC']
        return {
            'trip_file': dataset_config['TRIPS_FILE'],
            'person_id_col': 'perno',
            'trip_id_col': '_id',
            'mode_col': 'data_user_input_mode_confirm',
            'purpose_col': 'data_user_input_purpose_confirm'
        }
    elif dataset_name.lower() == 'nhts':
        dataset_config = config._config['NHTS']
        return {
            'trip_file': dataset_config['TRIP_FILE'],
            'person_id_col': 'person_id',
            'trip_id_col': 'trip_id',
            'mode_col': 'trip_mode',
            'purpose_col': 'trip_purpose'
        }
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Must be 'tdsc' or 'nhts'")


async def get_prediction(person_no: str, trip_id: str) -> tuple:
    """
    Get mode and purpose prediction from agent with Top-3 ranking.

    Returns:
        tuple: (rank1_mode, rank1_purpose, rank1_conf,
                rank2_mode, rank2_purpose, rank2_conf,
                rank3_mode, rank3_purpose, rank3_conf,
                reasoning)
               or (None, None, None, None, None, None, None, None, None, None) if error
    """
    query = f"""Predict the transportation mode and trip purpose for person {person_no} and trip {trip_id}.

CRITICAL: Provide your TOP 3 ranked predictions in the following JSON format:
{{
  "predictions": [
    {{"rank": 1, "mode": "<mode_value>", "purpose": "<purpose_value>", "confidence": "high|medium|low"}},
    {{"rank": 2, "mode": "<mode_value>", "purpose": "<purpose_value>", "confidence": "high|medium|low"}},
    {{"rank": 3, "mode": "<mode_value>", "purpose": "<purpose_value>", "confidence": "high|medium|low"}}
  ],
  "reasoning": "<single detailed reasoning string explaining all three predictions>"
}}

RANKING RULES (MUST FOLLOW):
- Rank 1: Your most confident prediction based on all evidence
- Rank 2: Must be a DIFFERENT mode than Rank 1 (explore alternative hypothesis)
- Rank 3: Must be DIFFERENT from both Rank 1 and Rank 2 (maximize diversity)
- NO DUPLICATES ALLOWED: All 3 modes must be different, all 3 purposes must be different
- Assign confidence honestly: high (>70% sure), medium (40-70%), low (<40%)

RESPOND WITH ONLY THE JSON - NO OTHER TEXT BEFORE OR AFTER"""

    try:
        # Create session
        session_id = f"{person_no}_{trip_id}_{int(time.time())}"
        session_service.create_session_sync(
            app_name="travel_survey_helper",
            user_id="test_user",
            session_id=session_id
        )

        # Create and send message
        message = types.Content(role="user", parts=[types.Part(text=query)])
        events = runner.run(user_id="test_user", session_id=session_id, new_message=message)

        # Collect response
        response_text = ""
        for event in events:
            if hasattr(event, 'content') and event.content:
                for part in event.content.parts:
                    if hasattr(part, 'text') and part.text:
                        response_text += part.text

        # Parse JSON response
        if not response_text or len(response_text.strip()) == 0:
            print(f"    Error: Empty response from agent")
            return None, None, None, None, None, None, None, None, None, None

        # Try to extract JSON from response (might be wrapped in markdown code blocks)
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find raw JSON object with nested structure
            json_match = re.search(r'\{.*?"predictions".*?\[.*?\].*?\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                print(f"    Error: Could not find JSON in response: {response_text[:200]}...")
                return None, None, None, None, None, None, None, None, None, None

        # Parse JSON
        data = json.loads(json_str)
        predictions = data.get('predictions', [])
        reasoning = data.get('reasoning', response_text)

        # Validate we have 3 predictions
        if len(predictions) < 3:
            print(f"    Error: Expected 3 predictions, got {len(predictions)}")
            return None, None, None, None, None, None, None, None, None, None

        # Extract Top-3 predictions
        rank1 = predictions[0] if len(predictions) > 0 else {}
        rank2 = predictions[1] if len(predictions) > 1 else {}
        rank3 = predictions[2] if len(predictions) > 2 else {}

        rank1_mode = rank1.get('mode')
        rank1_purpose = rank1.get('purpose')
        rank1_conf = rank1.get('confidence')

        rank2_mode = rank2.get('mode')
        rank2_purpose = rank2.get('purpose')
        rank2_conf = rank2.get('confidence')

        rank3_mode = rank3.get('mode')
        rank3_purpose = rank3.get('purpose')
        rank3_conf = rank3.get('confidence')

        # Validate no None values
        if None in [rank1_mode, rank1_purpose, rank2_mode, rank2_purpose, rank3_mode, rank3_purpose]:
            print(f"    Error: Missing values in predictions")
            return None, None, None, None, None, None, None, None, None, None

        # Validate no duplicates in modes
        modes = [rank1_mode, rank2_mode, rank3_mode]
        if len(modes) != len(set(modes)):
            print(f"    Warning: Duplicate modes detected: {modes}")
            # Don't return None - let it continue with duplicates for now

        # Validate no duplicates in purposes
        purposes = [rank1_purpose, rank2_purpose, rank3_purpose]
        if len(purposes) != len(set(purposes)):
            print(f"    Warning: Duplicate purposes detected: {purposes}")
            # Don't return None - let it continue with duplicates for now

        # Clean up reasoning text - convert to single paragraph
        reasoning_cleaned = ' '.join(reasoning.split())

        return (rank1_mode, rank1_purpose, rank1_conf,
                rank2_mode, rank2_purpose, rank2_conf,
                rank3_mode, rank3_purpose, rank3_conf,
                reasoning_cleaned)

    except Exception as e:
        print(f"    Error: {e}")
        return None, None, None, None, None, None, None, None, None, None


def calculate_category_metrics(y_true, y_pred, category):
    """Calculate TP, FP, TN, FN, precision, recall for a single category."""
    y_true_binary = (y_true == category).astype(int)
    y_pred_binary = (y_pred == category).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1]).ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn),
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def calculate_category_metrics_top2(y_true, y_pred1, y_pred2, category):
    """
    Calculate TP, FP, TN, FN for top-2 predictions with hierarchical evaluation.

    Logic:
    - TP: Actual = category AND (pred1 = category OR pred2 = category)
    - FN: Actual = category AND pred1 ≠ category AND pred2 ≠ category
    - TN: Actual ≠ category AND pred1 ≠ category
    - FP: Actual ≠ category AND pred1 = category

    Pred2 is only considered when actual = category and pred1 failed.
    """
    tp = 0
    fn = 0
    tn = 0
    fp = 0

    for i in range(len(y_true)):
        actual = y_true.iloc[i]
        pred1 = y_pred1.iloc[i]
        pred2 = y_pred2.iloc[i]

        if actual == category:
            # Actual IS the category - check if either prediction caught it
            if pred1 == category or pred2 == category:
                tp += 1
            else:
                fn += 1
        else:
            # Actual is NOT the category - only check pred1
            if pred1 == category:
                fp += 1
            else:
                tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn),
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def analyze_accuracy_results(dataset_name: str,
                             input_csv: str = None,
                             output_csv: str = None):
    """Analyze accuracy results and generate detailed metrics CSV."""
    # Default file paths with dataset name
    if input_csv is None:
        input_csv = f'tst/results/accuracy_results_{dataset_name}.csv'
    if output_csv is None:
        output_csv = f'tst/results/accuracy_metrics_{dataset_name}.csv'

    print("\n" + "="*80)
    print(f"ANALYZING ACCURACY RESULTS - {dataset_name.upper()} DATASET")
    print("="*80 + "\n")

    # Read results
    df = pd.read_csv(input_csv)

    # Calculate top-1 metrics
    mode_top1_correct = df['mode_top1_correct'].sum()
    purpose_top1_correct = df['purpose_top1_correct'].sum()

    mode_top1_acc = mode_top1_correct / len(df)
    purpose_top1_acc = purpose_top1_correct / len(df)

    mode_f1_overall = f1_score(df['actual_mode'], df['rank1_mode'],
                                average='weighted', zero_division=0)
    purpose_f1_overall = f1_score(df['actual_purpose'], df['rank1_purpose'],
                                   average='weighted', zero_division=0)

    # Calculate top-2 accuracy
    mode_top2_correct = df['mode_top2_correct'].sum()
    purpose_top2_correct = df['purpose_top2_correct'].sum()

    mode_top2_acc = mode_top2_correct / len(df)
    purpose_top2_acc = purpose_top2_correct / len(df)

    # Calculate top-3 accuracy
    mode_top3_correct = df['mode_top3_correct'].sum()
    purpose_top3_correct = df['purpose_top3_correct'].sum()

    mode_top3_acc = mode_top3_correct / len(df)
    purpose_top3_acc = purpose_top3_correct / len(df)

    # Calculate top-2 F1-score
    # Create binary predictions: for each row, predicted value is the actual if it matches rank1 or rank2
    # Otherwise use rank1 as the prediction
    mode_pred_top2 = df.apply(
        lambda row: row['actual_mode'] if (row['actual_mode'] == row['rank1_mode'] or
                                            row['actual_mode'] == row['rank2_mode'])
                    else row['rank1_mode'],
        axis=1
    )
    purpose_pred_top2 = df.apply(
        lambda row: row['actual_purpose'] if (row['actual_purpose'] == row['rank1_purpose'] or
                                               row['actual_purpose'] == row['rank2_purpose'])
                      else row['rank1_purpose'],
        axis=1
    )

    mode_f1_top2 = f1_score(df['actual_mode'], mode_pred_top2, average='weighted', zero_division=0)
    purpose_f1_top2 = f1_score(df['actual_purpose'], purpose_pred_top2, average='weighted', zero_division=0)

    # Calculate top-3 F1-score
    mode_pred_top3 = df.apply(
        lambda row: row['actual_mode'] if (row['actual_mode'] == row['rank1_mode'] or
                                            row['actual_mode'] == row['rank2_mode'] or
                                            row['actual_mode'] == row['rank3_mode'])
                    else row['rank1_mode'],
        axis=1
    )
    purpose_pred_top3 = df.apply(
        lambda row: row['actual_purpose'] if (row['actual_purpose'] == row['rank1_purpose'] or
                                               row['actual_purpose'] == row['rank2_purpose'] or
                                               row['actual_purpose'] == row['rank3_purpose'])
                      else row['rank1_purpose'],
        axis=1
    )

    mode_f1_top3 = f1_score(df['actual_mode'], mode_pred_top3, average='weighted', zero_division=0)
    purpose_f1_top3 = f1_score(df['actual_purpose'], purpose_pred_top3, average='weighted', zero_division=0)

    print(f"Top-1 Mode Accuracy: {mode_top1_acc:.4f} ({mode_top1_correct}/{len(df)})")
    print(f"Top-1 Mode F1-Score: {mode_f1_overall:.4f}")
    print(f"Top-1 Purpose Accuracy: {purpose_top1_acc:.4f} ({purpose_top1_correct}/{len(df)})")
    print(f"Top-1 Purpose F1-Score: {purpose_f1_overall:.4f}")
    print(f"\nTop-2 Mode Accuracy: {mode_top2_acc:.4f} ({mode_top2_correct}/{len(df)})")
    print(f"Top-2 Mode F1-Score: {mode_f1_top2:.4f}")
    print(f"Top-2 Purpose Accuracy: {purpose_top2_acc:.4f} ({purpose_top2_correct}/{len(df)})")
    print(f"Top-2 Purpose F1-Score: {purpose_f1_top2:.4f}")
    print(f"\nTop-3 Mode Accuracy: {mode_top3_acc:.4f} ({mode_top3_correct}/{len(df)})")
    print(f"Top-3 Mode F1-Score: {mode_f1_top3:.4f}")
    print(f"Top-3 Purpose Accuracy: {purpose_top3_acc:.4f} ({purpose_top3_correct}/{len(df)})")
    print(f"Top-3 Purpose F1-Score: {purpose_f1_top3:.4f}\n")

    # Collect metrics
    metrics = []

    # Mode metrics (top-1 only)
    print("MODE METRICS (Top-1):")
    print("-" * 80)
    for mode in sorted(df['actual_mode'].unique()):
        m = calculate_category_metrics(df['actual_mode'], df['rank1_mode'], mode)
        metrics.append({
            'prediction_type': 'top1',
            'category_type': 'mode',
            'category': mode,
            **m
        })
        print(f"{mode:20s} TP:{m['true_positives']:3d} FP:{m['false_positives']:3d} "
              f"TN:{m['true_negatives']:3d} FN:{m['false_negatives']:3d} "
              f"P:{m['precision']:.3f} R:{m['recall']:.3f} F1:{m['f1_score']:.3f}")

    # Purpose metrics (top-1 only)
    print("\nPURPOSE METRICS (Top-1):")
    print("-" * 80)
    for purpose in sorted(df['actual_purpose'].unique()):
        m = calculate_category_metrics(df['actual_purpose'], df['rank1_purpose'], purpose)
        metrics.append({
            'prediction_type': 'top1',
            'category_type': 'purpose',
            'category': purpose,
            **m
        })
        print(f"{purpose:20s} TP:{m['true_positives']:3d} FP:{m['false_positives']:3d} "
              f"TN:{m['true_negatives']:3d} FN:{m['false_negatives']:3d} "
              f"P:{m['precision']:.3f} R:{m['recall']:.3f} F1:{m['f1_score']:.3f}")

    # Mode metrics (top-2 with hierarchical evaluation)
    print("\nMODE METRICS (Top-2):")
    print("-" * 80)
    for mode in sorted(df['actual_mode'].unique()):
        m = calculate_category_metrics_top2(df['actual_mode'], df['rank1_mode'], df['rank2_mode'], mode)
        metrics.append({
            'prediction_type': 'top2',
            'category_type': 'mode',
            'category': mode,
            **m
        })
        print(f"{mode:20s} TP:{m['true_positives']:3d} FP:{m['false_positives']:3d} "
              f"TN:{m['true_negatives']:3d} FN:{m['false_negatives']:3d} "
              f"P:{m['precision']:.3f} R:{m['recall']:.3f} F1:{m['f1_score']:.3f}")

    # Purpose metrics (top-2 with hierarchical evaluation)
    print("\nPURPOSE METRICS (Top-2):")
    print("-" * 80)
    for purpose in sorted(df['actual_purpose'].unique()):
        m = calculate_category_metrics_top2(df['actual_purpose'], df['rank1_purpose'], df['rank2_purpose'], purpose)
        metrics.append({
            'prediction_type': 'top2',
            'category_type': 'purpose',
            'category': purpose,
            **m
        })
        print(f"{purpose:20s} TP:{m['true_positives']:3d} FP:{m['false_positives']:3d} "
              f"TN:{m['true_negatives']:3d} FN:{m['false_negatives']:3d} "
              f"P:{m['precision']:.3f} R:{m['recall']:.3f} F1:{m['f1_score']:.3f}")

    # Add overall rows
    metrics.extend([
        {'prediction_type': 'top1', 'category_type': 'overall', 'category': 'mode',
         'true_positives': None, 'false_positives': None, 'true_negatives': None,
         'false_negatives': None, 'precision': None, 'recall': None,
         'f1_score': mode_f1_overall, 'accuracy': mode_top1_acc},
        {'prediction_type': 'top1', 'category_type': 'overall', 'category': 'purpose',
         'true_positives': None, 'false_positives': None, 'true_negatives': None,
         'false_negatives': None, 'precision': None, 'recall': None,
         'f1_score': purpose_f1_overall, 'accuracy': purpose_top1_acc},
        {'prediction_type': 'top2', 'category_type': 'overall', 'category': 'mode',
         'true_positives': None, 'false_positives': None, 'true_negatives': None,
         'false_negatives': None, 'precision': None, 'recall': None,
         'f1_score': mode_f1_top2, 'accuracy': mode_top2_acc},
        {'prediction_type': 'top2', 'category_type': 'overall', 'category': 'purpose',
         'true_positives': None, 'false_positives': None, 'true_negatives': None,
         'false_negatives': None, 'precision': None, 'recall': None,
         'f1_score': purpose_f1_top2, 'accuracy': purpose_top2_acc},
        {'prediction_type': 'top3', 'category_type': 'overall', 'category': 'mode',
         'true_positives': None, 'false_positives': None, 'true_negatives': None,
         'false_negatives': None, 'precision': None, 'recall': None,
         'f1_score': mode_f1_top3, 'accuracy': mode_top3_acc},
        {'prediction_type': 'top3', 'category_type': 'overall', 'category': 'purpose',
         'true_positives': None, 'false_positives': None, 'true_negatives': None,
         'false_negatives': None, 'precision': None, 'recall': None,
         'f1_score': purpose_f1_top3, 'accuracy': purpose_top3_acc}
    ])

    # Save to CSV
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(output_csv, index=False)
    print(f"\nMetrics saved to: {output_csv}")
    print("="*80 + "\n")


async def main():
    """Test prediction accuracy on sample trips."""

    print("="*80)
    print(f"PREDICTION ACCURACY TEST - {DATASET_NAME.upper()} DATASET")
    print("="*80)

    # Get dataset-specific column names and file paths
    dataset_cols = get_dataset_columns(DATASET_NAME)

    # Load trips data
    trips_df = pd.read_csv(dataset_cols['trip_file'])

    # Filter trips that have actual mode and purpose labels
    valid_trips = trips_df[
        trips_df[dataset_cols['mode_col']].notna() &
        trips_df[dataset_cols['purpose_col']].notna()
    ]

    # Simple random sample of SAMPLE_SIZE trips from all valid trips
    # Uses fixed random_state for reproducibility
    sample_df = valid_trips.sample(
        n=min(SAMPLE_SIZE, len(valid_trips)),
        random_state=RANDOM_SEED
    )

    if len(sample_df) == 0:
        print("No valid trips found!")
        return

    total = len(sample_df)
    print(f"\nTesting {total} trips sequentially...")
    print()

    # Track results
    results = []
    mode_correct = 0
    purpose_correct = 0

    # Initialize CSV file with headers (include dataset name in filename)
    results_csv_path = f'tst/results/accuracy_results_{DATASET_NAME}.csv'

    # Verify that file paths are configured correctly
    print(f"Using dataset: {DATASET_NAME}")
    print(f"Using trips file: {dataset_cols['trip_file']}")
    print(f"Sample size: {SAMPLE_SIZE}")
    print(f"Random seed: {RANDOM_SEED}")
    print()
    with open(results_csv_path, 'w') as f:
        f.write('trip_id,person_no,actual_mode,rank1_mode,rank1_conf,rank2_mode,rank2_conf,rank3_mode,rank3_conf,mode_top1_correct,mode_top2_correct,mode_top3_correct,actual_purpose,rank1_purpose,rank2_purpose,rank3_purpose,purpose_top1_correct,purpose_top2_correct,purpose_top3_correct,reasoning\n')

    start_time = time.time()

    for idx, (_, trip) in enumerate(sample_df.iterrows(), 1):
        person_no = str(trip[dataset_cols['person_id_col']])
        trip_id = str(trip[dataset_cols['trip_id_col']])
        actual_mode = trip[dataset_cols['mode_col']]
        actual_purpose = trip[dataset_cols['purpose_col']]

        # For NHTS, we need household_id to uniquely identify trips and persons
        # Create a composite identifier if household_id exists
        if 'household_id' in trip.index and DATASET_NAME.lower() == 'nhts':
            household_id = str(trip['household_id'])
            # Pass composite identifier: "household_id:person_id" for person, "household_id:person_id:trip_id" for trip
            # This is because trip_id is only unique within (household_id, person_id) pair
            person_identifier = f"{household_id}:{person_no}"
            trip_identifier = f"{household_id}:{person_no}:{trip_id}"
        else:
            # For TDSC and other datasets, use simple IDs
            person_identifier = person_no
            trip_identifier = trip_id

        print(f"Trip {idx}/{total}: {trip_identifier[:20]}...")
        print(f"  Actual: mode={actual_mode}, purpose={actual_purpose}")

        # Get prediction (Top-3 ranked)
        (rank1_mode, rank1_purpose, rank1_conf,
         rank2_mode, rank2_purpose, rank2_conf,
         rank3_mode, rank3_purpose, rank3_conf,
         reasoning) = await get_prediction(person_identifier, trip_identifier)

        # Check accuracy for Top-1, Top-2, Top-3
        mode_top1_match = rank1_mode == actual_mode if rank1_mode else False
        mode_top2_match = mode_top1_match or (rank2_mode == actual_mode if rank2_mode else False)
        mode_top3_match = mode_top2_match or (rank3_mode == actual_mode if rank3_mode else False)

        purpose_top1_match = rank1_purpose == actual_purpose if rank1_purpose else False
        purpose_top2_match = purpose_top1_match or (rank2_purpose == actual_purpose if rank2_purpose else False)
        purpose_top3_match = purpose_top2_match or (rank3_purpose == actual_purpose if rank3_purpose else False)

        if mode_top1_match:
            mode_correct += 1
        if purpose_top1_match:
            purpose_correct += 1

        mode_symbol = "✓" if mode_top1_match else "✗"
        purpose_symbol = "✓" if purpose_top1_match else "✗"

        print(f"  Rank 1 ({rank1_conf}): mode={rank1_mode} {mode_symbol}, purpose={rank1_purpose} {purpose_symbol}")
        if rank2_mode or rank2_purpose:
            mode2_symbol = "✓" if rank2_mode == actual_mode else ""
            purpose2_symbol = "✓" if rank2_purpose == actual_purpose else ""
            print(f"  Rank 2 ({rank2_conf}): mode={rank2_mode} {mode2_symbol}, purpose={rank2_purpose} {purpose2_symbol}")
        if rank3_mode or rank3_purpose:
            mode3_symbol = "✓" if rank3_mode == actual_mode else ""
            purpose3_symbol = "✓" if rank3_purpose == actual_purpose else ""
            print(f"  Rank 3 ({rank3_conf}): mode={rank3_mode} {mode3_symbol}, purpose={rank3_purpose} {purpose3_symbol}")

        # Store result in memory
        result = {
            'trip_id': trip_id,
            'person_no': person_no,
            'actual_mode': actual_mode,
            'rank1_mode': rank1_mode,
            'rank1_conf': rank1_conf,
            'rank2_mode': rank2_mode,
            'rank2_conf': rank2_conf,
            'rank3_mode': rank3_mode,
            'rank3_conf': rank3_conf,
            'mode_top1_correct': mode_top1_match,
            'mode_top2_correct': mode_top2_match,
            'mode_top3_correct': mode_top3_match,
            'actual_purpose': actual_purpose,
            'rank1_purpose': rank1_purpose,
            'rank2_purpose': rank2_purpose,
            'rank3_purpose': rank3_purpose,
            'purpose_top1_correct': purpose_top1_match,
            'purpose_top2_correct': purpose_top2_match,
            'purpose_top3_correct': purpose_top3_match,
            'reasoning': reasoning
        }
        results.append(result)

        # Write this result to CSV immediately
        result_df = pd.DataFrame([result])
        result_df.to_csv(results_csv_path, mode='a', header=False, index=False)

        print()

    elapsed_time = time.time() - start_time
    print(f"\nCompleted {total} predictions in {elapsed_time:.1f}s ({total/elapsed_time:.2f} predictions/sec)")
    print()

    # Calculate accuracy
    mode_accuracy = (mode_correct / total) * 100
    purpose_accuracy = (purpose_correct / total) * 100

    print("="*80)
    print("RESULTS")
    print("="*80)
    print(f"Mode Accuracy:    {mode_accuracy:.1f}% ({mode_correct}/{total})")
    print(f"Purpose Accuracy: {purpose_accuracy:.1f}% ({purpose_correct}/{total})")
    print()

    # Results were already saved incrementally
    print(f"Detailed results saved to: {results_csv_path}")

    # Analyze results and generate metrics
    analyze_accuracy_results(DATASET_NAME)


if __name__ == "__main__":
    asyncio.run(main())
