"""Test script to verify both dataset adapters work correctly."""
import sys
from pathlib import Path

# Add parent directory to path to import src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.adapters import TDSCAdapter, NHTSAdapter


def test_tdsc_adapter():
    """Test TDSC adapter with actual data."""
    print("\n" + "="*80)
    print("TESTING TDSC ADAPTER")
    print("="*80)

    # Load TDSC config
    config = Config()
    assert config.dataset_type == 'tdsc', f"Expected 'tdsc', got '{config.dataset_type}'"

    dataset_config = config.get_dataset_config()
    adapter = TDSCAdapter(dataset_config)

    # Test get_person_attributes with a sample person ID
    print("\n1. Testing get_person_attributes...")
    # Read first person ID from file
    import pandas as pd
    df = pd.read_csv(dataset_config['HOUSEHOLDS_FILE'])
    sample_person_id = df['perno'].iloc[0]
    print(f"   Using person_id: {sample_person_id}")

    person_attrs = adapter.get_person_attributes(sample_person_id)
    if person_attrs:
        print(f"   ✓ Found person with {len(person_attrs)} attributes")
        print(f"   Sample attributes: gender={person_attrs.get('gender')}, "
              f"age={person_attrs.get('age_calculated_nrel_agebin')}")
    else:
        print("   ✗ Person not found")
        return False

    # Test get_current_trip_attributes
    print("\n2. Testing get_current_trip_attributes...")
    trips_df = pd.read_csv(dataset_config['TRIPS_FILE'])
    sample_trip_id = trips_df['_id'].iloc[0]
    print(f"   Using trip_id: {sample_trip_id}")

    trip_attrs = adapter.get_current_trip_attributes(sample_trip_id)
    if trip_attrs:
        print(f"   ✓ Found trip with {len(trip_attrs)} attributes")
        print(f"   Speed: {trip_attrs.get('speed_mps')} m/s - {trip_attrs.get('speed_interpretation')}")
        print(f"   Distance: {trip_attrs.get('data_distance')}m, Duration: {trip_attrs.get('data_duration')}s")
    else:
        print("   ✗ Trip not found")
        return False

    # Test get_trip_history
    print("\n3. Testing get_trip_history...")
    person_id_for_trip = trips_df[trips_df['_id'] == sample_trip_id]['perno'].iloc[0]
    history = adapter.get_trip_history(person_id_for_trip, sample_trip_id)
    print(f"   ✓ Found {len(history)} previous trips for person {person_id_for_trip}")

    # Test mode and purpose options
    print("\n4. Testing mode and purpose options...")
    modes = adapter.get_mode_options()
    purposes = adapter.get_purpose_options()
    print(f"   ✓ {len(modes)} mode options")
    print(f"   ✓ {len(purposes)} purpose options")

    print("\n✓ TDSC adapter tests passed!")
    return True


def test_nhts_adapter():
    """Test NHTS adapter with actual data."""
    print("\n" + "="*80)
    print("TESTING NHTS ADAPTER")
    print("="*80)

    # Create config for NHTS
    import yaml
    config_path = Path(__file__).parent.parent / 'configs' / 'config.yaml'

    # Temporarily modify config to use NHTS
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)

    original_dataset_type = config_data['DATASET_TYPE']
    config_data['DATASET_TYPE'] = 'nhts'

    # Write temporary config
    temp_config_path = Path(__file__).parent.parent / 'configs' / 'config_temp.yaml'
    with open(temp_config_path, 'w') as f:
        yaml.dump(config_data, f)

    try:
        config = Config(str(temp_config_path))
        assert config.dataset_type == 'nhts', f"Expected 'nhts', got '{config.dataset_type}'"

        dataset_config = config.get_dataset_config()
        adapter = NHTSAdapter(dataset_config)

        # Test get_person_attributes with a sample person ID
        print("\n1. Testing get_person_attributes...")
        import pandas as pd
        df = pd.read_csv(dataset_config['PERSON_FILE'])
        sample_person_id = df['person_id'].iloc[0]
        print(f"   Using person_id: {sample_person_id}")

        person_attrs = adapter.get_person_attributes(str(sample_person_id))
        if person_attrs:
            print(f"   ✓ Found person with {len(person_attrs)} attributes")
            print(f"   Sample attributes: age={person_attrs.get('age')}, "
                  f"sex={person_attrs.get('sex')}, driver={person_attrs.get('is_driver')}")
        else:
            print("   ✗ Person not found")
            return False

        # Test get_current_trip_attributes
        print("\n2. Testing get_current_trip_attributes...")
        trips_df = pd.read_csv(dataset_config['TRIP_FILE'])
        sample_trip_id = trips_df['trip_id'].iloc[0]
        print(f"   Using trip_id: {sample_trip_id}")

        trip_attrs = adapter.get_current_trip_attributes(str(sample_trip_id))
        if trip_attrs:
            print(f"   ✓ Found trip with {len(trip_attrs)} attributes")
            print(f"   Speed: {trip_attrs.get('speed_mps')} m/s - {trip_attrs.get('speed_interpretation')}")
            print(f"   Distance: {trip_attrs.get('trip_distance_mi')} miles, Duration: {trip_attrs.get('travel_time_min')} min")
        else:
            print("   ✗ Trip not found")
            return False

        # Test get_trip_history
        print("\n3. Testing get_trip_history...")
        person_id_for_trip = trips_df[trips_df['trip_id'] == sample_trip_id]['person_id'].iloc[0]
        history = adapter.get_trip_history(str(person_id_for_trip), str(sample_trip_id))
        print(f"   ✓ Found {len(history)} previous trips for person {person_id_for_trip}")

        # Test mode and purpose options
        print("\n4. Testing mode and purpose options...")
        modes = adapter.get_mode_options()
        purposes = adapter.get_purpose_options()
        print(f"   ✓ {len(modes)} mode options")
        print(f"   ✓ {len(purposes)} purpose options")

        print("\n✓ NHTS adapter tests passed!")
        return True

    finally:
        # Clean up temp config
        if temp_config_path.exists():
            temp_config_path.unlink()


def main():
    """Run all adapter tests."""
    print("\n" + "="*80)
    print("DATASET ADAPTER TEST SUITE")
    print("="*80)

    try:
        tdsc_passed = test_tdsc_adapter()
        nhts_passed = test_nhts_adapter()

        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        print(f"TDSC Adapter: {'✓ PASSED' if tdsc_passed else '✗ FAILED'}")
        print(f"NHTS Adapter: {'✓ PASSED' if nhts_passed else '✗ FAILED'}")

        if tdsc_passed and nhts_passed:
            print("\n✓ All tests passed! Both adapters are working correctly.")
            print("\nTo switch between datasets, change DATASET_TYPE in configs/config.yaml:")
            print("  - DATASET_TYPE: 'tdsc' (default)")
            print("  - DATASET_TYPE: 'nhts'")
            return 0
        else:
            print("\n✗ Some tests failed.")
            return 1

    except Exception as e:
        print(f"\n✗ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
