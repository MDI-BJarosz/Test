import pandas as pd
from typing import Optional, Dict, Any, List

def get_person_attributes(person_no: str, households_file_path: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve all attributes for a specific person from the households CSV file.

    Args:
        person_no: Unique person identifier (UUID string from 'perno' column)
        households_file_path: Path to the households CSV file

    Returns:
        Dictionary containing all person attributes with column names as keys, or None if person not found

    Available attributes include: gender, have_driver_license, student, education_level_attained,
    work_for_pay_or_profit, number_of_paid_jobs, primary_job_is_full_time, primary_job_classification,
    mode_used_to_get_to_work, how_long_was_your_commute, flexible_work_schedule_at_primary_job,
    remote_or_other_work_location_allowed, number_of_days_remote_or_other_work_location,
    employment_description, own_or_rent, housing_type, household_income, household_size,
    number_of_children_under_18, number_of_hh_member_with_driver_license, number_of_household_vehicles,
    alternative_mode_if_a_car_is_not_available, disability, length_of_disability, age_calculated_nrel_agebin

    Example:
        >>> attrs = get_person_attributes('e7b24d99-324d-4d6d-b247-9edc87d3c848', 'inputs/households.csv')
        >>> print(attrs['gender'])
        'Man'
    """
    df = pd.read_csv(households_file_path)
    person = df[df['perno'] == person_no]
    if person.empty:
        return None
    # Convert to dict and replace NaN with None for JSON serialization
    result = person.iloc[0].to_dict()
    return {k: (None if pd.isna(v) else v) for k, v in result.items()}


def get_trip_history(person_no: str, trip_id: str, trips_file_path: str) -> List[Dict[str, Any]]:
    """
    Retrieve all trips for a person that occurred before the specified trip.

    Args:
        person_no: Unique person identifier (UUID string from 'perno' column)
        trip_id: Current trip ID (from '_id' column) to get history before
        trips_file_path: Path to the trips CSV file

    Returns:
        List of dictionaries, each containing trip attributes, ordered chronologically.
        Returns empty list if person not found or trip_id not found.

    Trip attributes include: _id, perno, data_end_local_dt_year, data_end_local_dt_month,
    data_end_local_dt_day, data_start_local_dt_year, data_start_local_dt_month,
    data_start_local_dt_day, data_duration, data_distance, data_user_input_mode_confirm,
    data_user_input_purpose_confirm

    Example:
        >>> history = get_trip_history('1d292b85-c549-409a-a10d-746e957582a0',
        ...                             '600533265e173ffb99e07630',
        ...                             'inputs/trips.csv')
        >>> print(f"Found {len(history)} previous trips")
        >>> print(history[0]['data_user_input_mode_confirm'])
    """
    df = pd.read_csv(trips_file_path)
    person_trips = df[df['perno'] == person_no].copy()

    if person_trips.empty:
        return []

    person_trips = person_trips.sort_values(
        by=['data_start_local_dt_year', 'data_start_local_dt_month',
            'data_start_local_dt_day', '_id']
    )

    trip_idx = person_trips[person_trips['_id'] == trip_id].index
    if len(trip_idx) == 0:
        return []

    history = person_trips.loc[:trip_idx[0]].iloc[:-1]
    # Convert to dict and replace NaN with None for JSON serialization
    records = history.to_dict('records')
    return [{k: (None if pd.isna(v) else v) for k, v in record.items()} for record in records]


def get_current_trip_attributes(trip_id: str, trips_file_path: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve attributes for the current trip being analyzed, including derived speed/distance features.

    Args:
        trip_id: Current trip ID (from '_id' column)
        trips_file_path: Path to the trips CSV file

    Returns:
        Dictionary containing current trip attributes plus derived features, or None if trip not found.

    Returned fields include:
    - All original trip fields: _id, perno, data_duration, data_distance, date fields, etc.
    - Derived speed analysis:
      * speed_mps: Average speed in meters per second (key mode predictor)
      * speed_kmh: Speed in km/h
      * speed_interpretation: Human-readable speed category with mode suggestions
    - Distance/duration categories for quick reference

    Speed is the most reliable mode indicator:
    - 1-2 m/s → walk
    - 3-6 m/s → bike/bikeshare
    - 4-8 m/s → pilot_ebike/scootershare
    - 5-12 m/s → bus/shared_ride
    - 8-20 m/s → drove_alone/carpool/train

    Example:
        >>> attrs = get_current_trip_attributes('600533265e173ffb99e07630', 'inputs/trips.csv')
        >>> print(f"Speed: {attrs['speed_mps']} m/s - {attrs['speed_interpretation']}")
        >>> print(f"Distance: {attrs['data_distance']}m, Duration: {attrs['data_duration']}s")
    """
    df = pd.read_csv(trips_file_path)
    trip = df[df['_id'] == trip_id]
    if trip.empty:
        return None

    trip_data = trip.iloc[0]
    duration = trip_data['data_duration']
    distance = trip_data['data_distance']

    # Convert to dict and replace NaN with None
    result = trip_data.to_dict()
    result = {k: (None if pd.isna(v) else v) for k, v in result.items()}

    # Add derived speed features
    if pd.notna(duration) and pd.notna(distance) and duration > 0:
        speed_mps = distance / duration
        result['speed_mps'] = round(speed_mps, 2)
        result['speed_kmh'] = round(speed_mps * 3.6, 2)

        # Speed interpretation for mode prediction
        if speed_mps < 1:
            result['speed_interpretation'] = "Very slow (<1 m/s) - likely not_a_trip or stationary"
        elif speed_mps < 2:
            result['speed_interpretation'] = "Walking pace (1-2 m/s)"
        elif speed_mps < 3:
            result['speed_interpretation'] = "Slow (2-3 m/s) - slow walk or congested"
        elif speed_mps < 6:
            result['speed_interpretation'] = "Biking pace (3-6 m/s) - typical bicycle"
        elif speed_mps < 8:
            result['speed_interpretation'] = "Fast bike/E-bike (6-8 m/s) - e-bike, scooter"
        elif speed_mps < 12:
            result['speed_interpretation'] = "Transit/Urban driving (8-12 m/s) - bus, car in traffic"
        elif speed_mps < 20:
            result['speed_interpretation'] = "Highway/Fast transit (12-20 m/s) - car, train"
        else:
            result['speed_interpretation'] = f"Very fast (>{speed_mps:.1f} m/s) - train, highway"
    else:
        result['speed_mps'] = None
        result['speed_kmh'] = None
        result['speed_interpretation'] = "Unknown (missing duration or distance)"

    # Add quick reference categories
    if pd.notna(distance):
        if distance < 100:
            result['distance_category'] = "very_short"
        elif distance < 1000:
            result['distance_category'] = "short"
        elif distance < 5000:
            result['distance_category'] = "medium"
        else:
            result['distance_category'] = "long"

    if pd.notna(duration):
        duration_min = duration / 60
        result['duration_minutes'] = round(duration_min, 1)
        if duration_min < 5:
            result['duration_category'] = "very_short"
        elif duration_min < 15:
            result['duration_category'] = "short"
        elif duration_min < 30:
            result['duration_category'] = "medium"
        else:
            result['duration_category'] = "long"

    return result


def get_trip_features(trip_id: str, trips_file_path: str) -> Optional[Dict[str, Any]]:
    """
    Calculate derived features from trip data to assist with mode and purpose prediction.

    Args:
        trip_id: Current trip ID (from '_id' column)
        trips_file_path: Path to the trips CSV file

    Returns:
        Dictionary containing derived features, or None if trip not found.

    Derived features include:
    - speed_mps: Average speed in meters per second
    - speed_interpretation: Human-readable speed category with mode hints
    - distance_category: "very_short" (<100m), "short" (<1km), "medium" (<5km),
                        "long" (<20km), "very_long" (>20km)
    - duration_category: "very_short" (<5min), "short" (<15min), "medium" (<30min),
                        "long" (<60min), "very_long" (>60min)
    - is_walkable: Boolean indicating if distance/speed suggests walking
    - is_bikeable: Boolean indicating if distance/speed suggests biking

    Speed interpretation guide for mode prediction:
    - Walking: 1-2 m/s (3.6-7.2 km/h)
    - Biking: 3-6 m/s (10.8-21.6 km/h)
    - E-bike/Scooter: 4-8 m/s (14.4-28.8 km/h)
    - Bus/Local transit: 5-12 m/s (18-43.2 km/h)
    - Car/Highway: 8-25 m/s (28.8-90 km/h)
    - Train/Rail: 10-35 m/s (36-126 km/h)

    Example:
        >>> features = get_trip_features('600533265e173ffb99e07630', 'inputs/trips.csv')
        >>> print(f"Speed: {features['speed_mps']:.1f} m/s ({features['speed_interpretation']})")
        >>> print(f"Distance: {features['distance_category']}, Duration: {features['duration_category']}")
    """
    df = pd.read_csv(trips_file_path)
    trip = df[df['_id'] == trip_id]
    if trip.empty:
        return None

    trip_data = trip.iloc[0]
    duration = trip_data['data_duration']  # seconds
    distance = trip_data['data_distance']  # meters

    # Handle missing or zero values
    if pd.isna(duration) or pd.isna(distance) or duration <= 0:
        speed_mps = None
        speed_interpretation = "Unknown (missing duration or distance)"
    else:
        speed_mps = distance / duration

        # Interpret speed for mode prediction
        if speed_mps < 1:
            speed_interpretation = "Very slow (<1 m/s) - likely not_a_trip or stationary"
        elif speed_mps < 2:
            speed_interpretation = "Walking pace (1-2 m/s)"
        elif speed_mps < 3:
            speed_interpretation = "Slow (2-3 m/s) - slow walk or congested"
        elif speed_mps < 6:
            speed_interpretation = "Biking pace (3-6 m/s) - typical bicycle"
        elif speed_mps < 8:
            speed_interpretation = "Fast bike/E-bike (6-8 m/s) - e-bike, scooter"
        elif speed_mps < 12:
            speed_interpretation = "Transit/Urban driving (8-12 m/s) - bus, car in traffic"
        elif speed_mps < 20:
            speed_interpretation = "Highway/Fast transit (12-20 m/s) - car, train"
        else:
            speed_interpretation = f"Very fast (>{speed_mps:.1f} m/s) - train, highway driving"

    # Distance categories
    if distance < 100:
        distance_category = "very_short"
    elif distance < 1000:
        distance_category = "short"
    elif distance < 5000:
        distance_category = "medium"
    elif distance < 20000:
        distance_category = "long"
    else:
        distance_category = "very_long"

    # Duration categories (convert seconds to minutes)
    duration_minutes = duration / 60 if not pd.isna(duration) else None
    if duration_minutes is None:
        duration_category = "unknown"
    elif duration_minutes < 5:
        duration_category = "very_short"
    elif duration_minutes < 15:
        duration_category = "short"
    elif duration_minutes < 30:
        duration_category = "medium"
    elif duration_minutes < 60:
        duration_category = "long"
    else:
        duration_category = "very_long"

    # Derived booleans
    is_walkable = distance < 2000 and (speed_mps is None or speed_mps < 3)
    is_bikeable = 500 < distance < 15000 and (speed_mps is None or 2 < speed_mps < 8)

    return {
        'speed_mps': round(speed_mps, 2) if speed_mps is not None else None,
        'speed_kmh': round(speed_mps * 3.6, 2) if speed_mps is not None else None,
        'speed_interpretation': speed_interpretation,
        'distance_meters': distance,
        'distance_km': round(distance / 1000, 2),
        'distance_category': distance_category,
        'duration_seconds': duration,
        'duration_minutes': round(duration_minutes, 1) if duration_minutes is not None else None,
        'duration_category': duration_category,
        'is_walkable': is_walkable,
        'is_bikeable': is_bikeable,
    }
