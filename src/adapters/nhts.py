"""NHTS (National Household Travel Survey) dataset adapter."""
import pandas as pd
from typing import Optional, Dict, Any, List
from .base import BaseDatasetAdapter


class NHTSAdapter(BaseDatasetAdapter):
    """
    Adapter for NHTS (National Household Travel Survey) data.

    File structure:
    - household.csv: Contains household-level attributes
    - person.csv: Contains person demographics and characteristics
    - trip.csv: Contains trip records with mode, purpose, duration, distance
    - vehicle.csv: Contains vehicle information (optional)

    Key columns:
    - Person ID: 'person_id'
    - Trip ID: 'trip_id'
    - Household ID: 'household_id'
    - Mode: 'trip_mode'
    - Purpose: 'trip_purpose'
    - Duration: 'travel_time_min' (minutes)
    - Distance: 'trip_distance_mi' (miles)
    """

    def _validate_config(self) -> None:
        """Validate NHTS-specific configuration."""
        required = ['HOUSEHOLD_FILE', 'PERSON_FILE', 'TRIP_FILE', 'mode_options', 'purpose_options']
        for field in required:
            if field not in self.config:
                raise ValueError(f"NHTS adapter missing required field: {field}")

    def get_person_attributes(self, person_identifier: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve all attributes for a specific person by joining person and household data.

        Args:
            person_identifier: Composite identifier in format "household_id:person_id" (e.g., "9000013761:1").
                              Required because person_id is not globally unique in NHTS.

        Returns:
            Dictionary containing merged person and household attributes, or None if not found.

        Combines attributes from:
        - person.csv: is_driver, education_level, age, sex, race, hispanic, is_worker, work_location, etc.
        - household.csv: hh_size, vehicle_count, hh_income, home_ownership, bg_urban_rural, hh_lifecycle, etc.

        Note: In NHTS, person_id is only unique within a household. This method requires composite
        identifiers in the format "household_id:person_id".
        """
        # Parse composite identifier - required format: "household_id:person_id"
        if ':' not in person_identifier:
            raise ValueError(f"NHTS requires composite person_identifier in format 'household_id:person_id', got: {person_identifier}")

        household_id_str, person_id = person_identifier.split(':', 1)
        household_id = int(household_id_str)

        # Load person data
        person_df = pd.read_csv(self.config['PERSON_FILE'])

        # Filter by both household_id and person_id for unique identification
        person_row = person_df[
            (person_df['household_id'] == household_id) &
            (person_df['person_id'].astype(str) == str(person_id))
        ]

        if person_row.empty:
            return None

        # Load household data
        household_df = pd.read_csv(self.config['HOUSEHOLD_FILE'])
        household_row = household_df[household_df['household_id'] == household_id]

        # Merge person and household attributes
        result = person_row.iloc[0].to_dict()

        if not household_row.empty:
            household_attrs = household_row.iloc[0].to_dict()
            # Add household attributes with 'hh_' prefix to avoid collisions
            for k, v in household_attrs.items():
                if k != 'household_id':  # Skip duplicate household_id
                    result[f'hh_{k}'] = None if pd.isna(v) else v

        # Replace NaN with None for JSON serialization
        return {k: (None if pd.isna(v) else v) for k, v in result.items()}

    def get_trip_history(self, person_identifier: str, trip_identifier: str) -> List[Dict[str, Any]]:
        """
        Retrieve all trips for a person that occurred before the specified trip.

        Args:
            person_identifier: Composite identifier in format "household_id:person_id" (e.g., "9000013761:1")
            trip_identifier: Composite identifier in format "household_id:person_id:trip_id" (e.g., "9000013761:1:3")

        Returns:
            List of dictionaries containing trip attributes, ordered chronologically.
            Returns empty list if person or trip not found.

        Trip attributes include: trip_id, person_id, household_id, travel_time_min, trip_distance_mi,
        trip_mode, trip_purpose, start_time, seq_trip_id (sequence number)

        Note: In NHTS, neither person_id nor trip_id is globally unique. The unique identifier is
        (household_id, person_id, trip_id). This method requires composite identifiers.
        """
        df = pd.read_csv(self.config['TRIP_FILE'])

        # Parse trip composite identifier - required format: "household_id:person_id:trip_id"
        if ':' not in trip_identifier:
            raise ValueError(f"NHTS requires composite trip_identifier in format 'household_id:person_id:trip_id', got: {trip_identifier}")

        trip_parts = trip_identifier.split(':')
        if len(trip_parts) != 3:
            raise ValueError(f"NHTS trip_identifier must have 3 parts (household:person:trip), got {len(trip_parts)} parts: {trip_identifier}")

        household_id = int(trip_parts[0])
        person_id = trip_parts[1]
        trip_id = trip_parts[2]

        # Filter by household_id, person_id, AND trip_id for unique trip identification
        current_trip = df[
            (df['household_id'] == household_id) &
            (df['person_id'].astype(str) == str(person_id)) &
            (df['trip_id'].astype(str) == str(trip_id))
        ]

        if current_trip.empty:
            return []

        # Get all trips for this specific person in this household
        person_trips = df[
            (df['household_id'] == household_id) &
            (df['person_id'].astype(str) == str(person_id))
        ].copy()

        if person_trips.empty:
            return []

        # Sort by sequence ID (seq_trip_id is the chronological order within a day)
        person_trips = person_trips.sort_values(by=['seq_trip_id'])

        # Find the current trip index
        trip_idx = person_trips[
            (person_trips['trip_id'].astype(str) == str(trip_id))
        ].index

        if len(trip_idx) == 0:
            return []

        # Get all trips before the current one
        history = person_trips.loc[:trip_idx[0]].iloc[:-1]

        # Convert to dict and replace NaN with None
        records = history.to_dict('records')
        return [{k: (None if pd.isna(v) else v) for k, v in record.items()} for record in records]

    def get_current_trip_attributes(self, trip_identifier: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve attributes for the current trip, including derived speed/distance features.

        Args:
            trip_identifier: Composite identifier in format "household_id:person_id:trip_id" (e.g., "9000013761:1:3")

        Returns:
            Dictionary containing trip attributes plus derived features:
            - speed_mps: Average speed in meters per second
            - speed_kmh: Speed in km/h
            - speed_interpretation: Human-readable speed category with mode suggestions
            - distance_category: Quick reference distance category
            - duration_category: Quick reference duration category
            - start_time_hour: Hour of day trip started (0-23)
            - start_time_interpretation: Human-readable time category with purpose hints
            - dwell_time_category: Category for time spent at destination
            - dwell_time_interpretation: What dwell time suggests about purpose
            Or None if trip not found.

        Note: NHTS uses miles for distance (trip_distance_mi) and minutes for duration (travel_time_min).
        These are converted to meters and seconds for consistency with TDSC.
        """
        df = pd.read_csv(self.config['TRIP_FILE'])

        # Parse composite identifier - required format: "household_id:person_id:trip_id"
        if ':' not in trip_identifier:
            raise ValueError(f"NHTS requires composite trip_identifier in format 'household_id:person_id:trip_id', got: {trip_identifier}")

        trip_parts = trip_identifier.split(':')
        if len(trip_parts) != 3:
            raise ValueError(f"NHTS trip_identifier must have 3 parts (household:person:trip), got {len(trip_parts)} parts: {trip_identifier}")

        household_id = int(trip_parts[0])
        person_id = trip_parts[1]
        trip_id = trip_parts[2]

        # Filter by household_id, person_id, AND trip_id for unique trip identification
        trip = df[
            (df['household_id'] == household_id) &
            (df['person_id'].astype(str) == str(person_id)) &
            (df['trip_id'].astype(str) == str(trip_id))
        ]

        if trip.empty:
            return None

        trip_data = trip.iloc[0]

        # NHTS uses minutes for duration and miles for distance
        duration_minutes = trip_data.get('travel_time_min')
        distance_miles = trip_data.get('trip_distance_mi')

        # Convert to dict and replace NaN with None
        result = trip_data.to_dict()
        result = {k: (None if pd.isna(v) else v) for k, v in result.items()}

        # IMPORTANT: Exclude ground truth labels to prevent data leakage
        result.pop('trip_mode', None)
        result.pop('trip_purpose', None)

        # Convert NHTS units to standard units (seconds and meters)
        duration_seconds = None
        distance_meters = None

        if pd.notna(duration_minutes):
            duration_seconds = duration_minutes * 60
            result['data_duration'] = duration_seconds  # Add standardized field

        if pd.notna(distance_miles):
            distance_meters = distance_miles * 1609.34  # miles to meters
            result['data_distance'] = distance_meters  # Add standardized field

        # Add derived speed features
        if duration_seconds and distance_meters and duration_seconds > 0:
            speed_mps = distance_meters / duration_seconds
            result['speed_mps'] = round(speed_mps, 2)
            result['speed_kmh'] = round(speed_mps * 3.6, 2)

            # Speed interpretation for mode prediction (same logic as TDSC)
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

        # Add quick reference categories (using converted meters)
        if distance_meters:
            if distance_meters < 100:
                result['distance_category'] = "very_short"
            elif distance_meters < 1000:
                result['distance_category'] = "short"
            elif distance_meters < 5000:
                result['distance_category'] = "medium"
            else:
                result['distance_category'] = "long"

        if duration_minutes:
            result['duration_minutes'] = round(duration_minutes, 1)
            if duration_minutes < 5:
                result['duration_category'] = "very_short"
            elif duration_minutes < 15:
                result['duration_category'] = "short"
            elif duration_minutes < 30:
                result['duration_category'] = "medium"
            else:
                result['duration_category'] = "long"

        # Add temporal features for purpose prediction
        start_time = trip_data.get('start_time')
        if pd.notna(start_time):
            # Convert HHMM format to hour (e.g., 930 -> 9, 1830 -> 18)
            start_time_int = int(start_time)
            hour = start_time_int // 100
            minute = start_time_int % 100
            result['start_time_hour'] = hour
            result['start_time_minute'] = minute
            result['start_time_formatted'] = f"{hour:02d}:{minute:02d}"

            # Interpret time of day for purpose prediction
            if 6 <= hour < 9:
                result['start_time_interpretation'] = "Early morning (6-9 AM) - likely commute to Work or School"
            elif 9 <= hour < 12:
                result['start_time_interpretation'] = "Mid-morning (9 AM-12 PM) - likely Shopping/Errands, Medical, or Work-related"
            elif 12 <= hour < 14:
                result['start_time_interpretation'] = "Lunch time (12-2 PM) - likely Meals or lunch break"
            elif 14 <= hour < 17:
                result['start_time_interpretation'] = "Afternoon (2-5 PM) - likely Shopping/Errands, School pickup, or Work-related"
            elif 17 <= hour < 20:
                result['start_time_interpretation'] = "Evening (5-8 PM) - likely commute Home, Meals, or Social/Recreational"
            elif 20 <= hour < 24:
                result['start_time_interpretation'] = "Late evening (8 PM-midnight) - likely Home, Social/Recreational, or Entertainment"
            else:  # 0-6 AM
                result['start_time_interpretation'] = "Night/Early morning (midnight-6 AM) - unusual, possibly shift work or special occasion"

        # Add dwell time interpretation for destination purpose
        dwell_time = trip_data.get('dwell_time_min')
        if pd.notna(dwell_time):
            result['dwell_time_minutes'] = round(dwell_time, 1)

            if dwell_time < 15:
                result['dwell_time_category'] = "very_short"
                result['dwell_time_interpretation'] = "Very short stay (<15 min) - likely quick Shopping/Errands, pickup/dropoff (Transport someone), or transit transfer"
            elif dwell_time < 60:
                result['dwell_time_category'] = "short"
                result['dwell_time_interpretation'] = "Short stay (15-60 min) - likely Shopping/Errands, Meals, brief Social visit, or Medical appointment"
            elif dwell_time < 240:
                result['dwell_time_category'] = "medium"
                result['dwell_time_interpretation'] = "Medium stay (1-4 hours) - likely longer Shopping, Social/Recreational, extended Meals, or partial work day"
            elif dwell_time < 480:
                result['dwell_time_category'] = "long"
                result['dwell_time_interpretation'] = "Long stay (4-8 hours) - likely Work, School, or all-day Social/Recreational event"
            else:
                result['dwell_time_category'] = "very_long"
                result['dwell_time_interpretation'] = "Very long stay (8+ hours) - very likely full Work day or School day, or overnight stay at Home"

        return result

    def get_mode_options(self) -> List[str]:
        """Get valid transportation mode options for NHTS dataset."""
        return self.config['mode_options']

    def get_purpose_options(self) -> List[str]:
        """Get valid trip purpose options for NHTS dataset."""
        return self.config['purpose_options']

    def get_prediction_guidance(self) -> str:
        """Get NHTS-specific prediction guidance."""
        return """KEY GUIDANCE FOR MODE:
- speed_mps is the best mode indicator: 1-2=walk, 3-6=bike, 4-8=ebike, 5-12=bus, 8-20=car/train
- Check speed_interpretation field for human-readable category
- Soft constraints: No license (is_driver field) → unlikely drive alone; No vehicles (hh_vehicle_count) → unlikely drove_alone

KEY GUIDANCE FOR PURPOSE:
- Person attributes: Check is_worker status first (Worker→look for work commutes, Not worker→no work trips expected)
- Use trip history for recurring patterns (regular commute times, typical shopping/errands patterns)
- Temporal context: start_time and dwell_time provide helpful hints when combined with person context
- Consider distance + duration + time-of-day together with person employment status"""
