"""TDSC (Transportation Data Science Center) dataset adapter."""
import pandas as pd
from typing import Optional, Dict, Any, List
from .base import BaseDatasetAdapter


class TDSCAdapter(BaseDatasetAdapter):
    """
    Adapter for TDSC survey data.

    File structure:
    - households.csv: Contains person attributes (demographics, employment, household info)
    - trips.csv: Contains trip records with mode, purpose, duration, distance

    Key columns:
    - Person ID: 'perno' (UUID)
    - Trip ID: '_id'
    - Mode: 'data_user_input_mode_confirm'
    - Purpose: 'data_user_input_purpose_confirm'
    """

    def _validate_config(self) -> None:
        """Validate TDSC-specific configuration."""
        required = ['HOUSEHOLDS_FILE', 'TRIPS_FILE', 'mode_options', 'purpose_options']
        for field in required:
            if field not in self.config:
                raise ValueError(f"TDSC adapter missing required field: {field}")

    def get_person_attributes(self, person_identifier: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve all attributes for a specific person from the households CSV file.

        Args:
            person_identifier: Person identifier (UUID string from 'perno' column).
                              TDSC uses globally unique person IDs.

        Returns:
            Dictionary containing all person attributes, or None if person not found.

        Available attributes include: gender, have_driver_license, student, education_level_attained,
        work_for_pay_or_profit, number_of_paid_jobs, primary_job_is_full_time,
        primary_job_classification, mode_used_to_get_to_work, how_long_was_your_commute,
        flexible_work_schedule_at_primary_job, remote_or_other_work_location_allowed,
        number_of_days_remote_or_other_work_location, employment_description, own_or_rent,
        housing_type, household_income, household_size, number_of_children_under_18,
        number_of_hh_member_with_driver_license, number_of_household_vehicles,
        alternative_mode_if_a_car_is_not_available, disability, length_of_disability,
        age_calculated_nrel_agebin
        """
        df = pd.read_csv(self.config['HOUSEHOLDS_FILE'])
        person = df[df['perno'] == person_identifier]
        if person.empty:
            return None
        # Convert to dict and replace NaN with None for JSON serialization
        result = person.iloc[0].to_dict()
        return {k: (None if pd.isna(v) else v) for k, v in result.items()}

    def get_trip_history(self, person_identifier: str, trip_identifier: str) -> List[Dict[str, Any]]:
        """
        Retrieve all trips for a person that occurred before the specified trip.

        Args:
            person_identifier: Person identifier (UUID string from 'perno' column)
            trip_identifier: Trip identifier (from '_id' column)

        Returns:
            List of dictionaries containing trip attributes, ordered chronologically.
            Returns empty list if person or trip not found.

        Trip attributes include: _id, perno, data_end_local_dt_year, data_end_local_dt_month,
        data_end_local_dt_day, data_start_local_dt_year, data_start_local_dt_month,
        data_start_local_dt_day, data_duration, data_distance, data_user_input_mode_confirm,
        data_user_input_purpose_confirm
        """
        df = pd.read_csv(self.config['TRIPS_FILE'])
        person_trips = df[df['perno'] == person_identifier].copy()

        if person_trips.empty:
            return []

        person_trips = person_trips.sort_values(
            by=['data_start_local_dt_year', 'data_start_local_dt_month',
                'data_start_local_dt_day', '_id']
        )

        trip_idx = person_trips[person_trips['_id'] == trip_identifier].index
        if len(trip_idx) == 0:
            return []

        history = person_trips.loc[:trip_idx[0]].iloc[:-1]
        # Convert to dict and replace NaN with None for JSON serialization
        records = history.to_dict('records')
        return [{k: (None if pd.isna(v) else v) for k, v in record.items()} for record in records]

    def get_current_trip_attributes(self, trip_identifier: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve attributes for the current trip, including derived speed/distance features.

        Args:
            trip_identifier: Trip identifier (from '_id' column)

        Returns:
            Dictionary containing trip attributes plus derived features:
            - speed_mps: Average speed in meters per second
            - speed_kmh: Speed in km/h
            - speed_interpretation: Human-readable speed category with mode suggestions
            - distance_category: Quick reference distance category
            - duration_category: Quick reference duration category
            Or None if trip not found.

        Speed is the most reliable mode indicator:
        - 1-2 m/s → walk
        - 3-6 m/s → bike/bikeshare
        - 4-8 m/s → pilot_ebike/scootershare
        - 5-12 m/s → bus/shared_ride
        - 8-20 m/s → drove_alone/carpool/train
        """
        df = pd.read_csv(self.config['TRIPS_FILE'])
        trip = df[df['_id'] == trip_identifier]
        if trip.empty:
            return None

        trip_data = trip.iloc[0]
        duration = trip_data['data_duration']
        distance = trip_data['data_distance']

        # Convert to dict and replace NaN with None
        result = trip_data.to_dict()
        result = {k: (None if pd.isna(v) else v) for k, v in result.items()}

        # IMPORTANT: Exclude ground truth labels to prevent data leakage
        result.pop('data_user_input_mode_confirm', None)
        result.pop('data_user_input_purpose_confirm', None)

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

    def get_mode_options(self) -> List[str]:
        """Get valid transportation mode options for TDSC dataset."""
        return self.config['mode_options']

    def get_purpose_options(self) -> List[str]:
        """Get valid trip purpose options for TDSC dataset."""
        return self.config['purpose_options']

    def get_prediction_guidance(self) -> str:
        """Get TDSC-specific prediction guidance."""
        return """KEY GUIDANCE FOR MODE:
- speed_mps is the best mode indicator: 1-2=walk, 3-6=bike, 4-8=ebike, 5-12=bus, 8-20=car/train
- Check speed_interpretation field for human-readable category
- Soft constraints: No license → unlikely drive alone; No vehicles → unlikely drove_alone

KEY GUIDANCE FOR PURPOSE:
- Use trip history for recurring patterns (work commute patterns, regular shopping times)
- Person attributes: Check if person is a worker, student, or retiree
- TDSC data includes detailed trip timing (data_start_local_dt_hour, data_end_local_dt_hour)
- Consider distance + duration combination with historical patterns"""
