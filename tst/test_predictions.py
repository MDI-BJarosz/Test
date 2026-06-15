import asyncio
import sys
from pathlib import Path
from google.genai import types

# Add parent directory to path to import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.main import runner, session_service


async def predict_trip_mode_purpose(person_no: str, trip_id: str) -> None:
    """
    Query the agent to predict mode and purpose for a specific trip.

    Args:
        person_no: Person UUID from households.csv
        trip_id: Trip ID from trips.csv
    """
    query = (
        f"Predict the transportation mode and trip purpose "
        f"for person {person_no} and trip {trip_id}."
    )

    try:
        print(f"\n{'='*80}")
        print(f"QUERYING AGENT")
        print(f"Person: {person_no}")
        print(f"Trip: {trip_id}")
        print(f"{'='*80}\n")

        # Create a session for this query
        session_id = f"{person_no}_{trip_id}"
        session_service.create_session_sync(
            app_name="travel_survey_helper",
            user_id="test_user",
            session_id=session_id
        )

        # Create message content
        message = types.Content(
            role="user",
            parts=[types.Part(text=query)]
        )

        # Send query to agent
        events = runner.run(
            user_id="test_user",
            session_id=session_id,
            new_message=message
        )

        # Process events to get the response
        print("AGENT RESPONSE:")
        for event in events:
            if hasattr(event, 'content'):
                for part in event.content.parts:
                    if hasattr(part, 'text'):
                        print(part.text)
        print()

    except Exception as e:
        print(f"ERROR: {e}\n")
        import traceback
        traceback.print_exc()


async def main():
    """
    Run test cases with diverse trip scenarios.
    """
    print("\n" + "="*80)
    print("TRAVEL SURVEY AGENT - TEST SUITE")
    print("="*80)

    # Test Case 1: Multi-modal transit user (skateboard/train/bus)
    # No license, no vehicles, bicycle commuter, 11 previous trips
    print("\n[TEST CASE 1: Multi-modal transit user]")
    await predict_trip_mode_purpose(
        person_no="1d292b85-c549-409a-a10d-746e957582a0",
        trip_id="600533265e173ffb99e07630"
    )

    # Test Case 2: Licensed cyclist with extensive history
    # Has license, 1 vehicle, bicycle commuter, 50 previous trips available
    print("\n[TEST CASE 2: Licensed cyclist with extensive history]")
    await predict_trip_mode_purpose(
        person_no="960835ac-9d8a-421d-8b8a-bf816f8a4b92",
        trip_id="60053362db3d82eed967c0d0"
    )

    # Test Case 3: Public transit/ebike user (edge case: unhoused)
    # Has license, no vehicles, public bus commuter, 30 previous trips
    print("\n[TEST CASE 3: Public transit/ebike user (unhoused)]")
    await predict_trip_mode_purpose(
        person_no="898b1a5e-cdd4-4a0c-90e4-942fa298e456",
        trip_id="6005338edb3d82eed967c2e4"
    )

    print("\n" + "="*80)
    print("TEST SUITE COMPLETED")
    print("="*80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
