"""Main module for Survey Helper Agent."""
import os
import warnings
import logging

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner

from .config import config
from .adapters import TDSCAdapter, NHTSAdapter

# Configure logging and warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)
logging.getLogger('asyncio').setLevel(logging.CRITICAL)

# Suppress LiteLLM logging to prevent event loop conflicts
import litellm
litellm.suppress_debug_info = True
litellm.set_verbose = False
litellm.success_callback = []
litellm.failure_callback = []


def create_bedrock_model() -> LiteLlm:
    """Create and configure Bedrock LiteLLM model with AWS credentials."""
    model_id = f"bedrock/{config.bedrock_model_id}"

    if config.aws_profile:
        os.environ["AWS_PROFILE"] = config.aws_profile
        print(f"Using AWS profile: {config.aws_profile}")
        return LiteLlm(
            model=model_id,
            aws_profile_name=config.aws_profile,
            aws_region_name=config.aws_region
        )

    # Use direct credentials
    os.environ["AWS_ACCESS_KEY_ID"] = config.aws_access_key_id
    os.environ["AWS_SECRET_ACCESS_KEY"] = config.aws_secret_access_key
    os.environ["AWS_REGION_NAME"] = config.aws_region
    print("Using AWS direct credentials")
    return LiteLlm(
        model=model_id,
        aws_access_key_id=config.aws_access_key_id,
        aws_secret_access_key=config.aws_secret_access_key,
        aws_region_name=config.aws_region
    )


def create_adapter():
    """Create the appropriate dataset adapter based on configuration."""
    dataset_type = config.dataset_type
    dataset_config = config.get_dataset_config()

    if dataset_type == 'tdsc':
        adapter = TDSCAdapter(dataset_config)
        print(f"Using TDSC dataset adapter")
    elif dataset_type == 'nhts':
        adapter = NHTSAdapter(dataset_config)
        print(f"Using NHTS dataset adapter")
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    return adapter


def create_tools(adapter) -> list:
    """Create agent tools from the dataset adapter."""
    tools = [
        adapter.get_person_attributes,
        adapter.get_trip_history,
        adapter.get_current_trip_attributes
    ]

    return tools


def create_agent(model: LiteLlm, tools: list, adapter) -> Agent:
    """Create the survey prediction agent with dataset-specific guidance."""
    # Get dataset-specific guidance from the adapter
    dataset_guidance = adapter.get_prediction_guidance()

    instruction = f"""You are a trip mode and purpose predictor. Use available tools to analyze trip data, then predict:

1. **Predicted Mode**: From {config.get_mode_options_str()}
2. **Predicted Purpose**: From {config.get_purpose_options_str()}

{dataset_guidance}

GENERAL APPROACH:
- Use get_person_attributes() to understand demographics and constraints
- Use get_trip_history() to identify patterns and recurring trips
- Use get_current_trip_attributes() for trip-specific features
- Keep predictions concise
"""

    agent = Agent(
        name="mode_purpose_predictor_v1",
        model=model,
        description="Predicts the mode and purpose of a current trip.",
        instruction=instruction,
        tools=tools
    )

    print(f"Agent '{agent.name}' created using Claude via AWS Bedrock (Model: {config.bedrock_model_id}).")
    return agent


# Initialize components
bedrock_model = create_bedrock_model()
adapter = create_adapter()
tools = create_tools(adapter)
survey_agent = create_agent(bedrock_model, tools, adapter)

# Initialize session service and runner
session_service = InMemorySessionService()
runner = Runner(
    app_name="travel_survey_helper",
    agent=survey_agent,
    session_service=session_service
)
