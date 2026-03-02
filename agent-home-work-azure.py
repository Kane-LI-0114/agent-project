"""
Interactive homework helper CLI using Azure OpenAI (history/philosophy) and DeepSeek (math).
Loads credentials from .env via python-dotenv. Guardrail checks ensure prompts look like homework.
"""

import asyncio
import os
import re
from typing import Dict

from agents import (
    Agent,
    GuardrailFunctionOutput,
    InputGuardrail,
    OpenAIChatCompletionsModel,
    Runner,
    set_tracing_disabled,
)
from agents.exceptions import InputGuardrailTripwireTriggered
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI, AsyncOpenAI
from pydantic import BaseModel


# --------------------------------------------------------------------------- #
# Environment / settings
# --------------------------------------------------------------------------- #

load_dotenv()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY", "")

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL_NAME = "deepseek-chat"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Demo shortcuts that read like homework prompts to satisfy the guardrail
DEMOS: Dict[str, str] = {
    "demo-history": (
        "This is for my history homework: Who was the first president of the United States? "
        "Give a concise answer with one key fact."
    ),
    "demo-life": (
        "What is the meaning of life?"
    ),
    "demo-math": (
        "For my algebra homework, solve 2x + 3 = 15 for x. Show the steps clearly."
    ),
}


# --------------------------------------------------------------------------- #
# Model builders
# --------------------------------------------------------------------------- #

def build_azure_model() -> OpenAIChatCompletionsModel:
    """Create an Azure OpenAI chat model from env settings."""
    if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_KEY:
        raise RuntimeError(
            "Azure OpenAI not configured. Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_KEY."
        )

    match = re.match(
        r"(https?://[^/]+)/openai/deployments/([^/]+)/.*api-version=([^&]+)",
        AZURE_OPENAI_ENDPOINT,
    )
    if not match:
        raise ValueError(
            "AZURE_OPENAI_ENDPOINT must include deployment name and api-version, "
            "e.g. https://<resource>.openai.azure.com/openai/deployments/<deployment>/..."
        )

    azure_endpoint = match.group(1)
    deployment_name = match.group(2)
    api_version = match.group(3)

    azure_client = AsyncAzureOpenAI(
        api_key=AZURE_OPENAI_KEY,
        azure_endpoint=azure_endpoint,
        api_version=api_version,
    )
    return OpenAIChatCompletionsModel(
        openai_client=azure_client,
        model=deployment_name,
    )


def build_deepseek_model() -> OpenAIChatCompletionsModel:
    """DeepSeek model used for math tutoring."""
    return OpenAIChatCompletionsModel(
        openai_client=AsyncOpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
        ),
        model=DEEPSEEK_MODEL_NAME,
    )


# --------------------------------------------------------------------------- #
# Guardrail schema and agent
# --------------------------------------------------------------------------- #

class HomeworkOutput(BaseModel):
    is_homework: bool
    reasoning: str


def build_guardrail_agent(model: OpenAIChatCompletionsModel) -> Agent:
    return Agent(
        name="Guardrail check",
        instructions=(
            "Decide if the user is asking about homework. "
            "Return is_homework=true only when the query clearly requests help with a school or "
            "class assignment (history, math, etc.)."
        ),
        output_type=HomeworkOutput,
        model=model,
    )


async def homework_guardrail(ctx, _agent, input_data):
    result = await Runner.run(guardrail_agent, input_data, context=ctx.context)
    final_output = result.final_output_as(HomeworkOutput)
    return GuardrailFunctionOutput(
        output_info=final_output,
        tripwire_triggered=not final_output.is_homework,
    )


# --------------------------------------------------------------------------- #
# Specialist agents
# --------------------------------------------------------------------------- #

azure_model = build_azure_model()
deepseek_model = build_deepseek_model()
guardrail_agent = build_guardrail_agent(azure_model)

math_tutor_agent = Agent(
    name="Math Tutor",
    handoff_description="Specialist agent for math questions",
    instructions="You help with math homework. Show reasoning step-by-step and keep it concise.",
    model=deepseek_model,
)

history_tutor_agent = Agent(
    name="History Tutor",
    handoff_description="Specialist agent for historical questions",
    instructions="You help with history homework. Answer clearly and concisely.",
    model=azure_model,
)

# Triage Agent uses Azure by default, hands off to specialists, and enforces guardrail
triage_agent = Agent(
    name="Triage Agent",
    instructions="You determine which agent should answer this homework question.",
    model=azure_model,
    handoffs=[history_tutor_agent, math_tutor_agent],
    input_guardrails=[
        InputGuardrail(guardrail_function=homework_guardrail),
    ],
)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def print_header() -> None:
    print("--- Homework Helper CLI (Azure + DeepSeek) ---")
    print("Type a homework question and press Enter. Type 'exit' or 'quit' to stop.")
    print("Commands:")
    print("  demo-history -> History homework: Who was the first president of the United States?")
    print("  demo-math    -> Algebra homework: Solve 2x + 3 = 15 for x and show your steps.")
    print("  demo-life    -> What is the meaning of life?")


async def main() -> None:
    set_tracing_disabled(True)
    print_header()

    session = None

    while True:
        user_input = input("User: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Exiting chat. Bye!")
            break
        if user_input.lower() in DEMOS:
            user_input = DEMOS[user_input.lower()]
            print(f"[INFO] Running demo: {user_input}")

        try:
            result = await Runner.run(
                triage_agent,
                user_input,
                session=session,
            )
            print(f"Assistant: {result.final_output}\n")
        except InputGuardrailTripwireTriggered as e:
            print(f"Guardrail blocked this input: {e}\n")
        except (RuntimeError, ValueError, OSError) as exc:
            print(f"[ERROR] Agent run failed: {exc}\n")


if __name__ == "__main__":
    asyncio.run(main())
