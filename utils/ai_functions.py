from pydantic_ai import Agent
from pydantic import BaseModel
import asyncio

async def call_ai_async(model: str, system_prompt: str, user_prompt: str, result_schema: BaseModel) -> any:
    agent = Agent(
        model=model,
        system_prompt=system_prompt,
        result_type=result_schema
    )
    result = await agent.run(user_prompt)
    return result.output

def check_input_token_length(input_text: str, max_length: int) -> bool:
    # Rough estimation: ~4 characters per token for English text
    estimated_tokens = len(input_text) / 4
    return estimated_tokens <= max_length





