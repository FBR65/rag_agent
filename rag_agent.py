"""
Licence and Copyright

The use of this code or parts of it is permitted exclusively for private,
educational, or non-commercial purposes.
Any commercial use or use by governmental organizations is prohibited without prior
written permission from the author.

Copyright 2025 Frank Reis
"""

import os
from dataclasses import dataclass
from typing import Optional, List, Dict
import logging
import httpx
import traceback
import random
import time

import pydantic
import pydantic_ai
from pydantic import BaseModel, Field, ValidationError

from pydantic_ai import Agent, RunContext
import dotenv
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from data_retrieve import Hybrid_search, ScoredPoint
from pydantic_ai.exceptions import (
    UsageLimitExceeded,
    ModelHTTPError,
    UnexpectedModelBehavior,
    FallbackExceptionGroup,
    ModelRetry,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
dotenv.load_dotenv()

provider = OpenAIProvider(base_url=os.getenv("BASE_URL"), api_key=os.getenv("API_KEY"))

model = OpenAIModel(
    provider=provider,
    model_name=os.getenv("MODEL_NAME"),
)


# --- Dependencies ---
@dataclass
class RetrieveDependencies:
    hybrid_search: Hybrid_search


@dataclass
class ProblemDependencies:
    pass


@dataclass
class PreAnswerDependencies:
    pass


@dataclass
class ValidatorDependencies:
    pass


@dataclass
class OrchestratorDependencies:
    pass


# --- Result Models ---
class RetrieveResult(BaseModel):
    results: List[Dict] = Field(description="Results from the hybrid search")


class ProblemResult(BaseModel):
    query: str = Field(description="Query for the retrieve agent")


class PreAnswerResult(BaseModel):
    answer: str = Field(description="Answer to the user query")
    sources: List[str] = Field(description="Sources used to generate the answer")


class ValidatorResult(BaseModel):
    is_valid: bool = Field(description="Whether the answer is valid")
    feedback: Optional[str] = Field(description="Feedback for the pre-answer agent")


class OrchestratorResult(BaseModel):
    question: str = Field(description="The original question")
    answer: str = Field(description="The final answer in markdown format")
    sources: List[str] = Field(description="Sources used to generate the answer")


# --- Agents ---
# Retrieve agent
collection_name_prefix = "test_collection"
hybrid_search = Hybrid_search(collection_name_prefix)

retrieve_agent = Agent(
    model,
    deps_type=RetrieveDependencies,
    result_type=RetrieveResult,
    system_prompt="You are a data retrieval agent. You receive a query and return relevant data.",
)


@retrieve_agent.tool
async def retrieve_data(
    ctx: RunContext[RetrieveDependencies], query: str, limit: int = 5
) -> RetrieveResult:
    logger.info(f"retrieve_data: query={query}, limit={limit}")
    try:
        # Simulate a temporary network issue (e.g., 10% chance of failure)
        if random.random() < 0.1:
            raise ModelRetry(message="Temporary network issue. Please retry.")

        hybrid_results = ctx.deps.hybrid_search.query_hybrid_search(query, limit=limit)
        deduplicated_results = hybrid_results["deduplicated_combined_results"]

        # Convert ScoredPoint objects to dictionaries
        scored_points_dicts = [point.model_dump() for point in deduplicated_results]

        logger.info(f"retrieve_data: deduplicated_results={scored_points_dicts}")
        return RetrieveResult(results=scored_points_dicts)
    except KeyError as e:
        logger.error(f"KeyError in retrieve_data: {e}")
        logger.error(traceback.format_exc())
        raise
    except ModelRetry as e:
        logger.warning(f"ModelRetry in retrieve_data: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred in retrieve_data: {e}")
        logger.error(traceback.format_exc())
        raise


# Problem agent
problem_agent = Agent(
    model,
    deps_type=ProblemDependencies,
    result_type=ProblemResult,
    system_prompt="You are a problem analysis agent. You receive a user query and formulate a precise search query for the retrieval agent.",
)


@problem_agent.tool
async def analyze_problem(
    ctx: RunContext[ProblemDependencies], user_query: str
) -> ProblemResult:
    logger.info(f"analyze_problem: user_query={user_query}")
    return ProblemResult(query=user_query)


# Pre-answer agent
pre_answer_agent = Agent(
    model,
    deps_type=PreAnswerDependencies,
    result_type=PreAnswerResult,
    system_prompt="You are a pre-answer agent. You receive a query and data, and you formulate an answer only based on the given data.",
)


@pre_answer_agent.tool
async def formulate_answer(
    ctx: RunContext[PreAnswerDependencies], query: str, data: List[ScoredPoint]
) -> PreAnswerResult:
    logger.info(f"formulate_answer: query={query}, data={data}")
    sources: List[str] = []
    answer: str = ""
    try:
        for result in data:
            if "text" in result.payload:
                sources.append(result.payload["text"])
                answer += result.payload["text"] + "\n"
            else:
                logger.warning(
                    f"formulate_answer: 'text' key not found in payload for result: {result}"
                )
        logger.info(f"formulate_answer: answer={answer}, sources={sources}")
        return PreAnswerResult(answer=answer, sources=sources)
    except KeyError as e:
        logger.error(f"KeyError in formulate_answer: {e}")
        logger.error(traceback.format_exc())
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred in formulate_answer: {e}")
        logger.error(traceback.format_exc())
        raise


# Validator agent
validator_agent = Agent(
    model,
    deps_type=ValidatorDependencies,
    result_type=ValidatorResult,
    system_prompt="You are a validator agent. You check if the answer is valid based on the question and data given and provide feedback.",
)


@validator_agent.tool
async def validate_answer(
    ctx: RunContext[ValidatorDependencies],
    user_query: str,
    answer: str,
    sources: List[str],
) -> ValidatorResult:
    logger.info(
        f"validate_answer: user_query={user_query}, answer={answer}, sources={sources}"
    )
    if not answer:
        logger.warning("validate_answer: The answer is empty.")
        return ValidatorResult(is_valid=False, feedback="The answer is empty.")
    logger.info("validate_answer: The answer is valid.")
    return ValidatorResult(is_valid=True, feedback="The answer is valid.")


# Orchestrator agent
orchestrator_agent = Agent(
    model,
    deps_type=OrchestratorDependencies,
    result_type=OrchestratorResult,
    system_prompt="You are an orchestrator agent. You coordinate the other agents and provide the final answer.",
)


@orchestrator_agent.tool
async def orchestrate(
    ctx: RunContext[OrchestratorDependencies], user_query: str
) -> OrchestratorResult:
    logger.info(f"orchestrate: user_query={user_query}")
    try:
        problem_deps = ProblemDependencies()
        problem_result = await problem_agent.run(user_query, deps=problem_deps)
        logger.info(f"orchestrate: problem_result={problem_result}")

        retrieve_deps = RetrieveDependencies(hybrid_search=hybrid_search)
        retrieve_result = await retrieve_agent.run(
            problem_result.data.query, deps=retrieve_deps
        )
        logger.info(f"orchestrate: retrieve_result={retrieve_result}")
        if not retrieve_result.data.results:
            logger.warning("retrieve_data returned no results.")
            return OrchestratorResult(
                question=user_query,
                answer="No results found.",
                sources=[],
            )

        # Convert the list of dictionaries to a list of ScoredPoint objects
        scored_points = [
            ScoredPoint(**point_dict) for point_dict in retrieve_result.data.results
        ]

        pre_answer_deps = PreAnswerDependencies()
        # Corrected line: Pass deps as a keyword argument
        pre_answer_result = await pre_answer_agent.run(
            problem_result.data.query,
            scored_points,
            deps=pre_answer_deps,
        )
        logger.info(f"orchestrate: pre_answer_result={pre_answer_result}")

        validator_deps = ValidatorDependencies()
        validator_result = await validator_agent.run(
            user_query,
            deps=validator_deps,
            answer=pre_answer_result.data.answer,
            sources=pre_answer_result.data.sources,
        )
        logger.info(f"orchestrate: validator_result={validator_result}")

        max_retries = 3
        retries = 0
        while not validator_result.data.is_valid and retries < max_retries:
            logger.warning(f"orchestrate: Retrying... (attempt {retries + 1})")
            time.sleep(1)  # Add a 1-second delay between retries
            # Corrected line: Pass deps as a keyword argument
            pre_answer_result = await pre_answer_agent.run(
                problem_result.data.query,
                scored_points,
                deps=pre_answer_deps,
            )
            logger.info(f"orchestrate: pre_answer_result (retry)={pre_answer_result}")
            validator_result = await validator_agent.run(
                user_query,
                deps=validator_deps,
                answer=pre_answer_result.data.answer,
                sources=pre_answer_result.data.sources,
            )
            logger.info(f"orchestrate: validator_result={validator_result}")
            retries += 1

        if not validator_result.data.is_valid:
            logger.error(
                "orchestrate: Max retries reached. The answer is still not valid."
            )
            raise ValueError("Max retries reached. The answer is still not valid.")

        # Format the answer in markdown with footnotes
        markdown_answer = pre_answer_result.data.answer
        footnotes = ""
        for i, source in enumerate(pre_answer_result.data.sources):
            markdown_answer = markdown_answer.replace(
                source, f"{source}<sup>[{i + 1}]</sup>"
            )
            footnotes += f"<sup>[{i + 1}]</sup> {source}\n"

        markdown_answer += "\n\n**Quellen:**\n" + footnotes

        return OrchestratorResult(
            question=user_query,
            answer=markdown_answer,
            sources=pre_answer_result.data.sources,
        )
    except httpx.HTTPError as e:
        logger.error(f"HTTP error in orchestrate: {e}")
        logger.error(traceback.format_exc())
        raise
    except KeyError as e:
        logger.error(f"KeyError in orchestrate: {e}")
        logger.error(traceback.format_exc())
        raise
    except ValidationError as e:
        logger.error(f"ValidationError in orchestrate: {e}")
        logger.error(traceback.format_exc())
        raise
    except UsageLimitExceeded as e:
        logger.error(f"UsageLimitExceeded in orchestrate: {e}")
        logger.error(traceback.format_exc())
        raise
    except ModelHTTPError as e:
        logger.error(f"ModelHTTPError in orchestrate: {e}")
        logger.error(traceback.format_exc())
        raise
    except UnexpectedModelBehavior as e:
        logger.error(f"UnexpectedModelBehavior in orchestrate: {e}")
        logger.error(traceback.format_exc())
        raise
    except FallbackExceptionGroup as e:
        logger.error(f"FallbackExceptionGroup in orchestrate: {e}")
        logger.error(traceback.format_exc())
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred in orchestrate: {e}")
        logger.error(traceback.format_exc())
        raise


# --- Main ---
def main():
    print(f"Pydantic version {pydantic.__version__}")
    print(f"Pydantic AI version {pydantic_ai.__version__}")

    user_query = "Was soll gem der Bundesregierung im Datenschutz erreicht werden?"
    orchestrator_deps = OrchestratorDependencies()
    max_retries = 3
    retries = 0
    while retries < max_retries:
        try:
            result = orchestrator_agent.run_sync(user_query, deps=orchestrator_deps)

            print(f"**Frage:**\n{result.data.question}\n")
            print(f"**Antwort:**\n{result.data.answer}\n")
            return  # Success, exit the loop
        except TypeError as e:
            if "NoneType" in str(e):
                logger.error(
                    f"TypeError: 'NoneType' object error in main (attempt {retries + 1}): {e}"
                )
                logger.error(traceback.format_exc())
                retries += 1
                if retries >= max_retries:
                    logger.error(
                        "Max retries reached for TypeError: 'NoneType' object error in main."
                    )
                    raise
            else:
                logger.error(f"An unexpected TypeError occurred in main: {e}")
                logger.error(traceback.format_exc())
                raise
        except Exception as e:
            logger.error(f"An unexpected error occurred in main: {e}")
            logger.error(traceback.format_exc())
            raise

    logger.error("Max retries reached in main.")


if __name__ == "__main__":
    main()
