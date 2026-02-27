import ast
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from anthropic import Anthropic
from dotenv import load_dotenv
from openai import BadRequestError, OpenAI

load_dotenv()


@dataclass
class Response:
    content: str | None | Dict
    input_tokens: int
    output_tokens: int


class LLMClient:
    """
    Generic LLM API client supporting OpenAI and Anthropic providers.

    Usage:
        client = LLMClient()
        response, tokens = client.get_response(
            provider="openai",
            model="gpt-4",
            messages=[{"role": "system", "content": "..."}, ...],
            chat_parameters={"temperature": 0.7},
            token_counter={"input": 0, "output": 0}
        )
    """

    def __init__(self, client="anthropic", long_context: bool = False):
        self.long_context = long_context
        self._openai_client: Optional[OpenAI] = None
        self._anthropic_client: Optional[Anthropic] = None

        if client == "openai":
            self._call_client = self._call_openai
        elif client == "anthropic":
            self._call_client = self._call_anthropic
        else:
            raise ValueError(f"Unsupported client: {client}")

    def _get_openai_client(self) -> OpenAI:
        """Get or create OpenAI client."""
        if self._openai_client is None:
            self._openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        return self._openai_client

    def _get_anthropic_client(self) -> Anthropic:
        """Get or create Anthropic client."""
        if self._anthropic_client is None:
            if self.long_context:
                default_headers = {"anthropic-beta": "context-1m-2025-08-07"}
            else:
                default_headers = None
            self._anthropic_client = Anthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY"),
                default_headers=default_headers,
            )
        return self._anthropic_client

    def _remove_thinking_tags(self, text: str) -> str:
        """
        Remove thinking tags from model response if present.

        Args:
            text: Raw response text

        Returns:
            Text with thinking tags removed
        """
        if "</think>" in text:
            return text.split("</think>")[1]
        return text

    def _extract_json(
        self, text: str, has_response_format: bool
    ) -> tuple[str | None, str | None]:
        """
        Extract JSON from response text.

        Args:
            text: Response text
            has_response_format: Whether response_format was specified in chat_parameters

        Returns:
            Tuple of (json_string, error_message)
        """
        if has_response_format:
            # Model should return pure JSON
            return text, None

        # Try to extract JSON from markdown code blocks
        json_match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if json_match:
            return json_match.group(1), None

        # Try to find JSON without code blocks
        json_match = re.search(r"(\{.*\})", text, re.DOTALL)
        if json_match:
            return json_match.group(1), None

        return None, "Error: No valid JSON found in response"

    def _call_openai(
        self, model: str, messages: list[dict], chat_parameters: dict
    ) -> tuple[str, int, int]:
        """
        Call OpenAI API.

        Returns:
            Tuple of (response_text, input_tokens, output_tokens)
        """
        self._openai_client = self._get_openai_client()
        resp = self._openai_client.chat.completions.create(
            **{
                **chat_parameters,
                "model": model,
                "messages": messages,
            }
        )
        text = resp.choices[0].message.content.strip()
        input_tokens = resp.usage.prompt_tokens
        output_tokens = resp.usage.completion_tokens
        return text, input_tokens, output_tokens

    def _call_anthropic(
        self, model: str, messages: list[dict], chat_parameters: dict
    ) -> tuple[str, int, int]:
        """
        Call Anthropic API.

        Note: Anthropic API has different message format - system message is separate.

        Returns:
            Tuple of (response_text, input_tokens, output_tokens)
        """
        self._anthropic_client = self._get_anthropic_client()
        # Extract system message if present
        system_message = None
        anthropic_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                anthropic_messages.append(msg)

        # Prepare kwargs
        kwargs = {
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": chat_parameters.get("max_tokens", 10000),
        }

        # Add system message if present
        if system_message:
            kwargs["system"] = system_message

        # Add other chat parameters (excluding OpenAI-specific ones)
        for key, value in chat_parameters.items():
            if key not in ["max_tokens", "response_format"]:
                kwargs[key] = value

        resp = self._anthropic_client.messages.create(**kwargs)

        # Extract text from response
        if len(resp.content) == 0:
            error = "Response has no content."
            if resp.stop_reason == "refusal":
                error = "Model refused to complete the request."
            raise ValueError(error)

        text = resp.content[0].text
        input_tokens = resp.usage.input_tokens
        output_tokens = resp.usage.output_tokens

        return text, input_tokens, output_tokens

    def get_response(
        self,
        model: str,
        messages: list[dict],
        chat_parameters: dict | None = None,
        max_retries: int = 10,
        enable_error_reprompting: bool = True,
        track_tokens: bool = True,
        output_json: bool = True,
    ) -> Response:
        """
        Make an LLM API call with retry logic and JSON parsing.

        Args:
            provider: "openai" or "anthropic"
            model: Model name (e.g., "gpt-4", "claude-sonnet-4-5")
            messages: List of message dicts with "role" and "content"
            chat_parameters: Additional parameters for the API call (temperature, response_format, etc.)
            max_retries: Maximum number of retry attempts
            enable_error_reprompting: If True, append error messages to retry
            track_tokens: If True, track token usage
            token_counter: Dict to accumulate tokens (must have "input" and "output" keys)
            output_json: If True, parse response as JSON; otherwise return raw text

        Returns:
            Tuple of (response, token_counter) where:
                - response is dict if output_json=True, str otherwise
                - token_counter is the updated token counter dict

        Raises:
            BadRequestError: Re-raised if caller wants to handle token overflow (OpenAI only)
            Exception: If max retries exhausted or other unhandled errors
        """
        if chat_parameters is None:
            chat_parameters = {}

        token_counter = {"input": 0, "output": 0}

        # Create a working copy of messages to allow retry modifications
        messages_copy = list(messages)

        has_response_format = "response_format" in chat_parameters

        for trial in range(max_retries):
            try:
                # Make API call based on provider
                text, input_tokens, output_tokens = self._call_client(
                    model, messages_copy, chat_parameters
                )

                # Track tokens
                if track_tokens:
                    token_counter["input"] += input_tokens
                    token_counter["output"] += output_tokens

                # Remove thinking tags if present
                text = self._remove_thinking_tags(text)

                # If not expecting JSON, return raw text
                if not output_json:
                    return Response(
                        content=text,
                        input_tokens=token_counter["input"],
                        output_tokens=token_counter["output"],
                    )

                try:
                    parsed_json = json.loads(text)
                    return Response(
                        content=parsed_json,
                        input_tokens=token_counter["input"],
                        output_tokens=token_counter["output"],
                    )
                except json.JSONDecodeError:
                    pass  # Fall through to retry logic below

                # Parse JSON response
                json_str, error_msg = self._extract_json(text, has_response_format)

                if json_str:
                    try:
                        parsed_json = json.loads(json_str)
                        return Response(
                            content=parsed_json,
                            input_tokens=token_counter["input"],
                            output_tokens=token_counter["output"],
                        )
                    except json.JSONDecodeError:
                        try:
                            parsed_json = ast.literal_eval(json_str)
                            return Response(
                                content=parsed_json,
                                input_tokens=token_counter["input"],
                                output_tokens=token_counter["output"],
                            )
                        except (ValueError, SyntaxError) as e:
                            error_msg = f"JSON parsing error: {str(e)} \n Response was: {json_str}"
                    except Exception as e:
                        error_msg = f"Unexpected error parsing JSON: {str(e)} \n Response was: {json_str}"

                # If we get here, JSON parsing failed
                if not enable_error_reprompting:
                    raise ValueError(error_msg)

                # Append error message and retry
                messages_copy.extend(
                    [
                        {"role": "assistant", "content": text},
                        {
                            "role": "user",
                            "content": f"{error_msg}\nPlease provide a valid JSON response.",
                        },
                    ]
                )

            except BadRequestError as e:
                # Re-raise BadRequestError for caller to handle (e.g., token reduction)
                raise e
            except Exception as e:
                if trial == max_retries - 1:
                    raise Exception(
                        f"Failed after {max_retries} retries. Last error: {str(e)}"
                    )
                # Continue to next retry

        raise Exception(f"Failed to get valid response after {max_retries} retries")


class AgentClient:
    """
    Client used for the LLM agents in the environment.
    This is very simple as all the parsing etc is done outside by the agent itself.
    Supports both OpenAI-compatible endpoints and the Anthropic API.
    """

    # Keys in chat_params that are OpenAI-specific and must be stripped for Anthropic
    _OPENAI_ONLY_PARAMS = {"response_format", "reasoning_effort"}

    def __init__(self, provider: str = "openai", **kwargs):
        self.provider = provider
        self._client: Any
        if provider == "openai":
            self._client = OpenAI(**kwargs)
        elif provider == "anthropic":
            self._client = Anthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY"), **kwargs
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def get_response(
        self, messages: List[Dict[str, str]], chat_params: Dict
    ) -> Response:
        if self.provider == "openai":
            return self._get_response_openai(messages, chat_params)
        else:
            return self._get_response_anthropic(messages, chat_params)

    def _get_response_openai(
        self, messages: List[Dict[str, str]], chat_params: Dict
    ) -> Response:
        response = self._client.chat.completions.create(
            **{**chat_params, "messages": messages}
        )
        input_tokens = 0
        output_tokens = 0
        if response.usage is not None:
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
        return Response(
            content=response.choices[0].message.content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def _get_response_anthropic(
        self, messages: List[Dict[str, str]], chat_params: Dict
    ) -> Response:
        # Anthropic takes system as a top-level param, not inside messages
        system_message = None
        anthropic_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                anthropic_messages.append(msg)

        kwargs = {
            "model": chat_params.get("model"),
            "messages": anthropic_messages,
            "max_tokens": chat_params.get("max_tokens", 4096),
        }
        if system_message:
            kwargs["system"] = system_message

        # Pass through all chat_params, excluding OpenAI-specific keys
        for key, value in chat_params.items():
            if key not in self._OPENAI_ONLY_PARAMS and key != "max_tokens":
                kwargs[key] = value

        response = self._client.messages.create(**kwargs)

        if not response.content:
            raise ValueError("Empty response from Anthropic API")

        return Response(
            content=response.content[0].text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
