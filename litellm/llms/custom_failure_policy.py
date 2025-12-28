import litellm
from litellm.utils import ModelResponse

class CustomFailurePolicy:
    """
    A class to define custom failure policies for LLM responses.
    """

    @staticmethod
    def check_response(response: ModelResponse, model: str, llm_provider: str):
        """
        Checks if the response from the LLM should be considered a failure.

        Args:
            response (ModelResponse): The response object from the LLM.
            model (str): The model name used for the request.
            llm_provider (str): The provider of the LLM.

        Raises:
            litellm.BadRequestError: If the response is deemed a failure.
        """
        if not response.choices:
            raise litellm.BadRequestError(
                message="LLM returned no choices.", model=model, llm_provider=llm_provider
            )

        first_choice = response.choices[0]

        if first_choice.message is None:
            raise litellm.BadRequestError(
                message="LLM returned a choice with no message.",
                model=model,
                llm_provider=llm_provider,
            )

        has_content = (
            hasattr(first_choice.message, "content")
            and first_choice.message.content is not None
            and str(first_choice.message.content).strip() != ""
        )
        has_tool_calls = (
            hasattr(first_choice.message, "tool_calls")
            and first_choice.message.tool_calls is not None
        )

        if not has_content and not has_tool_calls:
            if first_choice.finish_reason in ["stop", "end_turn", None]:
                raise litellm.BadRequestError(
                    message="LLM returned an empty message with no tool calls.",
                    model=model,
                    llm_provider=llm_provider,
                )