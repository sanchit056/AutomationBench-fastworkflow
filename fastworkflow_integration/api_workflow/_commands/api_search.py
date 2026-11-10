"""FastWorkflow command wrapping AutomationBench's ``api_search`` tool (Design 1)."""

import fastworkflow
from fastworkflow import CommandOutput, CommandResponse
from fastworkflow.train.generate_synthetic import generate_diverse_utterances
from fastworkflow.workflow import Workflow
from pydantic import BaseModel, Field

from automationbench.tools.api import api_search


class Signature:
    """Search available API endpoints by keyword.
    Use this to discover which endpoint to call before using api_fetch.
    Returns full endpoint details: id, method, url, description, parameters, request body and response format.
    Always include both parameter tags: query and top_k."""

    class Input(BaseModel):
        query: str = Field(
            default="NOT_FOUND",
            description=(
                "Space-separated keywords (e.g. 'inbox messages', 'trash', 'send label'). "
                "Uses API-native terms: 'messages' not 'emails', 'trash' not 'delete'."
            ),
        )
        top_k: int = Field(
            default=5,
            description="Maximum number of results to return (default 5).",
        )

    class Output(BaseModel):
        result: str = Field(description="JSON with matching endpoints and the count of results")

    plain_utterances = [
        "search the api endpoints for inbox messages",
        "find the endpoint for sending a slack message",
        "which api endpoint updates a salesforce contact",
        "look up the api to list google sheets rows",
        "search endpoints for hubspot deals",
        "find the url to create a calendar event",
        "what endpoint do I use to trash an email",
        "discover the api for adding a row to a spreadsheet",
    ]

    template_utterances = []

    @staticmethod
    def generate_utterances(workflow: Workflow, command_name: str) -> list[str]:
        return [
            command_name.split("/")[-1].lower().replace("_", " ")
        ] + generate_diverse_utterances(Signature.plain_utterances, command_name)


class ResponseGenerator:
    def _process_command(self, workflow: Workflow, input: Signature.Input) -> Signature.Output:
        try:
            return Signature.Output(result=api_search(query=input.query, top_k=input.top_k))
        except Exception as e:  # surface tool errors to the agent instead of FW's retry loop
            return Signature.Output(result=f'{{"error": "{type(e).__name__}: {e}"}}')

    def __call__(
        self, workflow: Workflow, command: str, command_parameters: Signature.Input
    ) -> CommandOutput:
        output = self._process_command(workflow, command_parameters)
        return CommandOutput(command_response=CommandResponse(response=output.result))
