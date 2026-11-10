"""FastWorkflow command wrapping AutomationBench's ``api_fetch`` tool (Design 1).

The task's WorldState arrives through ``workflow.context["world"]``; ``api_fetch``
routes the URL to the same handlers, service gating and responses the baseline uses.
"""

import json

import fastworkflow
from fastworkflow import CommandOutput, CommandResponse
from fastworkflow.train.generate_synthetic import generate_diverse_utterances
from fastworkflow.workflow import Workflow
from pydantic import BaseModel, Field

from automationbench.tools.api import api_fetch


class Signature:
    """Call an API endpoint by its full URL.
    Use api_search first to discover the correct URL and parameters for an endpoint.
    Returns a JSON string mimicking the real API response.
    Always include all four parameter tags: method, url, params and body. Use {} for params or body when there is nothing to send."""

    class Input(BaseModel):
        method: str = Field(
            default="NOT_FOUND",
            description="HTTP method: GET, POST, PUT, PATCH or DELETE.",
        )
        url: str = Field(
            default="NOT_FOUND",
            description="Full API URL from api_search results, with path parameters filled in.",
        )
        params: str = Field(
            default="{}",
            description="Query parameters as a JSON object string, or {} when there are none.",
        )
        body: str = Field(
            default="{}",
            description="Request body as a JSON object string, or {} when there is none.",
        )

    class Output(BaseModel):
        result: str = Field(description="JSON string mimicking the real API response")

    plain_utterances = [
        "call the gmail api to list inbox messages",
        "send a GET request to the salesforce query endpoint",
        "post a message to the slack channel endpoint",
        "make a PATCH request to update the hubspot contact",
        "fetch the google sheets rows from the values endpoint",
        "delete the calendar event via the api",
        "call the endpoint with this request body",
        "put the updated record to the api url",
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
            result = api_fetch(
                world=workflow.context["world"],
                method=input.method,
                url=input.url,
                params=input.params,
                body=input.body,
            )
        except Exception as e:
            # Return the error rather than raising: FW retries a raised command, and
            # re-running a write would apply it twice to the WorldState.
            result = json.dumps({"error": {"message": f"{type(e).__name__}: {e}"}})
        return Signature.Output(result=result)

    def __call__(
        self, workflow: Workflow, command: str, command_parameters: Signature.Input
    ) -> CommandOutput:
        output = self._process_command(workflow, command_parameters)
        return CommandOutput(command_response=CommandResponse(response=output.result))
