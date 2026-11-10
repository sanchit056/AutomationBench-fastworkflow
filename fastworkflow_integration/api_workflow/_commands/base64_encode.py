"""FastWorkflow command wrapping AutomationBench's ``base64_encode`` tool (Design 1)."""

import fastworkflow
from fastworkflow import CommandOutput, CommandResponse
from fastworkflow.train.generate_synthetic import generate_diverse_utterances
from fastworkflow.workflow import Workflow
from pydantic import BaseModel, Field

from automationbench.tools.api import base64_encode


class Signature:
    """Encode text to base64url, the format Gmail API body fields require.
    The Gmail messages.send / drafts endpoints accept the message body only in base64url form:
    either raw (the full RFC 2822 message, base64url-encoded) or payload.body.data (the body text, base64url-encoded).
    Use this to produce that encoding, then pass the result to api_fetch. It does not call any API endpoint."""

    class Input(BaseModel):
        text: str = Field(
            default="NOT_FOUND",
            description="Plaintext to encode (an email body, or a full RFC 2822 message).",
        )

    class Output(BaseModel):
        result: str = Field(description="The base64url-encoded string")

    plain_utterances = [
        "base64 encode this email message",
        "encode the RFC 2822 message for gmail",
        "convert the email body to base64url",
        "encode this text so I can send it with the gmail api",
        "base64url encode the raw message",
        "make the base64 version of this email body",
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
            return Signature.Output(result=base64_encode(text=input.text))
        except Exception as e:
            return Signature.Output(result=f'{{"error": "{type(e).__name__}: {e}"}}')

    def __call__(
        self, workflow: Workflow, command: str, command_parameters: Signature.Input
    ) -> CommandOutput:
        output = self._process_command(workflow, command_parameters)
        return CommandOutput(command_response=CommandResponse(response=output.result))
