# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
PetClinic Environment Client.

Connects to the PetClinic environment server via WebSocket.
Used by inference.py to interact with the running environment.

Usage:
    # Connect to local server
    with PetclinicEnv(base_url="http://localhost:8000").sync() as env:
        result = env.reset()
        result = env.step(PetclinicAction(action_type="wait"))

    # Connect to HuggingFace Space
    with PetclinicEnv(base_url="https://your-space.hf.space").sync() as env:
        result = env.reset()
"""

from typing import Any, Dict, List, Optional

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import PetclinicAction, PetclinicObservation
except (ImportError, ModuleNotFoundError):
    from models import PetclinicAction, PetclinicObservation


class PetclinicEnv(
    EnvClient[PetclinicAction, PetclinicObservation, State]
):
    """
    WebSocket client for the PetClinic Environment.

    Maintains a persistent connection to the environment server.
    Each client instance gets its own isolated clinic session.

    Example (sync):
        with PetclinicEnv(base_url="http://localhost:8000").sync() as env:
            result = env.reset()
            print(result.observation.context)

            while not result.done:
                action = PetclinicAction(action_type="wait")
                result = env.step(action)

            print("Task 1:", result.observation.final_task1_score)
            print("Task 2:", result.observation.final_task2_score)
            print("Task 3:", result.observation.final_task3_score)

    Example (async):
        async with PetclinicEnv(base_url="http://localhost:8000") as env:
            result = await env.reset()
            result = await env.step(PetclinicAction(action_type="wait"))
    """

    def _step_payload(self, action: PetclinicAction) -> Dict[str, Any]:
        """
        Convert PetclinicAction to JSON payload for the WebSocket message.

        Only includes fields that are set — None fields are omitted
        to keep the payload clean.
        """
        payload: Dict[str, Any] = {
            "action_type": action.action_type,
        }

        # Only add optional fields if they are set
        if action.patient_id is not None:
            payload["patient_id"] = action.patient_id
        if action.doctor_id is not None:
            payload["doctor_id"] = action.doctor_id
        if action.room_id is not None:
            payload["room_id"] = action.room_id
        if action.time_slot is not None:
            payload["time_slot"] = action.time_slot
        if action.lab_result_id is not None:
            payload["lab_result_id"] = action.lab_result_id
        if action.reasoning is not None:
            payload["reasoning"] = action.reasoning

        return payload

    def _parse_result(
        self, payload: Dict[str, Any]
    ) -> StepResult[PetclinicObservation]:
        """
        Parse the server's JSON response into a typed StepResult.

        The server returns the full PetclinicObservation as a dict.
        We reconstruct it field by field so the client has typed access.
        """
        obs_data = payload.get("observation", {})

        observation = PetclinicObservation(
            # Episode progress
            step=obs_data.get("step", 0),
            max_steps=obs_data.get("max_steps", 20),
            time_of_day=obs_data.get("time_of_day", "09:00"),

            # Task 1 — Appointment Scheduling
            appointment_requests=obs_data.get("appointment_requests", []),
            scheduled_appointments=obs_data.get("scheduled_appointments", []),

            # Task 2 — Walk-in Triage
            walkin_queue=obs_data.get("walkin_queue", []),

            # Task 3 — Full Day Operations
            patients_in_treatment=obs_data.get("patients_in_treatment", []),
            pending_lab_results=obs_data.get("pending_lab_results", []),
            patients_ready_for_discharge=obs_data.get(
                "patients_ready_for_discharge", []
            ),
            prescriptions_to_dispense=obs_data.get("prescriptions_to_dispense", []),

            # Resources
            doctors=obs_data.get("doctors", []),
            rooms=obs_data.get("rooms", []),
            supplies_remaining=obs_data.get("supplies_remaining", {}),

            # Running scores
            task1_score_so_far=obs_data.get("task1_score_so_far", 0.0),
            task2_score_so_far=obs_data.get("task2_score_so_far", 0.0),
            task3_score_so_far=obs_data.get("task3_score_so_far", 0.0),

            # Action mask
            valid_actions=obs_data.get("valid_actions", ["wait"]),

            # Terminal signals
            done=payload.get("done", obs_data.get("done", False)),
            reward=payload.get("reward", obs_data.get("reward", 0.0)),

            # Final scores (only set when done=True)
            final_task1_score=obs_data.get("final_task1_score"),
            final_task2_score=obs_data.get("final_task2_score"),
            final_task3_score=obs_data.get("final_task3_score"),

            # LLM context
            context=obs_data.get("context", ""),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", obs_data.get("reward", 0.0)),
            done=payload.get("done", obs_data.get("done", False)),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        """
        Parse the server's state response into a State object.

        Called when env.state() is invoked to get episode metadata.
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )