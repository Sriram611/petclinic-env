# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the PetClinic Environment.

Defines the Action and Observation types for a veterinary clinic
day-to-day operations simulation.

Three tasks — each episode runs all three simultaneously:
  Task 1 (easy)   — Appointment Scheduling
  Task 2 (medium) — Walk-in Triage
  Task 3 (hard)   — Full Day Operations
"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import Field
from openenv.core.env_server.types import Action, Observation


# ---------------------------------------------------------------------------
# Action Model
# ---------------------------------------------------------------------------

class PetclinicAction(Action):
    """
    One action the agent can take inside the pet clinic.

    The agent picks an action_type and fills in the relevant fields.
    Any fields not needed for the chosen action_type can be left as None.

    Examples:
        Schedule an appointment:
            PetclinicAction(
                action_type="schedule_appointment",
                patient_id="P001",
                doctor_id="D1",
                room_id="R1",
                time_slot="09:00"
            )

        Insert a walk-in patient:
            PetclinicAction(
                action_type="insert_walkin",
                patient_id="W001",
                doctor_id="D2",
                room_id="R2"
            )

        Discharge a patient:
            PetclinicAction(
                action_type="discharge_patient",
                patient_id="P003"
            )
    """

    action_type: Literal[
        "schedule_appointment",  # Task 1 — book a patient into a slot
        "insert_walkin",         # Task 2 — insert urgent walk-in immediately
        "queue_walkin",          # Task 2 — add walk-in to waiting queue
        "refer_patient",         # Task 2/3 — refer to external specialist
        "assign_doctor",         # Task 3 — reassign doctor to different patient
        "dispense_prescription", # Task 3 — dispense medicine to ready patient
        "file_lab_result",       # Task 3 — match lab result to correct patient
        "discharge_patient",     # Task 3 — discharge treated patient
        "schedule_followup",     # Task 3 — book follow-up for chronic case
        "wait",                  # any task — do nothing this step
    ] = Field(..., description="The type of action to perform")

    # Patient and resource identifiers
    patient_id: Optional[str] = Field(
        None, description="ID of the patient this action applies to"
    )
    doctor_id: Optional[str] = Field(
        None, description="ID of the doctor to assign"
    )
    room_id: Optional[str] = Field(
        None, description="ID of the room to use"
    )
    time_slot: Optional[str] = Field(
        None, description="Time slot for scheduling e.g. '09:00', '10:30'"
    )
    lab_result_id: Optional[str] = Field(
        None, description="ID of the lab result to file (Task 3)"
    )

    # Optional reasoning field — logged but not used by environment logic
    reasoning: Optional[str] = Field(
        None, description="Agent's reasoning for this action (for logging)"
    )


# ---------------------------------------------------------------------------
# Supporting data shapes used inside the Observation
# ---------------------------------------------------------------------------

class PatientInfo(Dict[str, Any]):
    """
    Represents one patient in the clinic.
    Returned as a plain dict inside the observation for simplicity.

    Keys:
        patient_id   : str   — unique ID e.g. "P001"
        name         : str   — pet name e.g. "Max"
        species      : str   — "dog" | "cat" | "rabbit" | "bird"
        condition    : str   — reason for visit e.g. "vaccination"
        severity     : str   — "critical" | "moderate" | "minor"
        is_walkin    : bool  — True if unscheduled walk-in
        wait_steps   : int   — how many steps this patient has been waiting
        needs_surgery: bool  — True if surgery room required
        needs_specialist: str | None — e.g. "large_animal" | "small_animal" | None
        status       : str   — "waiting" | "in_treatment" | "treated" | "referred"
    """
    pass


class DoctorInfo(Dict[str, Any]):
    """
    Represents one doctor/vet in the clinic.

    Keys:
        doctor_id      : str   — unique ID e.g. "D1"
        name           : str   — e.g. "Dr. Priya"
        specialisation : str   — "small_animal" | "large_animal" | "general"
        status         : str   — "available" | "busy"
        current_patient: str | None — patient_id they are treating
        fatigue        : int   — 0–100, increases each step when busy (Task 3)
        busy_until     : int   — step number when they become free
    """
    pass


class RoomInfo(Dict[str, Any]):
    """
    Represents one examination room.

    Keys:
        room_id        : str   — unique ID e.g. "R1"
        room_type      : str   — "general" | "surgery" | "consultation"
        status         : str   — "free" | "occupied"
        current_patient: str | None — patient_id occupying the room
        free_at_step   : int   — step when room becomes available
    """
    pass


class LabResultInfo(Dict[str, Any]):
    """
    Represents a pending lab result (Task 3 only).

    Keys:
        lab_id     : str  — unique ID e.g. "L001"
        patient_id : str  — which patient this belongs to
        filed      : bool — True once agent has filed it correctly
    """
    pass


# ---------------------------------------------------------------------------
# Observation Model
# ---------------------------------------------------------------------------

class PetclinicObservation(Observation):
    """
    Everything the agent can see at each step of the clinic simulation.

    The agent reads this observation and decides which action to take next.
    All three task contexts are included so the agent can reason about
    scheduling, triage, and full operations simultaneously.
    """

    # ── Episode progress ──────────────────────────────────────────────────
    step: int = Field(
        default=0,
        description="Current step number (0-indexed)"
    )
    max_steps: int = Field(
        default=20,
        description="Total steps in this episode"
    )
    time_of_day: str = Field(
        default="09:00",
        description="Current simulated clinic time e.g. '09:00', '11:30'"
    )

    # ── Task 1 — Appointment Scheduling ───────────────────────────────────
    appointment_requests: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Task 1: Unscheduled appointment requests waiting to be booked. "
            "Each item contains patient_id, species, condition, severity, "
            "preferred_time, needs_surgery, needs_specialist."
        )
    )
    scheduled_appointments: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Task 1: Already booked appointments for today. "
            "Each item contains patient_id, doctor_id, room_id, time_slot."
        )
    )

    # ── Task 2 — Walk-in Triage ───────────────────────────────────────────
    walkin_queue: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Task 2: Unscheduled walk-in patients currently in the waiting area. "
            "Each item contains patient_id, species, condition, severity, "
            "wait_steps, is_walkin=True."
        )
    )

    # ── Task 3 — Full Day Operations ──────────────────────────────────────
    patients_in_treatment: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Task 3: Patients currently being treated. "
            "Includes treatment_step, prescription_ready, lab_result_pending."
        )
    )
    pending_lab_results: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Task 3: Lab results that have arrived and need to be "
            "filed to the correct patient record."
        )
    )
    patients_ready_for_discharge: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Task 3: Patients whose treatment is complete. "
            "Agent must discharge each one before episode ends."
        )
    )
    prescriptions_to_dispense: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Task 3: Prescriptions that are ready and must be "
            "dispensed before the patient is discharged."
        )
    )

    # ── Resources ─────────────────────────────────────────────────────────
    doctors: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "All doctors and their current status. "
            "Each item: doctor_id, name, specialisation, status, "
            "current_patient, fatigue, busy_until."
        )
    )
    rooms: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "All examination rooms and their current status. "
            "Each item: room_id, room_type, status, current_patient, free_at_step."
        )
    )
    supplies_remaining: Dict[str, int] = Field(
        default_factory=dict,
        description=(
            "Task 3: Remaining stock of key supplies. "
            "e.g. {'vaccines': 10, 'bandages': 8, 'anaesthesia': 3}"
        )
    )

    # ── Scores so far this episode ────────────────────────────────────────
    task1_score_so_far: float = Field(
        default=0.0,
        description="Running Task 1 score (appointment scheduling accuracy)"
    )
    task2_score_so_far: float = Field(
        default=0.0,
        description="Running Task 2 score (walk-in triage accuracy)"
    )
    task3_score_so_far: float = Field(
        default=0.0,
        description="Running Task 3 score (full operations efficiency)"
    )

    # ── Action mask ───────────────────────────────────────────────────────
    valid_actions: List[str] = Field(
        default_factory=list,
        description=(
            "List of action_type strings the agent can legally use right now. "
            "IMPORTANT: Only attempt actions from this list. "
            "Attempting an invalid action returns a penalty."
        )
    )

    # ── Terminal signals ──────────────────────────────────────────────────
    done: bool = Field(
        default=False,
        description="True when the episode has ended"
    )
    reward: float = Field(
        default=0.0,
        description="Reward earned on this step"
    )

    # ── Final scores (populated only when done=True) ──────────────────────
    final_task1_score: Optional[float] = Field(
        default=None,
        description="Final Task 1 score 0.0-1.0 (set when done=True)"
    )
    final_task2_score: Optional[float] = Field(
        default=None,
        description="Final Task 2 score 0.0-1.0 (set when done=True)"
    )
    final_task3_score: Optional[float] = Field(
        default=None,
        description="Final Task 3 score 0.0-1.0 (set when done=True)"
    )

    # ── Human-readable context for the LLM ───────────────────────────────
    context: str = Field(
        default="",
        description=(
            "Plain English summary of the current clinic situation. "
            "Read this to understand what is happening before deciding your action."
        )
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional episode metadata for logging"
    )