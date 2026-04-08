# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
PetClinic Environment — Core Game Logic

Simulates one full working day at a veterinary clinic.
The AI agent automates the day-to-day operations of the clinic
by making scheduling, triage, and treatment decisions.

Three tasks are evaluated on every episode:
  Task 1 (easy)   — Appointment Scheduling
                    Book incoming requests into correct slots
                    without double-booking or mismatching resources.

  Task 2 (medium) — Walk-in Triage
                    Task 1 + handle unexpected walk-in patients
                    by urgency, inserting them into the live schedule.

  Task 3 (hard)   — Full Day Operations
                    Task 1 + Task 2 + manage prescriptions,
                    lab results, patient discharge, and follow-ups.
"""

import random
from uuid import uuid4
from typing import List, Dict, Any, Optional

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

# FIXED
try:
    from ..models import PetclinicAction, PetclinicObservation
except (ImportError, ModuleNotFoundError):
    from models import PetclinicAction, PetclinicObservation


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_STEPS = 20
NUM_DOCTORS = 2
NUM_ROOMS = 3
NUM_APPOINTMENT_REQUESTS = 6
NUM_WALKINS = 4

SPECIES = ["dog", "cat", "rabbit", "bird"]
CONDITIONS = ["vaccination", "checkup", "injury", "infection", "surgery", "dental"]
SEVERITIES = ["critical", "moderate", "minor"]

TIME_SLOTS = [
    "09:00", "09:30", "10:00", "10:30", "11:00", "11:30",
    "12:00", "12:30", "13:00", "13:30", "14:00", "14:30",
    "15:00", "15:30", "16:00", "16:30",
]

SUPPLIES_INITIAL = {
    "vaccines": 10,
    "bandages": 12,
    "anaesthesia": 4,
    "antibiotics": 8,
}

# Reward values
REWARD_CORRECT_SCHEDULE    =  0.20
REWARD_CORRECT_TRIAGE      =  0.25
REWARD_CORRECT_DISCHARGE   =  0.20
REWARD_LAB_FILED           =  0.15
REWARD_PRESCRIPTION_DONE   =  0.15
REWARD_FOLLOWUP_BOOKED     =  0.10
REWARD_WAIT                =  0.00

PENALTY_DOUBLE_BOOK        = -0.30
PENALTY_WRONG_DOCTOR       = -0.20
PENALTY_WRONG_ROOM         = -0.20
PENALTY_MISSED_CRITICAL    = -0.40
PENALTY_WRONG_TRIAGE       = -0.25
PENALTY_INVALID_ACTION     = -0.10
PENALTY_IDLE_DOCTOR        = -0.05


# ---------------------------------------------------------------------------
# Helper — generate clinic data
# ---------------------------------------------------------------------------

def _make_patient(patient_id: str, is_walkin: bool = False) -> Dict[str, Any]:
    """Generate one realistic patient record."""
    condition = random.choice(CONDITIONS)
    severity = (
        random.choices(SEVERITIES, weights=[20, 45, 35])[0]
        if is_walkin
        else random.choices(SEVERITIES, weights=[10, 40, 50])[0]
    )
    needs_surgery = condition == "surgery"
    species = random.choice(SPECIES)
    needs_specialist = (
        "large_animal" if species in ["dog"] and condition in ["surgery", "injury"]
        else "small_animal" if species in ["cat", "rabbit", "bird"]
        else None
    )
    return {
        "patient_id":        patient_id,
        "name":              f"Pet_{patient_id}",
        "species":           species,
        "condition":         condition,
        "severity":          severity,
        "is_walkin":         is_walkin,
        "wait_steps":        0,
        "needs_surgery":     needs_surgery,
        "needs_specialist":  needs_specialist,
        "status":            "waiting",
        "preferred_time":    random.choice(TIME_SLOTS) if not is_walkin else None,
        "prescription_ready":  False,
        "lab_result_pending":  condition in ["infection", "injury"],
        "treatment_steps_remaining": random.randint(1, 3),
        "followup_needed":   condition in ["infection", "surgery", "dental"],
        "followup_booked":   False,
    }


def _make_doctors() -> List[Dict[str, Any]]:
    return [
        {
            "doctor_id":       "D1",
            "name":            "Dr. Priya",
            "specialisation":  "small_animal",
            "status":          "available",
            "current_patient": None,
            "fatigue":         0,
            "busy_until":      0,
        },
        {
            "doctor_id":       "D2",
            "name":            "Dr. Raj",
            "specialisation":  "large_animal",
            "status":          "available",
            "current_patient": None,
            "fatigue":         0,
            "busy_until":      0,
        },
    ]


def _make_rooms() -> List[Dict[str, Any]]:
    return [
        {
            "room_id":         "R1",
            "room_type":       "general",
            "status":          "free",
            "current_patient": None,
            "free_at_step":    0,
        },
        {
            "room_id":         "R2",
            "room_type":       "surgery",
            "status":          "free",
            "current_patient": None,
            "free_at_step":    0,
        },
        {
            "room_id":         "R3",
            "room_type":       "consultation",
            "status":          "free",
            "current_patient": None,
            "free_at_step":    0,
        },
    ]


# ---------------------------------------------------------------------------
# Main Environment
# ---------------------------------------------------------------------------

class PetclinicEnvironment(Environment):
    """
    PetClinic RL Environment.

    The agent operates a veterinary clinic for one simulated day.
    At each step, the agent sees the current clinic state and
    picks one action. The episode runs for MAX_STEPS steps.

    At episode end, three graders evaluate the same history:
      - Task 1: Were appointments scheduled correctly?
      - Task 2: Were walk-ins triaged by urgency?
      - Task 3: Were all clinic operations completed efficiently?
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0
        self._seed = None

        # Episode data
        self._step = 0
        self._doctors: List[Dict] = []
        self._rooms: List[Dict] = []
        self._appointment_requests: List[Dict] = []
        self._scheduled_appointments: List[Dict] = []
        self._walkin_queue: List[Dict] = []
        self._patients_in_treatment: List[Dict] = []
        self._pending_lab_results: List[Dict] = []
        self._patients_ready_for_discharge: List[Dict] = []
        self._prescriptions_to_dispense: List[Dict] = []
        self._supplies = dict(SUPPLIES_INITIAL)
        self._discharged_patients: List[Dict] = []
        self._referred_patients: List[Dict] = []

        # Tracking for graders
        self._history: List[Dict] = []
        self._task1_events: List[Dict] = []
        self._task2_events: List[Dict] = []
        self._task3_events: List[Dict] = []

        # Running reward accumulator
        self._episode_reward = 0.0

    # ── Reset ────────────────────────────────────────────────────────────

    def reset(self) -> PetclinicObservation:
        """Start a new clinic day episode."""
        random.seed(self._seed)
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count += 1
        self._step = 0
        self._episode_reward = 0.0

        # Generate clinic resources
        self._doctors = _make_doctors()
        self._rooms = _make_rooms()
        self._supplies = dict(SUPPLIES_INITIAL)

        # Generate patients
        self._appointment_requests = [
            _make_patient(f"P{i:03d}", is_walkin=False)
            for i in range(1, NUM_APPOINTMENT_REQUESTS + 1)
        ]
        # Walk-ins arrive mid-episode — start empty, populated in _advance
        self._walkin_queue = []
        self._walkin_pool = [
            _make_patient(f"W{i:03d}", is_walkin=True)
            for i in range(1, NUM_WALKINS + 1)
        ]
        self._walkin_arrival_steps = sorted(
            random.sample(range(3, MAX_STEPS - 3), min(NUM_WALKINS, MAX_STEPS - 6))
        )

        # Clear all tracking
        self._scheduled_appointments = []
        self._patients_in_treatment = []
        self._pending_lab_results = []
        self._patients_ready_for_discharge = []
        self._prescriptions_to_dispense = []
        self._discharged_patients = []
        self._referred_patients = []
        self._history = []
        self._task1_events = []
        self._task2_events = []
        self._task3_events = []

        return self._build_observation(reward=0.0)

    # ── Step ─────────────────────────────────────────────────────────────

    def step(self, action: PetclinicAction) -> PetclinicObservation:  # type: ignore[override]
        """Execute one agent action and advance the clinic simulation."""
        self._state.step_count += 1
        self._step += 1

        # Validate action is currently legal
        valid = self._get_valid_actions()
        if action.action_type not in valid:
            reward = PENALTY_INVALID_ACTION
            self._record_history(action, reward, "invalid_action")
            return self._build_observation(reward=reward)

        # Dispatch to handler
        reward, event_type = self._dispatch(action)
        self._episode_reward += reward
        self._record_history(action, reward, event_type)

        # Advance simulation — update timers, release rooms/doctors,
        # add walk-ins, generate prescriptions and lab results
        self._advance_simulation()

        # Check if episode is done
        done = self._step >= MAX_STEPS
        obs = self._build_observation(reward=reward, done=done)

        if done:
            # Run all three graders and attach final scores
            obs.final_task1_score = self._grade_task1()
            obs.final_task2_score = self._grade_task2()
            obs.final_task3_score = self._grade_task3()

        return obs

    # ── Action dispatch ──────────────────────────────────────────────────

    def _dispatch(self, action: PetclinicAction):
        """Route action to correct handler. Returns (reward, event_type)."""
        handlers = {
            "schedule_appointment":  self._handle_schedule,
            "insert_walkin":         self._handle_insert_walkin,
            "queue_walkin":          self._handle_queue_walkin,
            "refer_patient":         self._handle_refer,
            "assign_doctor":         self._handle_assign_doctor,
            "dispense_prescription": self._handle_dispense,
            "file_lab_result":       self._handle_file_lab,
            "discharge_patient":     self._handle_discharge,
            "schedule_followup":     self._handle_followup,
            "wait":                  lambda a: (REWARD_WAIT, "wait"),
        }
        handler = handlers.get(action.action_type)
        if handler:
            return handler(action)
        return PENALTY_INVALID_ACTION, "invalid"

    # ── Action handlers ──────────────────────────────────────────────────

    def _handle_schedule(self, action: PetclinicAction):
        """Task 1 — Book an appointment request into a slot."""
        patient = self._find_patient(action.patient_id, self._appointment_requests)
        doctor  = self._find_resource(action.doctor_id, self._doctors)
        room    = self._find_resource(action.room_id, self._rooms)

        if not patient or not doctor or not room:
            return PENALTY_INVALID_ACTION, "schedule_invalid"

        # Check double booking — same doctor or room at same time
        for appt in self._scheduled_appointments:
            if appt["time_slot"] == action.time_slot:
                if appt["doctor_id"] == action.doctor_id:
                    self._task1_events.append({"type": "double_book", "patient_id": patient["patient_id"]})
                    return PENALTY_DOUBLE_BOOK, "double_book"
                if appt["room_id"] == action.room_id:
                    self._task1_events.append({"type": "double_book_room", "patient_id": patient["patient_id"]})
                    return PENALTY_DOUBLE_BOOK, "double_book_room"

        reward = REWARD_CORRECT_SCHEDULE

        # Check doctor specialisation match
        spec_needed = patient.get("needs_specialist")
        if spec_needed and doctor["specialisation"] != spec_needed and doctor["specialisation"] != "general":
            reward += PENALTY_WRONG_DOCTOR
            self._task1_events.append({"type": "wrong_doctor", "patient_id": patient["patient_id"]})
        else:
            self._task1_events.append({"type": "correct_doctor", "patient_id": patient["patient_id"]})

        # Check room type match
        if patient.get("needs_surgery") and room["room_type"] != "surgery":
            reward += PENALTY_WRONG_ROOM
            self._task1_events.append({"type": "wrong_room", "patient_id": patient["patient_id"]})
        else:
            self._task1_events.append({"type": "correct_room", "patient_id": patient["patient_id"]})

        # Book the appointment
        self._scheduled_appointments.append({
            "patient_id": patient["patient_id"],
            "doctor_id":  action.doctor_id,
            "room_id":    action.room_id,
            "time_slot":  action.time_slot,
            "severity":   patient["severity"],
        })
        patient["status"] = "scheduled"
        self._appointment_requests = [
            p for p in self._appointment_requests
            if p["patient_id"] != patient["patient_id"]
        ]

        return max(0.0, reward), "schedule_ok"

    def _handle_insert_walkin(self, action: PetclinicAction):
        """Task 2 — Insert walk-in immediately into treatment."""
        patient = self._find_patient(action.patient_id, self._walkin_queue)
        doctor  = self._find_resource(action.doctor_id, self._doctors, status="available")
        room    = self._find_resource(action.room_id, self._rooms, status="free")

        if not patient or not doctor or not room:
            return PENALTY_INVALID_ACTION, "insert_invalid"

        reward = REWARD_CORRECT_TRIAGE

        # Critical walk-in inserted immediately — correct
        # Minor walk-in inserted when critical patients are also waiting — wrong
        critical_waiting = any(
            p["severity"] == "critical"
            for p in self._walkin_queue
            if p["patient_id"] != patient["patient_id"]
        )
        if patient["severity"] == "minor" and critical_waiting:
            reward = PENALTY_WRONG_TRIAGE
            self._task2_events.append({"type": "minor_over_critical", "patient_id": patient["patient_id"]})
        else:
            self._task2_events.append({"type": "correct_insert", "patient_id": patient["patient_id"],
                                        "severity": patient["severity"]})

        # Move patient to treatment
        doctor["status"] = "busy"
        doctor["current_patient"] = patient["patient_id"]
        doctor["busy_until"] = self._step + patient["treatment_steps_remaining"]
        doctor["fatigue"] = min(100, doctor["fatigue"] + 15)

        room["status"] = "occupied"
        room["current_patient"] = patient["patient_id"]
        room["free_at_step"] = self._step + patient["treatment_steps_remaining"]

        patient["status"] = "in_treatment"
        self._patients_in_treatment.append(patient)
        self._walkin_queue = [p for p in self._walkin_queue if p["patient_id"] != patient["patient_id"]]

        return reward, "insert_ok"

    def _handle_queue_walkin(self, action: PetclinicAction):
        """Task 2 — Add walk-in to waiting queue without immediate assignment."""
        patient = self._find_patient(action.patient_id, self._walkin_queue)
        if not patient:
            return PENALTY_INVALID_ACTION, "queue_invalid"

        # Correct to queue minor/moderate walk-ins when no resources free
        # Wrong to queue critical walk-ins when a doctor is available
        available_doctor = any(d["status"] == "available" for d in self._doctors)
        if patient["severity"] == "critical" and available_doctor:
            self._task2_events.append({"type": "queued_critical", "patient_id": patient["patient_id"]})
            return PENALTY_WRONG_TRIAGE, "queued_critical"

        self._task2_events.append({"type": "correct_queue", "patient_id": patient["patient_id"]})
        return REWARD_CORRECT_TRIAGE * 0.5, "queue_ok"

    def _handle_refer(self, action: PetclinicAction):
        """Task 2/3 — Refer patient to external specialist."""
        patient = (
            self._find_patient(action.patient_id, self._walkin_queue)
            or self._find_patient(action.patient_id, self._appointment_requests)
        )
        if not patient:
            return PENALTY_INVALID_ACTION, "refer_invalid"

        # Only refer truly complex cases
        if patient["severity"] == "minor":
            self._task2_events.append({"type": "unnecessary_referral", "patient_id": patient["patient_id"]})
            return -0.15, "unnecessary_refer"

        patient["status"] = "referred"
        self._referred_patients.append(patient)
        self._walkin_queue = [p for p in self._walkin_queue if p["patient_id"] != patient["patient_id"]]
        self._appointment_requests = [p for p in self._appointment_requests if p["patient_id"] != patient["patient_id"]]

        self._task2_events.append({"type": "correct_referral", "patient_id": patient["patient_id"]})
        return 0.10, "refer_ok"

    def _handle_assign_doctor(self, action: PetclinicAction):
        """Task 3 — Reassign a free doctor to a waiting patient."""
        patient = (
            self._find_patient(action.patient_id, self._walkin_queue)
            or self._find_patient(action.patient_id, self._appointment_requests)
        )
        doctor = self._find_resource(action.doctor_id, self._doctors, status="available")
        room   = self._find_resource(action.room_id, self._rooms, status="free")

        if not patient or not doctor or not room:
            return PENALTY_INVALID_ACTION, "assign_invalid"

        doctor["status"] = "busy"
        doctor["current_patient"] = patient["patient_id"]
        doctor["busy_until"] = self._step + patient["treatment_steps_remaining"]
        doctor["fatigue"] = min(100, doctor["fatigue"] + 10)

        room["status"] = "occupied"
        room["current_patient"] = patient["patient_id"]
        room["free_at_step"] = self._step + patient["treatment_steps_remaining"]

        patient["status"] = "in_treatment"
        self._patients_in_treatment.append(patient)
        self._walkin_queue = [p for p in self._walkin_queue if p["patient_id"] != patient["patient_id"]]
        self._appointment_requests = [p for p in self._appointment_requests if p["patient_id"] != patient["patient_id"]]

        self._task3_events.append({"type": "doctor_assigned", "patient_id": patient["patient_id"]})
        return 0.15, "assign_ok"

    def _handle_dispense(self, action: PetclinicAction):
        """Task 3 — Dispense prescription to a ready patient."""
        patient = self._find_patient(action.patient_id, self._prescriptions_to_dispense)
        if not patient:
            return PENALTY_INVALID_ACTION, "dispense_invalid"

        # Consume supply
        condition = patient.get("condition", "")
        supply_map = {
            "infection":   "antibiotics",
            "surgery":     "anaesthesia",
            "injury":      "bandages",
            "vaccination": "vaccines",
        }
        supply_key = supply_map.get(condition, "bandages")
        if self._supplies.get(supply_key, 0) <= 0:
            self._task3_events.append({"type": "supply_depleted", "patient_id": patient["patient_id"]})
            return -0.10, "supply_depleted"

        self._supplies[supply_key] -= 1
        patient["prescription_ready"] = False
        self._prescriptions_to_dispense = [
            p for p in self._prescriptions_to_dispense
            if p["patient_id"] != patient["patient_id"]
        ]

        # Move to ready for discharge
        self._patients_ready_for_discharge.append(patient)
        self._patients_in_treatment = [
            p for p in self._patients_in_treatment
            if p["patient_id"] != patient["patient_id"]
        ]

        self._task3_events.append({"type": "prescription_dispensed", "patient_id": patient["patient_id"]})
        return REWARD_PRESCRIPTION_DONE, "dispense_ok"

    def _handle_file_lab(self, action: PetclinicAction):
        """Task 3 — File a lab result to the correct patient."""
        lab = next(
            (l for l in self._pending_lab_results
             if l["lab_id"] == action.lab_result_id and not l["filed"]),
            None
        )
        if not lab:
            return PENALTY_INVALID_ACTION, "lab_invalid"

        # Check if filed to correct patient
        if lab["patient_id"] == action.patient_id:
            lab["filed"] = True
            self._task3_events.append({"type": "lab_filed_correct", "lab_id": lab["lab_id"]})
            return REWARD_LAB_FILED, "lab_ok"
        else:
            self._task3_events.append({"type": "lab_filed_wrong", "lab_id": lab["lab_id"]})
            return -0.15, "lab_wrong"

    def _handle_discharge(self, action: PetclinicAction):
        """Task 3 — Discharge a treated patient."""
        patient = self._find_patient(action.patient_id, self._patients_ready_for_discharge)
        if not patient:
            return PENALTY_INVALID_ACTION, "discharge_invalid"

        # Check prescription was dispensed first
        still_needs_prescription = any(
            p["patient_id"] == action.patient_id
            for p in self._prescriptions_to_dispense
        )
        if still_needs_prescription:
            self._task3_events.append({"type": "discharged_without_prescription",
                                        "patient_id": patient["patient_id"]})
            return -0.20, "discharge_no_prescription"

        patient["status"] = "discharged"
        self._discharged_patients.append(patient)
        self._patients_ready_for_discharge = [
            p for p in self._patients_ready_for_discharge
            if p["patient_id"] != patient["patient_id"]
        ]

        # Free the room
        for room in self._rooms:
            if room["current_patient"] == patient["patient_id"]:
                room["status"] = "free"
                room["current_patient"] = None
                room["free_at_step"] = self._step

        self._task3_events.append({"type": "discharged", "patient_id": patient["patient_id"]})
        return REWARD_CORRECT_DISCHARGE, "discharge_ok"

    def _handle_followup(self, action: PetclinicAction):
        """Task 3 — Schedule a follow-up for a chronic case."""
        patient = self._find_patient(action.patient_id, self._discharged_patients)
        if not patient:
            return PENALTY_INVALID_ACTION, "followup_invalid"

        if not patient.get("followup_needed", False):
            return -0.05, "followup_unnecessary"

        if patient.get("followup_booked", False):
            return -0.05, "followup_already_booked"

        patient["followup_booked"] = True
        self._task3_events.append({"type": "followup_booked", "patient_id": patient["patient_id"]})
        return REWARD_FOLLOWUP_BOOKED, "followup_ok"

    # ── Simulation advance ───────────────────────────────────────────────

    def _advance_simulation(self):
        """Advance clinic timers by one step."""

        # Release doctors and rooms whose timer has expired
        for doctor in self._doctors:
            if doctor["status"] == "busy" and doctor["busy_until"] <= self._step:
                patient_id = doctor["current_patient"]
                doctor["status"] = "available"
                doctor["current_patient"] = None

                # Move patient to prescription / ready for discharge
                patient = self._find_patient(patient_id, self._patients_in_treatment)
                if patient:
                    if patient.get("lab_result_pending"):
                        # Add lab result
                        self._pending_lab_results.append({
                            "lab_id":     f"L{len(self._pending_lab_results)+1:03d}",
                            "patient_id": patient_id,
                            "filed":      False,
                        })
                        patient["lab_result_pending"] = False

                    if patient.get("condition") in ["infection", "surgery", "injury", "vaccination"]:
                        patient["prescription_ready"] = True
                        self._prescriptions_to_dispense.append(patient)
                    else:
                        self._patients_ready_for_discharge.append(patient)

                    self._patients_in_treatment = [
                        p for p in self._patients_in_treatment
                        if p["patient_id"] != patient_id
                    ]

        # Release rooms
        for room in self._rooms:
            if room["status"] == "occupied" and room["free_at_step"] <= self._step:
                room["status"] = "free"
                room["current_patient"] = None

        # Add scheduled patients to treatment if their time has come
        time_idx = min(self._step, len(TIME_SLOTS) - 1)
        current_time = TIME_SLOTS[time_idx]
        for appt in list(self._scheduled_appointments):
            if appt["time_slot"] == current_time:
                doctor = self._find_resource(appt["doctor_id"], self._doctors, status="available")
                room   = self._find_resource(appt["room_id"],   self._rooms,   status="free")
                if doctor and room:
                    patient = self._find_patient_by_id(appt["patient_id"])
                    if patient and patient["status"] == "scheduled":
                        doctor["status"] = "busy"
                        doctor["current_patient"] = patient["patient_id"]
                        doctor["busy_until"] = self._step + patient["treatment_steps_remaining"]
                        doctor["fatigue"] = min(100, doctor["fatigue"] + 10)
                        room["status"] = "occupied"
                        room["current_patient"] = patient["patient_id"]
                        room["free_at_step"] = self._step + patient["treatment_steps_remaining"]
                        patient["status"] = "in_treatment"
                        self._patients_in_treatment.append(patient)

        # Increase wait steps for patients in queues
        for patient in self._walkin_queue + self._appointment_requests:
            patient["wait_steps"] += 1

        # Arrive walk-in patients at scheduled steps
        while self._walkin_arrival_steps and self._walkin_arrival_steps[0] <= self._step:
            self._walkin_arrival_steps.pop(0)
            if self._walkin_pool:
                walkin = self._walkin_pool.pop(0)
                self._walkin_queue.append(walkin)

        # Idle doctor penalty — if doctor free but patients waiting
        available_doctors = [d for d in self._doctors if d["status"] == "available"]
        waiting_patients  = self._walkin_queue + self._appointment_requests
        if available_doctors and waiting_patients:
            self._episode_reward += PENALTY_IDLE_DOCTOR

    # ── Valid actions ────────────────────────────────────────────────────

    def _get_valid_actions(self) -> List[str]:
        """Return list of action_types the agent can legally use right now."""
        valid = ["wait"]

        if self._appointment_requests:
            available_doc  = any(d["status"] == "available" for d in self._doctors)
            free_room      = any(r["status"] == "free" for r in self._rooms)
            if available_doc and free_room:
                valid.append("schedule_appointment")

        if self._walkin_queue:
            available_doc = any(d["status"] == "available" for d in self._doctors)
            free_room     = any(r["status"] == "free" for r in self._rooms)
            if available_doc and free_room:
                valid.append("insert_walkin")
                valid.append("assign_doctor")
            valid.append("queue_walkin")
            valid.append("refer_patient")

        if self._prescriptions_to_dispense:
            valid.append("dispense_prescription")

        if self._pending_lab_results:
            valid.append("file_lab_result")

        if self._patients_ready_for_discharge:
            valid.append("discharge_patient")

        if any(
            p.get("followup_needed") and not p.get("followup_booked")
            for p in self._discharged_patients
        ):
            valid.append("schedule_followup")

        return list(set(valid))

    # ── Graders ──────────────────────────────────────────────────────────

    def _grade_task1(self) -> float:
        """
        Task 1 — Appointment Scheduling (easy)

        Measures: correct bookings / total booking opportunities.
        Penalises double-bookings, wrong doctor type, wrong room type.
        Score range: 0.0 – 1.0
        """
        if not self._task1_events:
            # No scheduling happened — minimal score
            unscheduled = len(self._appointment_requests)
            if unscheduled == NUM_APPOINTMENT_REQUESTS:
                return 0.05
            return 0.1

        correct  = sum(1 for e in self._task1_events if e["type"] in
                       ["correct_doctor", "correct_room"])
        wrong    = sum(1 for e in self._task1_events if e["type"] in
                       ["double_book", "double_book_room", "wrong_doctor", "wrong_room"])
        total    = correct + wrong

        if total == 0:
            return 0.1

        base_score = correct / total

        # Bonus: scheduled all requests
        scheduled_count = len(self._scheduled_appointments)
        completion_bonus = min(0.2, scheduled_count / NUM_APPOINTMENT_REQUESTS * 0.2)

        return min(1.0, base_score + completion_bonus)

    def _grade_task2(self) -> float:
        """
        Task 2 — Walk-in Triage (medium)

        Task 1 criteria + measures urgency handling.
        Did critical walk-ins get treated before minor ones?
        Score range: 0.0 – 1.0
        """
        task1_score = self._grade_task1()

        if not self._task2_events:
            walkins_arrived = NUM_WALKINS - len(self._walkin_pool)
            if walkins_arrived == 0:
                # No walk-ins arrived yet — base on Task 1 only
                return task1_score * 0.5
            return 0.1

        correct_triage = sum(1 for e in self._task2_events if e["type"] in
                             ["correct_insert", "correct_queue", "correct_referral"])
        wrong_triage   = sum(1 for e in self._task2_events if e["type"] in
                             ["minor_over_critical", "queued_critical",
                              "unnecessary_referral"])
        total = correct_triage + wrong_triage

        triage_score = correct_triage / total if total > 0 else 0.5

        # Weighted: 40% Task 1, 60% triage accuracy
        return min(1.0, 0.40 * task1_score + 0.60 * triage_score)

    def _grade_task3(self) -> float:
        """
        Task 3 — Full Day Operations (hard)

        Task 1 + Task 2 + measures:
          - Patients discharged vs total treated
          - Lab results filed correctly
          - Prescriptions dispensed on time
          - Follow-ups booked for chronic cases
          - No supply depletions
        Score range: 0.0 – 1.0
        """
        task2_score = self._grade_task2()

        if not self._task3_events:
            return task2_score * 0.4

        # Discharge rate
        total_treated   = len(self._discharged_patients) + len(self._patients_ready_for_discharge)
        discharged      = len(self._discharged_patients)
        discharge_rate  = discharged / total_treated if total_treated > 0 else 0.0

        # Lab filing accuracy
        total_labs   = len(self._pending_lab_results)
        filed_correct = sum(1 for l in self._pending_lab_results if l.get("filed"))
        lab_score    = filed_correct / total_labs if total_labs > 0 else 1.0

        # Prescription completion
        dispensed = sum(1 for e in self._task3_events if e["type"] == "prescription_dispensed")
        total_rx  = dispensed + sum(1 for e in self._task3_events if e["type"] == "supply_depleted")
        rx_score  = dispensed / total_rx if total_rx > 0 else 1.0

        # Follow-up bookings
        followup_needed = sum(
            1 for p in self._discharged_patients if p.get("followup_needed")
        )
        followups_booked = sum(
            1 for p in self._discharged_patients
            if p.get("followup_needed") and p.get("followup_booked")
        )
        followup_score = followups_booked / followup_needed if followup_needed > 0 else 1.0

        # Weighted combination
        ops_score = (
            0.35 * discharge_rate +
            0.25 * lab_score +
            0.25 * rx_score +
            0.15 * followup_score
        )

        # Final: 30% Task2, 70% full operations
        return min(1.0, 0.30 * task2_score + 0.70 * ops_score)

    # ── Helpers ──────────────────────────────────────────────────────────

    def _find_patient(
        self, patient_id: Optional[str], pool: List[Dict]
    ) -> Optional[Dict]:
        if not patient_id:
            return None
        return next((p for p in pool if p["patient_id"] == patient_id), None)

    def _find_patient_by_id(self, patient_id: str) -> Optional[Dict]:
        all_patients = (
            self._appointment_requests
            + self._walkin_queue
            + self._patients_in_treatment
            + self._patients_ready_for_discharge
            + self._discharged_patients
            + self._referred_patients
        )
        return next((p for p in all_patients if p["patient_id"] == patient_id), None)

    def _find_resource(
        self,
        resource_id: Optional[str],
        pool: List[Dict],
        status: Optional[str] = None,
    ) -> Optional[Dict]:
        if not resource_id:
            return None
        for item in pool:
            id_key = "doctor_id" if "doctor_id" in item else "room_id"
            if item[id_key] == resource_id:
                if status is None or item["status"] == status:
                    return item
        return None

    def _record_history(self, action: PetclinicAction, reward: float, event: str):
        self._history.append({
            "step":        self._step,
            "action_type": action.action_type,
            "patient_id":  action.patient_id,
            "doctor_id":   action.doctor_id,
            "room_id":     action.room_id,
            "reward":      reward,
            "event":       event,
        })

    def _build_context(self) -> str:
        """Build plain English description of current clinic state for the LLM."""
        parts = []

        # Time and progress
        time_idx = min(self._step, len(TIME_SLOTS) - 1)
        parts.append(
            f"Step {self._step}/{MAX_STEPS} | Time: {TIME_SLOTS[time_idx]}"
        )

        # Doctors
        for d in self._doctors:
            if d["status"] == "available":
                parts.append(f"{d['name']} ({d['specialisation']}) is AVAILABLE.")
            else:
                parts.append(
                    f"{d['name']} is BUSY with {d['current_patient']}, "
                    f"free at step {d['busy_until']}. Fatigue: {d['fatigue']}%."
                )

        # Rooms
        for r in self._rooms:
            if r["status"] == "free":
                parts.append(f"Room {r['room_id']} ({r['room_type']}) is FREE.")
            else:
                parts.append(
                    f"Room {r['room_id']} ({r['room_type']}) OCCUPIED "
                    f"until step {r['free_at_step']}."
                )

        # Queues
        if self._appointment_requests:
            parts.append(
                f"Appointment requests waiting: "
                + ", ".join(
                    f"{p['patient_id']}({p['severity']})"
                    for p in self._appointment_requests
                )
            )

        if self._walkin_queue:
            parts.append(
                "Walk-in patients: "
                + ", ".join(
                    f"{p['patient_id']}({p['severity']}, waited {p['wait_steps']} steps)"
                    for p in self._walkin_queue
                )
            )

        if self._prescriptions_to_dispense:
            parts.append(
                "Prescriptions to dispense: "
                + ", ".join(p["patient_id"] for p in self._prescriptions_to_dispense)
            )

        if self._pending_lab_results:
            parts.append(
                "Lab results to file: "
                + ", ".join(
                    f"{l['lab_id']} → {l['patient_id']}"
                    for l in self._pending_lab_results
                    if not l["filed"]
                )
            )

        if self._patients_ready_for_discharge:
            parts.append(
                "Ready to discharge: "
                + ", ".join(p["patient_id"] for p in self._patients_ready_for_discharge)
            )

        parts.append(f"Supplies: {self._supplies}")

        return " | ".join(parts)

    def _build_observation(
        self, reward: float = 0.0, done: bool = False
    ) -> PetclinicObservation:
        """Construct the full observation from current state."""

        # Running scores
        t1 = self._grade_task1() if self._task1_events else 0.0
        t2 = self._grade_task2() if self._task2_events else 0.0
        t3 = self._grade_task3() if self._task3_events else 0.0

        return PetclinicObservation(
            # Progress
            step=self._step,
            max_steps=MAX_STEPS,
            time_of_day=TIME_SLOTS[min(self._step, len(TIME_SLOTS) - 1)],

            # Task 1
            appointment_requests=list(self._appointment_requests),
            scheduled_appointments=list(self._scheduled_appointments),

            # Task 2
            walkin_queue=list(self._walkin_queue),

            # Task 3
            patients_in_treatment=list(self._patients_in_treatment),
            pending_lab_results=list(self._pending_lab_results),
            patients_ready_for_discharge=list(self._patients_ready_for_discharge),
            prescriptions_to_dispense=list(self._prescriptions_to_dispense),

            # Resources
            doctors=list(self._doctors),
            rooms=list(self._rooms),
            supplies_remaining=dict(self._supplies),

            # Running scores
            task1_score_so_far=round(t1, 3),
            task2_score_so_far=round(t2, 3),
            task3_score_so_far=round(t3, 3),

            # Action mask
            valid_actions=self._get_valid_actions(),

            # Terminal
            done=done,
            reward=reward,

            # LLM-readable context
            context=self._build_context(),
            metadata={
                "episode_id":         self._state.episode_id,
                "total_reward":       round(self._episode_reward, 3),
                "discharged_count":   len(self._discharged_patients),
                "referred_count":     len(self._referred_patients),
            },
        )

    # ── State property ───────────────────────────────────────────────────

    @property
    def state(self) -> State:
        return self._state