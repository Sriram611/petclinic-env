---
title: PetClinic Env Environment Server
colorFrom: green
colorTo: blue
sdk: docker
pinned: true
app_port: 8000
base_path: /web
tags:
  - openenv
  - reinforcement-learning
  - veterinary
  - scheduling
  - healthcare
---

# PetClinic-Env

> **An OpenEnv-compatible reinforcement learning environment where an AI agent
> automates the day-to-day operations of a veterinary clinic.**

Built for the **Meta PyTorch OpenEnv Hackathon 2026**
- Scaler School of Technology x Meta x HuggingFace

---

## What Is This?

PetClinic-Env simulates one full working day at a veterinary clinic.
The AI agent acts as a clinic manager - scheduling appointments,
triaging walk-in patients by urgency, dispensing prescriptions,
filing lab results, and discharging treated patients before the day ends.

Every action the agent takes maps to a real task a human receptionist
or clinic manager performs every single day. This is not a game or a toy -
it is a faithful simulation of real clinical workflow.

---

## Why This Is Novel

Veterinary clinic operations involve a class of real-world decision-making
that has never been modelled as a reinforcement learning environment.
PetClinic-Env is the first environment
in the OpenEnv ecosystem that simulates a real healthcare operations
workflow.

The domain is genuinely challenging for AI agents because clinic
management involves cascading dependencies — a prescription cannot be
dispensed until treatment is complete, a patient cannot be discharged
until a prescription is dispensed, and a follow-up cannot be scheduled
until a patient is discharged. These sequential constraints, combined
with concurrent resource management across doctors, rooms, and supplies,
create a decision space that is simple enough to define clearly but
difficult enough to solve optimally.

Unlike scheduling benchmarks that use abstract job queues, PetClinic-Env
grounds every decision in a recognisable human context. The agent reads
patient names, species, conditions, and severity levels — the same
information a real clinic receptionist processes every morning. This
makes the environment immediately interpretable to any evaluator, which
matters for both research reproducibility and real-world deployment.

The three-task structure is also novel in its design. Rather than running
separate episodes for each task, all three graders evaluate the same
episode history simultaneously — meaning the agent cannot optimise for
one task at the expense of another. An agent that schedules appointments
perfectly but ignores walk-in triage will score well on Task 1 and poorly
on Task 2 and 3. This joint evaluation more accurately reflects how
real clinic managers work, where every decision affects multiple
performance dimensions at once.

---

## Three Tasks - One Episode

The agent plays one episode. At the end, three graders evaluate
the same history from three different angles:

### Task 1 - Appointment Scheduling (Easy)
**Objective:** Book overnight appointment requests into today's schedule
without double-booking doctors or rooms, matching the right doctor
specialisation and room type to each patient.

**What the grader measures:**
- Correct doctor-patient specialisation matches
- Correct room-patient type matches (surgery vs general vs consultation)
- Zero double-bookings at the same time slot
- Completion rate (how many requests were scheduled)

**Expected scores:** Random agent ~0.25 | Good agent 0.85+

---

### Task 2 - Walk-in Triage (Medium)
**Objective:** Task 1 + handle unexpected walk-in patients mid-episode
by urgency. Critical walk-ins must be seen before moderate or minor ones.

**What the grader measures:**
- Task 1 score (weighted 40%)
- Walk-in urgency ordering (weighted 60%)
- Critical patients never queued when a doctor is available
- Minor patients not inserted ahead of critical ones

**Expected scores:** Random agent ~0.30 | Good agent 0.70+

---

### Task 3 - Full Day Operations (Hard)
**Objective:** Task 1 + Task 2 + manage the complete treatment pipeline.
Dispense prescriptions, file lab results to correct patients, discharge
all treated patients, and schedule follow-up appointments for chronic cases.

**What the grader measures:**
- Task 2 score (weighted 30%)
- Patient discharge rate (weighted 25%)
- Lab result filing accuracy (weighted 25%)
- Prescription dispensing completion (weighted 25%)
- Follow-up booking rate (weighted 15%)

**Expected scores:** Random agent ~0.15 | Good agent 0.60+

---

## Action Space

The agent can take one action per step:

| Action | Description | Task |
|---|---|---|
| `schedule_appointment` | Book a patient into a time slot with a doctor and room | 1 |
| `insert_walkin` | Immediately assign a walk-in patient to a free doctor and room | 2 |
| `queue_walkin` | Add a walk-in to the waiting list without immediate assignment | 2 |
| `refer_patient` | Send a complex case to an external specialist | 2/3 |
| `assign_doctor` | Reassign a free doctor to a waiting patient | 3 |
| `dispense_prescription` | Dispense medicine to a patient whose treatment is complete | 3 |
| `file_lab_result` | Match a lab result to the correct patient record | 3 |
| `discharge_patient` | Discharge a fully treated patient from the clinic | 3 |
| `schedule_followup` | Book a follow-up appointment for a chronic case | 3 |
| `wait` | Take no action this step | any |

**Action mask:** The `valid_actions` field in every observation lists
exactly which actions are legal at the current step. The agent should
only attempt actions from this list.

---

## Observation Space

Every step the agent receives a `PetclinicObservation` containing:

| Field | Type | Description |
|---|---|---|
| `step` | int | Current step (0-20) |
| `time_of_day` | str | Simulated clinic time e.g. "09:30" |
| `appointment_requests` | list | Unbooked patients needing scheduling |
| `scheduled_appointments` | list | Already booked appointments |
| `walkin_queue` | list | Unexpected walk-in patients waiting |
| `patients_in_treatment` | list | Patients currently being treated |
| `pending_lab_results` | list | Lab results needing to be filed |
| `patients_ready_for_discharge` | list | Treated patients waiting for discharge |
| `prescriptions_to_dispense` | list | Prescriptions ready to hand out |
| `doctors` | list | Doctor availability, specialisation, fatigue |
| `rooms` | list | Room status and type |
| `supplies_remaining` | dict | Stock of vaccines, bandages, anaesthesia, antibiotics |
| `task1_score_so_far` | float | Running Task 1 score |
| `task2_score_so_far` | float | Running Task 2 score |
| `task3_score_so_far` | float | Running Task 3 score |
| `valid_actions` | list | Action mask - only attempt these |
| `context` | str | Plain English summary of current clinic state |
| `done` | bool | True when episode ends |
| `reward` | float | Reward earned this step |
| `final_task1_score` | float | Final Task 1 score (set when done=True) |
| `final_task2_score` | float | Final Task 2 score (set when done=True) |
| `final_task3_score` | float | Final Task 3 score (set when done=True) |

---

## Reward Signals

| Event | Reward |
|---|---|
| Correct appointment scheduled | +0.20 |
| Correct walk-in triage decision | +0.25 |
| Patient discharged successfully | +0.20 |
| Lab result filed correctly | +0.15 |
| Prescription dispensed | +0.15 |
| Follow-up appointment booked | +0.10 |
| Double-booking detected | -0.30 |
| Wrong doctor specialisation | -0.20 |
| Wrong room type | -0.20 |
| Critical patient missed | -0.40 |
| Minor patient over critical | -0.25 |
| Invalid action attempted | -0.10 |
| Doctor idle with patients waiting | -0.05 |

---

## Project Structure

```
petclinic_env/
 Dockerfile                           Container (root level, required)
 inference.py                         Baseline script (root level, required)
 README.md                            This file
 models.py                            Pydantic Action + Observation models
 client.py                            WebSocket EnvClient
 openenv.yaml                         OpenEnv spec metadata
 pyproject.toml                       Package definition
 uv.lock                              Locked dependencies
 server/
     app.py                           FastAPI server (create_app)
     petclinic_env_environment.py     Core game logic + graders
```

---

## Quick Start

### Option 1 - Run locally

```bash
# 1. Clone or download the project
cd petclinic_env

# 2. Install dependencies
uv sync

# 3. Start the environment server
uv run uvicorn server.app:app --host 0.0.0.0 --port 8000

# 4. In a new terminal, set environment variables
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your_huggingface_token"
export ENV_URL="http://localhost:8000"

# 5. Run the inference script
uv run python inference.py
```

### Option 2 - Using .env file

Create a `.env` file in the project root:
```
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
HF_TOKEN=your_huggingface_token
ENV_URL=http://localhost:8000
```

Then run:
```bash
uv sync
uv run uvicorn server.app:app --host 0.0.0.0 --port 8000
uv run python inference.py
```

### Option 3 - Connect to HuggingFace Space

```bash
export ENV_URL="https://your-username-petclinic-env.hf.space"
uv run python inference.py
```

---

## Validate the Environment

```bash
# Start server first, then run
openenv validate --url http://localhost:8000
```

---

## Deploy to HuggingFace Spaces

```bash
# From inside the petclinic_env directory
openenv push --repo-id your-username/petclinic-env
```

After deployment your space will be available at:
`https://huggingface.co/spaces/your-username/petclinic-env`

The deployed space includes:
- **Web interface** at `/web` - interactive UI for exploring the environment
- **API docs** at `/docs` - full OpenAPI interface
- **Health check** at `/health` - deployment verification
- **WebSocket** at `/ws` - persistent session for inference

---

## Environment Details

### Clinic Setup (per episode)
- **2 doctors:** Dr. Priya (small animal specialist) and Dr. Raj (large animal specialist)
- **3 rooms:** General examination, Surgery, Consultation
- **6 appointment requests** arrive at episode start
- **4 walk-in patients** arrive at random steps between step 3 and step 17
- **20 steps** per episode (simulates one clinic day 09:00-17:00)

### Patient Properties
Each patient has: species (dog/cat/rabbit/bird), condition, severity (critical/moderate/minor),
specialisation requirement, surgery requirement, and flags for lab results, prescriptions,
and follow-up needs.

### Resource Constraints
- Doctors have fatigue that increases each busy step
- Supplies (vaccines, bandages, anaesthesia, antibiotics) are limited
- Rooms can only hold one patient at a time
- Treatments take 1-3 steps to complete

---

## Baseline Scores

Scores produced by `Qwen/Qwen2.5-72B-Instruct` via HuggingFace router:

| Task | Score | Description |
|---|---|---|
| Task 1 | ~0.72 | Appointment Scheduling |
| Task 2 | ~0.58 | Walk-in Triage |
| Task 3 | ~0.41 | Full Day Operations |

Scores vary per episode due to random patient generation and walk-in timing.

---

## Team

Built by **Team DEXTER** for the
Meta PyTorch OpenEnv Hackathon 2026
Scaler School of Technology, India

- Surendra Purohit A (Team Lead)
- Sriram R
- Sitharthan M

---

## License

BSD 3-Clause License - same as OpenEnv core.
