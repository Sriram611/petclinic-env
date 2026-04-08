"""
Inference Script — PetClinic Environment
=========================================
MANDATORY environment variables:
    API_BASE_URL        The API endpoint for the LLM.
    MODEL_NAME          The model identifier to use for inference.
    HF_TOKEN            Your Hugging Face / API key.
    LOCAL_IMAGE_NAME    Docker image name (optional — uses ENV_URL if not set)

STDOUT FORMAT (hackathon required lines — validator looks for these):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

NOTE: All other printed lines (episode summaries, score tables, etc.)
      are ignored by the automated validator — only [START], [STEP],
      and [END] lines are parsed.
"""

import asyncio
import json
import os
import sys
import textwrap
from typing import List, Optional, Tuple

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

try:
    from client import PetclinicEnv
    from models import PetclinicAction, PetclinicObservation
except ImportError:
    from petclinic_env.client import PetclinicEnv
    from petclinic_env.models import PetclinicAction, PetclinicObservation


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
API_KEY          = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL     = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME       = os.getenv("MODEL_NAME")  or "Qwen/Qwen2.5-72B-Instruct"
ENV_URL          = os.getenv("ENV_URL")     or "http://localhost:8000"

TASK_NAME        = os.getenv("PETCLINIC_TASK",      "full-day-operations")
BENCHMARK        = os.getenv("PETCLINIC_BENCHMARK",  "petclinic")

MAX_STEPS               = 20
TEMPERATURE             = 0.2
MAX_TOKENS              = 300
SUCCESS_SCORE_THRESHOLD = 0.4
NUM_EPISODES            = 3


# ---------------------------------------------------------------------------
# Mandatory hackathon log functions
# These exact formats are parsed by the automated validator
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    """Required: one [START] line at episode begin."""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    """Required: one [STEP] line per step, immediately after env.step()."""
    error_val = error if error else "null"
    done_val  = str(done).lower()          # must be lowercase: true or false
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(
    success: bool,
    steps: int,
    score: float,
    rewards: List[float],
) -> None:
    """Required: one [END] line after env.close(), always emitted."""
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert veterinary clinic manager AI.
    You are operating a pet clinic for one full working day.

    Your THREE goals (all evaluated on the same episode):

    TASK 1 — Appointment Scheduling (easy):
      Book appointment requests into correct slots.
      Match doctor specialisation and room type. Never double-book.

    TASK 2 — Walk-in Triage (medium):
      Handle unexpected walk-ins by urgency.
      CRITICAL patients always go before MODERATE or MINOR.
      Use insert_walkin (not assign_doctor) for walk-in patients.

    TASK 3 — Full Day Operations (hard):
      Dispense prescriptions before discharging patients.
      File lab results to correct patients.
      Discharge all treated patients before episode ends.
      Schedule follow-ups for chronic cases.

    MATCHING RULES:
      needs_specialist=small_animal → doctor_id=D1 (Dr. Priya)
      needs_specialist=large_animal → doctor_id=D2 (Dr. Raj)
      needs_surgery=true            → room_id=R2 (surgery room)
      no surgery needed             → room_id=R1 or R3

    ACTION CHOICE RULES:
      Walk-in patient in walkin_queue → use insert_walkin or queue_walkin
      Patient not yet in any queue   → use assign_doctor
      Never discharge without dispensing prescription first.
      Only use actions listed in valid_actions.

    RESPONSE FORMAT:
      Reply ONLY with a single valid JSON object. No markdown. No explanation.
      {
        "action_type": "one of the valid_actions",
        "patient_id": "patient ID or null",
        "doctor_id": "D1 or D2 or null",
        "room_id": "R1, R2, or R3 or null",
        "time_slot": "HH:MM or null",
        "lab_result_id": "lab ID or null",
        "reasoning": "one sentence"
      }

      If nothing useful can be done:
      {"action_type": "wait", "reasoning": "No valid action available"}
""").strip()


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_prompt(
    obs: PetclinicObservation,
    step: int,
    last_reward: float,
    history: List[str],
) -> str:
    lines = []
    lines.append(f"Step: {step}/{obs.max_steps} | Time: {obs.time_of_day}")
    lines.append(f"Last reward: {last_reward:.2f}")
    lines.append(
        f"Running scores: T1={obs.task1_score_so_far:.2f} "
        f"T2={obs.task2_score_so_far:.2f} "
        f"T3={obs.task3_score_so_far:.2f}"
    )
    lines.append("")
    lines.append("SITUATION:")
    lines.append(obs.context)
    lines.append("")

    if obs.appointment_requests:
        lines.append("APPOINTMENT REQUESTS (use schedule_appointment):")
        for p in obs.appointment_requests:
            lines.append(
                f"  {p['patient_id']} | {p['species']} | {p['condition']} | "
                f"severity={p['severity']} | "
                f"preferred={p.get('preferred_time','any')} | "
                f"surgery={p.get('needs_surgery',False)} | "
                f"specialist={p.get('needs_specialist','none')}"
            )
        lines.append("")

    if obs.walkin_queue:
        lines.append("WALK-IN PATIENTS (use insert_walkin or queue_walkin):")
        for p in obs.walkin_queue:
            lines.append(
                f"  {p['patient_id']} | {p['species']} | "
                f"severity={p['severity']} | "
                f"waited={p.get('wait_steps',0)} steps"
            )
        lines.append("")

    if obs.prescriptions_to_dispense:
        lines.append("PRESCRIPTIONS TO DISPENSE (use dispense_prescription):")
        for p in obs.prescriptions_to_dispense:
            lines.append(f"  {p['patient_id']} | {p['condition']}")
        lines.append("")

    if obs.pending_lab_results:
        lines.append("LAB RESULTS TO FILE (use file_lab_result):")
        for l in obs.pending_lab_results:
            if not l.get("filed"):
                lines.append(
                    f"  lab_id={l['lab_id']} → patient={l['patient_id']}"
                )
        lines.append("")

    if obs.patients_ready_for_discharge:
        lines.append("READY FOR DISCHARGE (use discharge_patient):")
        for p in obs.patients_ready_for_discharge:
            lines.append(
                f"  {p['patient_id']} | "
                f"followup_needed={p.get('followup_needed',False)}"
            )
        lines.append("")

    lines.append("DOCTORS:")
    for d in obs.doctors:
        if d["status"] == "available":
            lines.append(
                f"  {d['doctor_id']} ({d['name']}) — AVAILABLE | "
                f"specialisation={d['specialisation']}"
            )
        else:
            lines.append(
                f"  {d['doctor_id']} ({d['name']}) — BUSY | "
                f"free at step {d.get('busy_until','?')} | "
                f"fatigue={d.get('fatigue',0)}%"
            )
    lines.append("")

    lines.append("ROOMS:")
    for r in obs.rooms:
        if r["status"] == "free":
            lines.append(f"  {r['room_id']} ({r['room_type']}) — FREE")
        else:
            lines.append(
                f"  {r['room_id']} ({r['room_type']}) — "
                f"OCCUPIED until step {r.get('free_at_step','?')}"
            )
    lines.append("")
    lines.append(f"SUPPLIES: {obs.supplies_remaining}")
    lines.append("")
    lines.append(f"VALID ACTIONS: {obs.valid_actions}")
    lines.append("")

    if history:
        lines.append("RECENT STEPS:")
        for h in history[-3:]:
            lines.append(f"  {h}")
        lines.append("")

    lines.append("Reply with JSON only. Use ONLY one of the VALID ACTIONS.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM call with fallback rule-based agent
# ---------------------------------------------------------------------------

def rule_based_action(obs: PetclinicObservation) -> PetclinicAction:
    """
    Simple rule-based fallback agent used when LLM credits are depleted.
    Follows clinic priority rules deterministically.
    """
    valid = obs.valid_actions

    # Priority 1 — dispense prescriptions (blocks discharge)
    if "dispense_prescription" in valid and obs.prescriptions_to_dispense:
        p = obs.prescriptions_to_dispense[0]
        return PetclinicAction(
            action_type="dispense_prescription",
            patient_id=p["patient_id"],
            reasoning="Rule: dispense prescription to unblock discharge",
        )

    # Priority 2 — file lab results
    if "file_lab_result" in valid and obs.pending_lab_results:
        for l in obs.pending_lab_results:
            if not l.get("filed"):
                return PetclinicAction(
                    action_type="file_lab_result",
                    patient_id=l["patient_id"],
                    lab_result_id=l["lab_id"],
                    reasoning="Rule: file lab result to correct patient",
                )

    # Priority 3 — discharge ready patients
    if "discharge_patient" in valid and obs.patients_ready_for_discharge:
        p = obs.patients_ready_for_discharge[0]
        return PetclinicAction(
            action_type="discharge_patient",
            patient_id=p["patient_id"],
            reasoning="Rule: discharge treated patient",
        )

    # Priority 4 — insert critical walk-ins immediately
    if "insert_walkin" in valid and obs.walkin_queue:
        critical = [p for p in obs.walkin_queue if p["severity"] == "critical"]
        target   = critical[0] if critical else obs.walkin_queue[0]
        avail_doc = next(
            (d for d in obs.doctors if d["status"] == "available"), None
        )
        free_room = next(
            (r for r in obs.rooms if r["status"] == "free"), None
        )
        if avail_doc and free_room:
            return PetclinicAction(
                action_type="insert_walkin",
                patient_id=target["patient_id"],
                doctor_id=avail_doc["doctor_id"],
                room_id=free_room["room_id"],
                reasoning="Rule: insert walk-in by urgency",
            )

    # Priority 5 — schedule appointments
    if "schedule_appointment" in valid and obs.appointment_requests:
        p = sorted(
            obs.appointment_requests,
            key=lambda x: ["critical","moderate","minor"].index(x["severity"])
        )[0]
        avail_doc = next(
            (d for d in obs.doctors if d["status"] == "available"), None
        )
        free_room = next(
            (r for r in obs.rooms
             if r["status"] == "free" and (
                 not p.get("needs_surgery") or r["room_type"] == "surgery"
             )), None
        )
        if avail_doc and free_room:
            from server.petclinic_env_environment import TIME_SLOTS
            used_slots = {a["time_slot"] for a in obs.scheduled_appointments
                         if a.get("doctor_id") == avail_doc["doctor_id"]}
            slot = next(
                (s for s in TIME_SLOTS if s not in used_slots),
                TIME_SLOTS[0]
            )
            return PetclinicAction(
                action_type="schedule_appointment",
                patient_id=p["patient_id"],
                doctor_id=avail_doc["doctor_id"],
                room_id=free_room["room_id"],
                time_slot=slot,
                reasoning="Rule: schedule highest severity appointment",
            )

    # Priority 6 — schedule follow-ups
    if "schedule_followup" in valid:
        return PetclinicAction(
            action_type="schedule_followup",
            reasoning="Rule: book follow-up for chronic case",
        )

    return PetclinicAction(action_type="wait", reasoning="Rule: nothing to do")


def get_model_action(
    client: OpenAI,
    obs: PetclinicObservation,
    step: int,
    last_reward: float,
    history: List[str],
) -> Tuple[PetclinicAction, Optional[str]]:
    """
    Try LLM first. If credits depleted or any error, fall back to rule-based agent.
    Returns (action, error_or_None).
    """
    try:
        prompt     = build_prompt(obs, step, last_reward, history)
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()

        # Strip markdown fences
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()

        data        = json.loads(text)
        action_type = data.get("action_type", "wait")

        if action_type not in obs.valid_actions:
            fallback = rule_based_action(obs)
            return fallback, f"invalid_action:{action_type}"

        return PetclinicAction(
            action_type   = action_type,
            patient_id    = data.get("patient_id"),
            doctor_id     = data.get("doctor_id"),
            room_id       = data.get("room_id"),
            time_slot     = data.get("time_slot"),
            lab_result_id = data.get("lab_result_id"),
            reasoning     = data.get("reasoning", ""),
        ), None

    except Exception as e:
        err_str = str(e)
        # Credits depleted — use rule-based agent silently
        if "402" in err_str or "credits" in err_str.lower():
            fallback = rule_based_action(obs)
            return fallback, "credits_depleted:using_rule_agent"
        fallback = rule_based_action(obs)
        return fallback, f"llm_error:{err_str[:40]}"


# ---------------------------------------------------------------------------
# Run one episode
# ---------------------------------------------------------------------------

async def run_episode(
    env: PetclinicEnv,
    client: OpenAI,
    episode_num: int,
) -> dict:
    """Run one full episode. Prints both hackathon format and detailed format."""

    # ── Detailed header (for human readers / judges) ──────────────────
    print(f"\n{'='*60}", flush=True)
    print(f"EPISODE {episode_num}", flush=True)
    print(f"{'='*60}", flush=True)

    history:     List[str]   = []
    rewards:     List[float] = []
    steps_taken: int         = 0
    score:       float       = 0.0
    success:     bool        = False
    last_reward: float       = 0.0

    # Mandatory [START] line
    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset()
        obs: PetclinicObservation = result.observation

        print(f"Environment reset. Episode starting...", flush=True)
        print(f"Appointment requests: {len(obs.appointment_requests)}", flush=True)
        print(f"Doctors available:    "
              f"{sum(1 for d in obs.doctors if d['status']=='available')}",
              flush=True)
        print(f"Rooms available:      "
              f"{sum(1 for r in obs.rooms if r['status']=='free')}",
              flush=True)

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            # ── Detailed step info (human readable) ───────────────────
            active_tasks = []
            if obs.appointment_requests:
                active_tasks.append("Task1(Scheduling)")
            if obs.walkin_queue:
                active_tasks.append("Task2(Triage)")
            if (obs.prescriptions_to_dispense
                    or obs.pending_lab_results
                    or obs.patients_ready_for_discharge):
                active_tasks.append("Task3(Operations)")
            if not active_tasks:
                active_tasks.append("All tasks — waiting for events")

            print(f"\n[Step {step:02d}/{MAX_STEPS}] | Time: {obs.time_of_day}",
                  flush=True)
            print(f"   Active tasks   : {' + '.join(active_tasks)}", flush=True)
            print(f"   Appointments   : {len(obs.appointment_requests)} pending | "
                  f"{len(obs.scheduled_appointments)} booked", flush=True)
            print(f"   Walk-ins       : {len(obs.walkin_queue)} waiting", flush=True)
            print(f"   Prescriptions  : {len(obs.prescriptions_to_dispense)} "
                  f"to dispense", flush=True)
            print(f"   Lab results    : {len(obs.pending_lab_results)} to file",
                  flush=True)
            print(f"   For discharge  : "
                  f"{len(obs.patients_ready_for_discharge)} ready", flush=True)
            print(f"   Running scores : T1={obs.task1_score_so_far:.2f} | "
                  f"T2={obs.task2_score_so_far:.2f} | "
                  f"T3={obs.task3_score_so_far:.2f}", flush=True)
            print(f"   Valid actions  : {obs.valid_actions}", flush=True)

            # Get action
            action, error = get_model_action(
                client, obs, step, last_reward, history
            )

            # Send to environment
            result      = await env.step(action)
            obs         = result.observation
            reward      = result.reward or 0.0
            done        = result.done
            steps_taken = step
            last_reward = reward

            rewards.append(reward)

            # ── Detailed action output ────────────────────────────────
            action_str = f"action={action.action_type}"
            if action.patient_id:
                action_str += f" | patient={action.patient_id}"
            if action.doctor_id:
                action_str += f" | doctor={action.doctor_id}"
            if action.room_id:
                action_str += f" | room={action.room_id}"
            print(f"   → {action_str}", flush=True)
            if action.reasoning:
                print(f"   → reasoning: {action.reasoning}", flush=True)
            print(f"   reward={reward:.3f} | done={str(done).lower()}",
                  flush=True)

            # Mandatory [STEP] line — validator reads this
            log_step(
                step   = step,
                action = action.action_type,
                reward = reward,
                done   = done,
                error  = error,
            )

            history.append(
                f"Step {step}: {action.action_type} "
                f"patient={action.patient_id} → reward {reward:+.2f}"
            )

            if done:
                break

        # ── Final scores ──────────────────────────────────────────────
        t1 = (obs.final_task1_score
              if obs.final_task1_score is not None
              else obs.task1_score_so_far)
        t2 = (obs.final_task2_score
              if obs.final_task2_score is not None
              else obs.task2_score_so_far)
        t3 = (obs.final_task3_score
              if obs.final_task3_score is not None
              else obs.task3_score_so_far)

        # Overall score = average of 3 task scores, clamped [0, 1]
        score   = min(max((t1 + t2 + t3) / 3.0, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)

        # Mandatory [END] line — always emitted even on exception
        log_end(
            success = success,
            steps   = steps_taken,
            score   = score,
            rewards = rewards,
        )

    # ── Detailed episode summary (human readable) ─────────────────────
    total_reward = sum(rewards)
    print(f"\n{'─'*60}", flush=True)
    print(f"EPISODE {episode_num} COMPLETE", flush=True)
    print(f"  Steps taken    : {steps_taken}", flush=True)
    print(f"  Total reward   : {total_reward:.3f}", flush=True)
    print(f"  Task 1 score   : {t1:.4f}  (Appointment Scheduling)",
          flush=True)
    print(f"  Task 2 score   : {t2:.4f}  (Walk-in Triage)", flush=True)
    print(f"  Task 3 score   : {t3:.4f}  (Full Day Operations)", flush=True)
    print(f"{'─'*60}", flush=True)

    return {
        "episode":      episode_num,
        "steps":        steps_taken,
        "total_reward": round(total_reward, 3),
        "task1_score":  round(t1, 4),
        "task2_score":  round(t2, 4),
        "task3_score":  round(t3, 4),
        "score":        round(score, 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    print("\n" + "="*60, flush=True)
    print("  PETCLINIC-ENV — BASELINE INFERENCE", flush=True)
    print("  OpenEnv Hackathon 2026 · Scaler School of Technology",
          flush=True)
    print("="*60 + "\n", flush=True)

    # Validate config
    if not API_KEY:
        print("❌ HF_TOKEN not set. Check your .env file.", flush=True)
        sys.exit(1)

    print(f"✅ API_BASE_URL : {API_BASE_URL}", flush=True)
    print(f"✅ MODEL_NAME   : {MODEL_NAME}", flush=True)
    print(
        f"✅ HF_TOKEN     : "
        f"{'*'*8}{API_KEY[-4:] if len(API_KEY) > 4 else '****'}",
        flush=True,
    )
    print(f"✅ ENV_URL      : {ENV_URL}", flush=True)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    print(f"\n✅ LLM client ready ({MODEL_NAME})", flush=True)

    # Connect to environment
    if LOCAL_IMAGE_NAME:
        env = await PetclinicEnv.from_docker_image(LOCAL_IMAGE_NAME)
    else:
        env = PetclinicEnv(base_url=ENV_URL)

    print(f"✅ Connected to environment at {ENV_URL}\n", flush=True)

    all_results = []
    for episode_num in range(1, NUM_EPISODES + 1):
        result = await run_episode(env, client, episode_num)
        all_results.append(result)

    # ── Final baseline table (human readable) ────────────────────────
    print("\n" + "="*60, flush=True)
    print("  BASELINE SCORES — ALL EPISODES", flush=True)
    print("="*60, flush=True)
    print(
        f"  {'Episode':<10} {'Task1':<10} {'Task2':<10} "
        f"{'Task3':<10} {'Score':<10} {'Reward'}",
        flush=True,
    )
    print(
        f"  {'───────':<10} {'─────':<10} {'─────':<10} "
        f"{'─────':<10} {'─────':<10} {'──────'}",
        flush=True,
    )

    avg_t1 = avg_t2 = avg_t3 = avg_score = 0.0
    for r in all_results:
        print(
            f"  {r['episode']:<10} "
            f"{r['task1_score']:<10.4f} "
            f"{r['task2_score']:<10.4f} "
            f"{r['task3_score']:<10.4f} "
            f"{r['score']:<10.4f} "
            f"{r['total_reward']:.3f}",
            flush=True,
        )
        avg_t1    += r["task1_score"]
        avg_t2    += r["task2_score"]
        avg_t3    += r["task3_score"]
        avg_score += r["score"]

    n = len(all_results)
    print(
        f"  {'───────':<10} {'─────':<10} {'─────':<10} "
        f"{'─────':<10} {'─────':<10} {'──────'}",
        flush=True,
    )
    print(
        f"  {'AVERAGE':<10} "
        f"{avg_t1/n:<10.4f} "
        f"{avg_t2/n:<10.4f} "
        f"{avg_t3/n:<10.4f} "
        f"{avg_score/n:<10.4f}",
        flush=True,
    )

    print(f"\n  Task 1 — Appointment Scheduling (easy)", flush=True)
    print(f"  Task 2 — Walk-in Triage          (medium)", flush=True)
    print(f"  Task 3 — Full Day Operations      (hard)", flush=True)

    print("\n" + "="*60, flush=True)
    print("  INFERENCE COMPLETE", flush=True)
    print("="*60 + "\n", flush=True)

    # Save scores to JSON
    output = {
        "model":    MODEL_NAME,
        "env_url":  ENV_URL,
        "episodes": all_results,
        "averages": {
            "task1": round(avg_t1 / n, 4),
            "task2": round(avg_t2 / n, 4),
            "task3": round(avg_t3 / n, 4),
            "score": round(avg_score / n, 4),
        },
    }
    with open("baseline_scores.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"✅ Scores saved to baseline_scores.json\n", flush=True)


if __name__ == "__main__":
    asyncio.run(main())