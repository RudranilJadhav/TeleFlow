from groq import Groq
import os
import json
from dotenv import load_dotenv
from typing import Dict, Any
from datetime import datetime

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

MODEL = "llama-3.3-70b-versatile"
with open("../utils/momgeneratorprompt.txt", "r") as f:
            SYSTEM_PROMPT = f.read()

def generate_mom(chat_history: str) -> Dict[str, Any]:
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": chat_history}
            ],
            temperature=0.2,
            max_tokens=700,
        )

        raw = response.choices[0].message.content.strip()

        json_start = raw.find("{")
        json_end = raw.rfind("}") + 1
        cleaned = raw[json_start:json_end]

        return json.loads(cleaned)

    except Exception as e:
        print("MoM Generation Error:", e)
        return {
            "customer_name": None,
            "city": None,
            "configuration": None,
            "budget_range": None,
            "timeline": None,
            "customer_intent": None,
            "lead_quality": "Cold",
            "other_notes": [],
            "next_action": None,
            "sales_stage": "Qualification"
        }


def generate_mom_document(mom: Dict[str, Any]) -> str:
    today = datetime.now().strftime("%d %B %Y")

    def bullet(text):
        return f"• {text}"

    # --- Key Discussion Points (only qualitative insights) ---
    key_points = []

    if mom.get("customer_intent"):
        key_points.append(mom["customer_intent"])

    if mom.get("other_notes"):
        key_points.append(mom["other_notes"])

    key_points_text = (
        "\n".join(bullet(p) for p in key_points)
        if key_points
        else bullet("No significant discussion points captured.")
    )

    # --- Customer Requirements (pure facts, no commentary) ---
    requirements = []

    if mom.get("city"):
        requirements.append(f"Location: {mom['city']}")

    if mom.get("configuration"):
        requirements.append(f"Configuration: {mom['configuration']}")

    if mom.get("budget_range"):
        requirements.append(f"Budget Range: {mom['budget_range']}")

    if mom.get("timeline"):
        requirements.append(f"Timeline: {mom['timeline']}")

    requirements_text = (
        "\n".join(bullet(r) for r in requirements)
        if requirements
        else bullet("No explicit requirements identified.")
    )

    # --- Action Items (deterministic + LLM suggested) ---
    action_items = []

    lead_quality = mom.get("lead_quality", "Cold")

    default_actions = {
        "Hot": "Schedule site visit or booking discussion.",
        "Warm": "Share relevant property options and follow up.",
        "Cold": "Re-engage later to reassess interest."
    }

    action_items.append(default_actions.get(lead_quality, default_actions["Cold"]))

    if mom.get("next_action"):
        action_items.append(mom["next_action"])

    action_items_text = "\n".join(bullet(a) for a in action_items)

    # --- Final Document ---
    return f"""
STARK REAL ESTATE
Minutes of Meeting (MoM)
Date: {today}

Customer Name: {mom.get('customer_name') or 'Not Captured'}
Lead Quality: {lead_quality}
Sales Stage: {mom.get('sales_stage')}

1. Key Discussion Summary
{key_points_text}

2. Customer Requirements
{requirements_text}

3. Action Items
{action_items_text}
""".strip()
