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
SYSTEM_PROMPT = """
You are a Real Estate Sales Call Analyzer.

Generate a structured Minutes of Meeting (MoM).

Return ONLY valid JSON in the exact schema below:

{
  "customer_name": string | null,
  "city": string | null,
  "configuration": string | null,
  "budget_range": string | null,
  "timeline": string | null,
  "customer_intent": string | null,
  "lead_quality": "Hot" | "Warm" | "Cold",
  "other_notes": string | null,
  "next_action": string | null,
  "sales_stage": "Qualification" | "Recommendation" | "Objection Handling" | "Closing"
}

Rules:
- Do NOT hallucinate.
- If information missing, use null.
- No markdown.
- Only JSON.
"""

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

        # Hard JSON isolation
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

    key_points = []

    if mom.get("city"):
        key_points.append(f"Customer expressed interest in properties in {mom['city']}")

    if mom.get("configuration"):
        key_points.append(f"Preferred configuration: {mom['configuration']}")

    if mom.get("budget_range"):
        key_points.append(f"Budget discussed: {mom['budget_range']}")

    if mom.get("timeline"):
        key_points.append(f"Purchase timeline: {mom['timeline']}")

    if mom.get("customer_intent"):
        key_points.append(f"Customer intent: {mom['customer_intent']}")

    if mom.get("other_notes"):
        key_points.append(f"Other Notes: {mom['other_notes']}")

    key_points_text = (
        "\n".join([f"• {p}" for p in key_points])
        if key_points else "• No major discussion points recorded."
    )

    requirements_text = (
        f"• Location: {mom.get('city')}\n"
        f"• Configuration: {mom.get('configuration')}\n"
        f"• Budget: {mom.get('budget_range')}\n"
        f"• Timeline: {mom.get('timeline')}"
    )

    action_items = []

    lead_quality = mom.get("lead_quality", "Cold")

    if lead_quality == "Hot":
        action_items.append("Schedule site visit immediately.")
    elif lead_quality == "Warm":
        action_items.append("Share shortlisted property options.")
    else:
        action_items.append("Plan follow-up to assess interest.")

    if mom.get("next_action"):
        action_items.append(mom["next_action"])

    action_items_text = "\n".join([f"• {a}" for a in action_items])

    return f"""
STARK REAL ESTATE
Minutes of Meeting (MoM)
Date: {today}

Customer Name: {mom.get('customer_name')}
Lead Quality: {lead_quality}
Sales Stage: {mom.get('sales_stage')}

1. Key Discussion Points
{key_points_text}

2. Customer Requirements Identified
{requirements_text}

3. Action Items for Sales Team
{action_items_text}
""".strip()
