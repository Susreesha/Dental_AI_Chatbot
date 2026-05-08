from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load API key
load_dotenv()

app = Flask(__name__)
CORS(app)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------- CHAT FUNCTION ----------------
def calculate_risk(context):
    severity = context.get("severity")
    duration = context.get("duration")
    conditions = context.get("conditions", [])

    if conditions:
        return "high"

    if severity == "severe" or duration == ">2w":
        return "high"

    if severity == "mild" and duration == "<1w":
        return "low"

    return "medium"
def chat(context, history):
    context["risk"] = calculate_risk(context)

    # 🟢 Safety check
    if not context:
        return "No input received."

    # 🟢 Build history text
    history_text = ""

    if history:
        history_text = "Previous interactions:\n"

        for h in history:
            complaint = h.get("complaint", "unknown")
            severity = h.get("severity", "unknown")
            duration = h.get("duration", "unknown")

            history_text += f"- {complaint} ({severity}, {duration})\n"

    # 🟢 Build structured input
    structured_input = f"""
Patient details:

Complaint: {context.get('complaint')}
Severity: {context.get('severity')}
Duration: {context.get('duration')}
Medical Conditions: {', '.join(context.get('conditions', [])) if context.get('conditions') else 'None'}
Risk Level: {context.get('risk')}

{history_text}
"""

    # 🟢 OpenAI call
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """
You are a calm, friendly dental assistant helping patients understand their symptoms.

You will receive structured patient data.

STRICT RESPONSE FORMAT:

1. Start with a short emotional connection (1–2 lines)
2. Add heading: "Common causes:"
3. Give 3–5 bullet points relevant to complaint
4. If risk is LOW or MEDIUM:
   - Add simple habits (brushing, flossing, hygiene tips)
5. If risk is HIGH:
   - Gently encourage visiting a dentist soon
6. End with:
   "Would you like help with consultation or booking?"

STYLE:
- Warm, reassuring
- No long paragraphs
- Use bullet points
- Use simple language

RULES:
- No diagnosis
- No medicines
- No fear-based tone
- If conditions present → prioritize safety
- If history exists → acknowledge briefly
"""
            },
            {
                "role": "user",
                "content": structured_input
            }
        ]
    )

    return response.choices[0].message.content


# ---------------- API ROUTE ----------------
@app.route("/chat", methods=["POST"])
def chat_api():
    data = request.json

    context = data.get("context")
    history = data.get("history", [])

    reply = chat(context, history)

    return jsonify({"response": reply})


# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)
