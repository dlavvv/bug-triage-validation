import os
import re
import json
import joblib
import requests
import pandas as pd

# 1. Load the incoming GitHub issue event payload
event_path = os.environ["GITHUB_EVENT_PATH"]
with open(event_path, "r", encoding="utf-8") as f:
    event = json.load(f)

issue = event["issue"]
repo = os.environ["GITHUB_REPOSITORY"]
token = os.environ["GITHUB_TOKEN"]

title = issue.get("title") or ""
body = issue.get("body") or ""
author_assoc = issue.get("author_association") or "NONE"

# 2. Extract exact features used during training
has_code_block = int("```" in body)
has_traceback = int(bool(re.search(r"Traceback|Error:|Exception", body)))
title_length = len(title)
body_length = len(body)

def clean_text(text):
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"###\s*(describe the bug|steps to reproduce|expected behavior|actual results)", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[^a-zA-Z0-9_\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()

combined_text = f"{clean_text(title)} {clean_text(title)} {clean_text(body)}"

X_live = pd.DataFrame([{
    "combined_text": combined_text,
    "author_association": author_assoc,
    "has_code_block": has_code_block,
    "has_traceback": has_traceback,
    "title_length": title_length,
    "body_length": body_length
}])

# 3. Load trained pipeline and predict
model = joblib.load("triage_model.joblib")
pred_label = model.predict(X_live)[0]
probs = model.predict_proba(X_live)[0]
classes = model.classes_
confidence = float(max(probs))

# 4. Apply label via GitHub REST API (fallback to 'needs-triage' if < 60% confident)
headers = {
    "Authorization": f"token {token}",
    "Accept": "application/vnd.github.v3+json"
}

applied_label = pred_label if confidence >= 0.60 else "needs-triage"

requests.post(
    f"[https://api.github.com/repos/](https://api.github.com/repos/){repo}/issues/{issue['number']}/labels",
    headers=headers,
    json={"labels": [applied_label]}
)

# 5. Post a clean triage summary comment
prob_lines = "\n".join(
    [f"  - `{cls}`: {prob:.1%}" for cls, prob in sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)]
)

comment = (
    f"### 🤖 Automated Issue Triage Assessment\n"
    f"- **Predicted Category:** `{pred_label}` (**{confidence:.1%}** confidence)\n"
    f"- **Action Taken:** Applied label `{applied_label}`\n\n"
    f"**Class Probability Breakdown:**\n"
    f"{prob_lines}"
)

requests.post(
    f"[https://api.github.com/repos/](https://api.github.com/repos/){repo}/issues/{issue['number']}/comments",
    headers=headers,
    json={"body": comment}
)
