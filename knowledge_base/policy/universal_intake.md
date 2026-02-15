---
category: intake_policy
type: universal_intake
used_by: guidance_llm
version: 1.0
---

This document defines the standard intake questions TrackSanté must use for any user concern.
The goal is to collect sufficient information to:

- generate a doctor-facing summary
- identify potential red flags
- determine whether TrackSanté measurements are relevant

The assistant should ask approximately 5–7 questions total.

---

## Core Intake Questions

These questions should be asked in logical order, adapting to the user's answers.

### 1. Main concern
Ask:
- What is the main symptom or concern?
- Where is it located?

Example:
"What is bothering you most right now?"

---

### 2. Onset and duration
Ask:
- When did this start?
- Has it been constant or intermittent?

---

### 3. Severity and functional impact
Ask:
- How severe is it (0–10)?
- Does it affect daily activities?

---

### 4. Timing and pattern
Ask if relevant:
- Does it happen at rest or during activity?
- Is it worse at certain times of day?

---

### 5. Triggers and context
Ask if relevant:
- Does anything make it better or worse?
- Does it occur indoors, outdoors, or both?

---

### 6. Associated symptoms
Ask only relevant ones:

Examples:
- shortness of breath
- dizziness
- fever or chills
- fatigue
- palpitations

---

### 7. Relevant history (only if relevant)
Ask if needed:

Examples:
- known medical conditions
- recent illness
- relevant allergies

Do NOT ask excessive medical history unrelated to the complaint.

---

## Intake Completion Criteria

The intake is considered sufficient when:

- the main symptom is clearly defined
- onset and severity are known
- relevant associated symptoms have been screened
- enough information exists to choose a measurement OR recommend in-person care

The assistant should then:

- recommend a TrackSanté sensor, OR
- recommend in-person evaluation if no measurement is appropriate.