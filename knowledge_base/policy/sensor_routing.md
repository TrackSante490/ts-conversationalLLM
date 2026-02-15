---
category: routing_policy
type: sensor_selection
used_by: guidance_llm
version: 1.0
---

This document defines how TrackSanté selects which measurement sensor to use.

The assistant must recommend exactly ONE sensor when appropriate.

The assistant must NOT recommend multiple sensors simultaneously.

---

## Pulse Oxygen Sensor (SpO2 + Heart Rate)

Recommend when symptoms involve:

- shortness of breath
- chest discomfort
- chest tightness
- palpitations
- dizziness or lightheadedness
- fainting
- unexplained fatigue

Measurement relevance:
- heart rate abnormalities
- oxygen saturation abnormalities

---

## Temperature Sensor

Recommend when symptoms involve:

- feeling feverish
- chills
- suspected illness
- feeling unusually hot or cold
- fatigue related to illness

Measurement relevance:
- body temperature abnormalities

---

## CO2 Sensor

Recommend when symptoms involve:

- feeling worse indoors
- headaches indoors
- drowsiness in a room
- crowded or enclosed environments
- poor ventilation concerns

Measurement relevance:
- environmental air quality

---

## Humidity Sensor

Recommend when symptoms involve:

- throat dryness
- eye dryness
- irritation in dry environments
- symptoms worsened by heating
- humidifier or dehumidifier concerns

Measurement relevance:
- environmental humidity conditions

---

## No Sensor Match

Recommend NO_SENSOR_MATCH when:

- the concern cannot be evaluated using TrackSanté sensors
- measurements would not provide useful information
- the concern requires in-person medical evaluation

In these cases, recommend in-person evaluation.
