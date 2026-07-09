# Backend Runtime Config

`llm_settings.json` and `gestures_26.json` are runtime files and should not be committed.

For the hand-training gesture library:

1. Review `gestures_26.example.json` with the clinical rehabilitation team.
2. Copy it to `gestures_26.json` only after the names, indications, force wording, and safety notes are approved.
3. Restart the backend.

Until `gestures_26.json` exists, reports keep the gesture-plan section in placeholder mode and do not ask the LLM to prescribe concrete gestures.
