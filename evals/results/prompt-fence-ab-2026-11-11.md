# Prompt fencing: dev set check, 2026-11-11

The answer prompt now fences the question and every excerpt as data (commit 98d441f and the
rag/prompt_safety.py helpers). A prompt change needs an eval before it ships (ADR 0001), so the dev
set ran on Vertex (`gemini-2.5-flash`, location `global`, top_k 8, scope stage 2 off).

The first fenced run is the record (`e2e-gemini-2.5-flash-2026-11-11.json`). The same day, the
fenced prompt ran once more and the previous prompt (9374bd7) once, as a control, on the same
backend and settings.

| run | prompt | correct answer | correct refuse | false refuse | false answer | citation validity | errors |
| --- | --- | --- | --- | --- | --- | --- | --- |
| record | fenced | 46 | 11 | 3 (CID-11, DR-02, RT-07) | 0 | 39/46 | 0 |
| repeat | fenced | 47 | 11 | 2 (CID-11, RT-07) | 0 | 40/47 | 0 |
| control | previous | 44 | 11 | 4 (CID-11, RT-01, RT-07, SP-02) | 0 | 38/44 | 1 (CID-08 timed out) |
| 2026-09-02 | previous | 48 | 11 | 1 (CID-11) | 0 | 41/48 | 0 |

What this shows:
- No run gave a false answer.
- The fenced prompt refused no more than the previous prompt did on the same day. RT-07 and
  CID-11 were refused under both prompts.
- False refusals vary between 2 and 4 across same-day runs, against 1 on 2026-09-02. That
  difference comes with the backend and the date, not with the fence. The 2026-09-02 run is not a
  like-for-like baseline for today's numbers.

Files: `ab-2026-11-11/e2e-gemini-2.5-flash-2026-11-11-fenced-repeat.json` and
`ab-2026-11-11/e2e-gemini-2.5-flash-2026-11-11-unfenced-control.json`, kept out of the gate's folder so it reads only the record.
