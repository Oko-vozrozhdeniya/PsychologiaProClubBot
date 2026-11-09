# Dialogue action evaluation — 9 November 2026

Status: experimental only. No candidate from this lab is used by ordinary private
conversation. Project documents on Google Drive have not been updated by this change.

## Confirmed evidence before this change

Base release: `7873ed95170c55d3031661523cd3797e212a4e02` (PR 17).
The owner supplied two live Groq candidate reports from that release:

- First: 3/4 actions correct; grounded-topic case failed with HTTP 400.
- Repeat after a pause: 1/4 correct; fatigue case failed with HTTP 400,
  sufficiency without context incorrectly closed the conversation, and grounded
  topic incorrectly requested clarification. Clear farewell passed.

The earlier 3/4 result does not establish reliability. HTTP failures have no
semantic result. HTTP 400 alone does not identify quota pressure or overload.
The old report discarded the sanitized provider error type/code; the exact cause
of those historical failures cannot be recovered from the screenshots.

## Changes and limits

The action policy now continues when the referent is resolved and no closing act
exists. Previously multiple candidate areas forced clarification even in that
case. This fixes a deterministic decision error; it does not fix a model falsely
claiming that the user explicitly ended the conversation.

The native Groq adapter can now accept an explicit structured-task model,
response schema, token budget, and reasoning effort. These overrides require an
explicit system instruction and parser. Existing callers retain their parameters.
HTTP errors retain only sanitized machine type/code; remote messages and
`failed_generation` are never included in the Telegram report. Error body reads
are bounded to 64 KiB so typical failed-generation bodies do not truncate the
JSON before its machine code can be parsed.

`/dialogue_lab verify` is owner-only, private-only, and runs in the background:

1. Compare `openai/gpt-oss-20b` and `openai/gpt-oss-120b` under identical settings:
   strict native JSON schema, temperature 0, medium reasoning, 4096 output tokens.
   Returned reasoning is disabled; it is not printed or stored.
2. For each model, run four smoke cases twice as independent requests. Only a
   perfect round advances to the next stage. After both pass, run six additional
   cases that were not used in those smoke rounds.
3. Use sequential calls, at least six seconds between starts, hard cap 28 requests.
   This pacing does not guarantee compatibility with every account quota.
4. Stop a profile on any API/format failure without retry. Stop the whole run on
   HTTP 429 or transient server errors. On a semantic failure, finish the current
   round and disqualify that profile from later stages.
5. Report every attempted case and safe error identifiers. No model is silently
   selected or promoted; environment, ordinary AI mode, chat history and DB are
   not modified. The command needs only the configured Groq key.

Strict schema controls output structure, not meaning. Both new model profiles
use a larger reasoning budget than the legacy lab; this is not a single-variable
experiment proving why the old HTTP 400 occurred. A 14/14 score applies only to
these ten distinct synthetic cases with four repeated. It does not certify
natural responses, full memory behavior, or universal intent recognition.

The semantic instruction is unchanged. User utterances are fixtures, not runtime
few-shot examples or phrase matching rules. Normal answers remain model-generated.

Groq documentation consulted:
- https://console.groq.com/docs/structured-outputs
- https://console.groq.com/docs/reasoning

## Next acceptance step

Deploy this release through the normal Bothost rebuild, then invoke
`/dialogue_lab verify` once. Do not repeat the old four-case command unchanged.
Use the returned machine codes to distinguish transport/format errors from
semantic failures. Any profile that fails remains experimental. Even a profile
passing all stages needs evaluation of generated replies and the normal consent,
session, retrieval and memory path before integration.

Offline tests validate transport payloads, schema parsing, decision boundaries,
repetition gates, request limits, early stops and private owner access. They do
not establish that a real model understands any test utterance. No live provider
keys are available in the development workspace.
