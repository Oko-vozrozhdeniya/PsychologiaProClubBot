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


## Fixture review after the owner's 05:41 report

Release `3c0dbd52a8caaec017a67a91660057de7a32e3b1` (PR 18) received a live
`/dialogue_lab verify` report: eight requests, no API/format errors. 20B scored
2/4; 120B scored 3/4. 120B chose clarification for both original ambiguous
utterances and farewell for the explicit ending. Both profiles stopped in round
one; repeatability and the additional cases were not tested. The absence of 400
in this run does not establish the cause of historical 400 errors.

Review found that the old grounded-topic fixture was not a sufficiently clear
single-answer gate: after the assistant asks which topic to choose, "Я устал от
этого" could refer to the prior topic or the exchange itself. Its clarification
must not be treated as conclusive model failure. The historical scores remain
unchanged; they are not relabelled as 4/4 successes.

The scored grounded case now explicitly preserves the user's wish to continue
and uses an immediate question naming the subject: the user wants to postpone
accounting and discuss travel, the assistant asks whether discussing accounting
was tiring, and the user confirms with "Да, я устал от этого." The referent and
wish to continue are supported by the conversation. This tests context binding,
rather than requiring a guess about an isolated pronoun.

The exact original fixture is retained in `REVIEW_CASES`, outside pass/fail
scoring and paid requests; a relevant clarification or continuation may be
acceptable, while farewell is unsupported. Definite ambiguity remains scored
in the original context-free fatigue and sufficiency cases. No semantic prompt,
action policy, parser, provider settings, production route, request budget or
promotion rules changed in this fixture-only correction. Do not compare the new
score directly with old totals as if the evaluation inputs were identical.

Next: after installing this correction, run the same verification command once.
No live result for the corrected suite exists yet. Passing synthetic action
selection still does not establish natural dialogue or full memory acceptance.

## Two-stage native-message experiment after corrected-suite failure

On the corrected PR 19 suite the owner observed 3/4 for each model, with no API
or format errors in eight requests. 20B still closed on sufficiency without
context; 120B still marked the now-grounded topic as unresolved. Neither
profile advanced to repetitions. Do not reinterpret these as a combined 4/4.

`/dialogue_lab context` now evaluates a different pipeline on 120B only:

- First reconstruct referents from native, chronologically ordered user and
  assistant messages. The current message is the last actual user message,
  rather than a field alongside nested Gemini `parts` in a JSON prompt string.
- Then independently decide answer/clarify/close from the original native
  dialogue and the separately marked reference observations. The second task
  is not given a preliminary reply-mode verdict or an enumeration of scopes.
- Validate both contracts and exact evidence quotes locally. Stage one may use
  assistant questions to bind referents; closure evidence must come from users.
  A stage-one unresolved referent blocks a stage-two farewell. These checks
  verify quote origin and envelope consistency, not semantic truth.
- Each case uses at most two sequential calls, six seconds apart. Reuse the
  existing two smoke rounds plus six additional cases for a single profile,
  at most 28 API requests; failures stop progression and there are no retries.
  Pacing also applies after a case's final stage. Errors count actual attempted
  calls, including a failure before stage two.

Native message presentation is a confirmed difference in implementation, not a
proven cause of historical semantic errors. The two-stage split is a hypothesis
until tested live. No phrase catalogue, fixed replies, case-specific routing,
production model switch, consent or memory-path change was introduced. The
previous experiments remain available for comparison. The new pipeline is
synthetic-only; no ordinary chat invokes it. Full integration is still blocked
on live action selection and then generated-answer/context-path acceptance.

Next bounded acceptance command after installing this change:
`/dialogue_lab context`. It replaces repeating `/dialogue_lab verify` unchanged.
No live result for this pipeline is available in the development workspace;
provider keys are only configured in the deployed container. Google Drive was
not changed by this work.

## Natural-reply comparison after the two-stage failure

The owner's live `/dialogue_lab context` report on PR 20 showed 3/4, eight API
calls, no format/API errors. Fatigue was clarified only because the unresolved
reference guard overrode stage-two close. Sufficiency was marked resolved and
closed. Clear farewell and grounded topic passed. Repetitions were not reached.
A local parser check also showed that an exact source quote could accompany an
invented referent; quote provenance is not semantic proof. The report did not
expose the actual referent, so its precise content in the live failure is unknown.

The owner supplied a separate ChatGPT conversation showing successful
clarification after an explicit correction, then approved checking direct
natural replies without a preassigned farewell mode. This is a new hypothesis,
not evidence that changing the instructions alone solves it.

`/dialogue_lab replies` now produces actual natural responses, not an action
score. Both arms use native role history, the same private rules plus a concise
ambiguity/wait-for-answer rule, the same 120B generator, temperature 0, medium
reasoning, 4096-token budget and strict JSON containing only `reply`. No
semantic action field or expected answer is passed to the direct generator.
The wrapper validates text structure, not correctness of dialogue.

The baseline arm reproduces the memory planner's reply-mode selection and
closure review on the configured Groq model, then uses that mode in the same
120B response generator. The direct arm skips both preliminary tasks and uses
private reply_mode=answer. Since both receive the additional short rule and a
controlled generator, this is not an exact replay of every production setting
or the entire memory path. No active memory, archive, consent or DB is accessed.

Budget: four cases in both arms (baseline at most three calls, direct one),
independent repetitions of the four direct cases, six other direct cases, then
two independent follow-up branches after the actual first direct fatigue reply:
continue only another topic, or explicitly finish the conversation. At most
28 serial native calls; six seconds between requests; first API/format failure
stops the run and preserves a partial report. Follow-ups test an answer to a
clarification only if the preceding generated response really asked one.

The complete synthetic input contexts, response texts, expected evaluation
criteria, baseline mode trace, partial failure if any, and request count are
returned as `dialogue-replies.txt` in Telegram. Progress messages contain no
semantic pass scores. There is no same-model automated judge hiding or grading
mistakes: generated texts must be reviewed substantively, including repeated
cases and follow-ups. The report does not change the ordinary model, AI mode,
chat history or memory; production integration remains unapproved by evidence.

After deployment the next acceptance command is `/dialogue_lab replies`, not a
repeat of the unchanged action-classification commands. The owner can attach
one text report instead of a sequence of screenshots. No live provider keys
are available in this development workspace. Google Drive was not modified.

## Minimal-instruction control after the real reply report

The owner supplied the complete `dialogue-replies.txt` from PR 21. Eighteen
responses were collected (four baseline, fourteen direct). Both ambiguous core
inputs received endings in both direct rounds. Clear farewell and grounded
context passed. Other problems included ending on "с меня хватит", no recovery
from an assistant-initiated farewell, and an unnecessary question after an
explicit pause. Thus removal of preclassification alone did not solve the issue.

Call 26 failed with Groq HTTP 429, type tokens, code rate_limit_exceeded. Both
follow-up branches have no result. This establishes a token rate-limit error
for this request, not the cause of older HTTP 400 failures. Already collected
responses remain evidence; repeating the 26-request run is unnecessary.

A local audit found the full direct system instruction was 7615 characters and
contained the uncertainty rule. No explicit instruction to end on fatigue was
found. Length alone is not evidence of overload or causation.

`/dialogue_lab minimal` now runs at most six calls: the same first four cases
under a short general conversational prefix plus the exact existing RULE and
FORMAT, then independent repeats of only the two ambiguous cases. Model 120B,
native history, schema, token budget, temperature and reasoning settings are
unchanged. No planner or reply-mode classification is called. The comparison
changes the full BASE+PRIVATE instruction package, not merely its character
count; do not attribute any difference to length alone. No evaluation examples
or phrase catalogue were inserted into instructions.

Calls are serial, 15 seconds apart after completion. Pacing and fewer inputs
reduce load but cannot guarantee the account quota will permit the run. The
first error stops without retry; any collected answers and safe error codes
are returned in dialogue-minimal.txt. Semantic correctness is not auto-scored.
Compare the six replies with the already collected full-instruction report.
A successful short control is not a production policy: privacy and memory
requirements must be preserved in any later integration, and broader acceptance
remains necessary. The experiment does not alter ordinary instructions, active
model, environment, history, consent, archive or memory. Google Drive unchanged.

Next: after installing this diagnostic-only correction, run
`/dialogue_lab minimal` once and provide its text report. No live results for
this control are available in the development environment; keys remain only in
the deployed container. Production conversation behavior is still unresolved.


Control validity: the short profile also retains, verbatim, the existing private
instruction paragraph distinguishing clear farewell, topic ending, uncertainty,
fatigue and pause. A test asserts this paragraph is still a substring of the
full private policy. Thus the minimal run does not silently remove the key
fatigue/ambiguity rule while retaining only a weaker generic statement. The
short prefix plus this paragraph, RULE and FORMAT replace the unrelated full
instruction package. Native payload tests still require every other field and
all conversation messages to remain identical.

## Native plaintext provider control

The owner supplied dialogue-minimal.txt from PR 22: all six calls completed.
Both ambiguous inputs received farewells in both rounds. Explicit farewell was
handled correctly; the grounded-topic answer continued but began with the
insensitive "Отлично!". Shrinking the instruction package did not solve the
core failure under this JSON response protocol. Do not generalize this into
proof that the model can never follow a different instruction or protocol.

A remaining methodological constraint was the strict JSON reply envelope in
all recent natural-response controls. The next diagnostic separates native
text generation from that envelope and obtains an independent provider result.
Groq docs confirm include_reasoning=false suppresses returned reasoning, not a
claim that reasoning was disabled. JSON causation has not been established.

`/dialogue_lab plain` sends the same four core cases to configured Gemini and
Groq 120B using the same short policy, with FORMAT removed, no response schema,
no JSON response format or responseMimeType, and no action classification or
memory planner. Native message roles, temperature 0 and output budget 4096 are
preserved. Groq keeps medium reasoning; Gemini uses its model-native reasoning
configuration. This is a bounded protocol/provider control, not a strict model
ranking with identical reasoning implementations. Removing the JSON request
instruction and constrained decoding together does not isolate those two
subfactors individually.

The common Groq adapter adds explicit response_task=plain, gated by an explicit
instruction and text parser and rejecting response_schema. Default ordinary
and structured callers retain their previous paths. Native text is checked
only for presence, bounds and truncation; there is no semantic validator or
same-model judge. Gemini thought parts are omitted from the report. Keys are
sent in headers, never model prompts, URLs or reports.

Maximum eight native calls, serial with 15-second pauses after completion.
Any API/format error stops that provider without retries or model substitution;
the other configured provider can still run. Partial replies and safe error
codes are preserved in dialogue-plain.txt. Ordinary AI mode, instructions,
model, DB, archive, consent and memory remain unchanged. No paid model account
or new external service is added; existing configured keys are used. No live
result for this control exists in development. Google Drive unchanged.

After installing this change, next bounded live command: /dialogue_lab plain.
Do not repeat the full or minimal JSON controls unchanged.

Primary protocol references consulted:
- https://console.groq.com/docs/api-reference
- https://console.groq.com/docs/reasoning
- https://ai.google.dev/api/generate-content
