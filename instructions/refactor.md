Fix guide for Codex
Goal

Restore the transcript so it behaves as a human-readable dialogue between roles, while keeping full debug information available only inside collapsible debug sections.

This is not a gameplay redesign task.
This is a projection / rendering / event-contract fix.

Core diagnosis
Current root causes
Public transcript text is bound to raw engine fields
Cluer public bubble uses clue_text_raw
Guesser public bubble uses guess_text_raw
Those fields now sometimes carry structured/internal payloads rather than final visible utterances
Structured payloads can leak into gameplay state
parse_cluer_candidates() currently has a raw fallback
A malformed JSON response can become a selected clue candidate
That selected clue can then become accepted_clue
That accepted clue is then passed to Guesser
Hidden internal stages create public bubbles
Cluer transcript bubbles are keyed by (attempt_no, repair_no)
Status-only messages are renderable
So hidden retries become visible dialogue turns
Debug is not layered
Expanded debug shows raw fields too early
No compact timeline/summary first
No strong separation between summary vs raw artifacts
Required behavioral target
Public transcript

Public transcript must show only:

Cluer: final visible clue text
Judge (clue): short verdict such as Approved. / Rejected clue.
Guesser: final visible guess text
Judge (guess): short verdict such as Correct guess confirmed. / Incorrect guess confirmed.
Debug

Debug must show:

stage history of that role action
intermediate processing steps
hidden retries
candidates / shortlist
validation / canonicalization details
raw payloads

But only inside an expandable debug area.

Absolute rule

No raw JSON or structured payload may ever appear in the main body of a bubble.

Required architecture changes
1. Introduce a strict public/debug split in transcript view models

Add or revise transcript view models so every bubble has:

public_text
status_label
debug_timeline
debug_sections
raw_artifacts
is_public_turn

Recommended model shape:

@dataclass
class TranscriptBubbleView:
    role: str
    label: str
    tone: str
    alignment: str
    message_id: str
    public_text: str = ""
    status_label: str | None = None
    is_public_turn: bool = True
    debug_timeline: list[str] = field(default_factory=list)
    debug_sections: list[BubbleDebugSection] = field(default_factory=list)
    raw_artifacts: list[BubbleRawArtifact] = field(default_factory=list)

Public rendering must use only:

label
status_label
public_text

Never render raw event payloads directly.

2. Stop using clue_text_raw / guess_text_raw as public transcript text

Create dedicated fields for visible utterances.

Engine / event contract change

When the engine has selected the final visible clue, emit a dedicated field:

visible_clue_text

When the engine has selected the final visible guess, emit:

visible_guess_text

Use these fields for transcript rendering.

Important

Keep:

clue_text_raw
guess_text_raw

for debug/logging only.

Do not use them for bubble body rendering anymore.

3. Prevent structured payloads from becoming visible utterances at protocol level

This is not only a UI problem.

Cluer side

parse_cluer_candidates() currently allows a raw fallback. That must no longer be allowed to become a visible clue.

New rule:

if structured cluer output cannot be parsed cleanly into candidates,
mark it as parse failure,
log raw payload in debug,
request hidden retry,
do not promote raw text into a selected visible clue.
Guesser side

Apply the same rule:

if shortlist payload cannot be parsed cleanly,
hidden retry,
raw payload only in debug,
no raw JSON may become visible guess text.
Add a defensive guard

Add a helper such as:

def looks_like_structured_payload(text: str) -> bool:
    ...

Detect obvious structured outputs like:

starts with { or [
contains top-level "candidates" / "guesses"
looks like serialized JSON object/array

If a would-be visible clue or guess still looks like structured payload:

treat as invalid visible utterance
send to hidden retry
never show it publicly
never pass it downstream as the role’s visible message
4. One public bubble per visible role turn, not per hidden repair cycle

Current cluer bubbles are keyed too granularly.

Required transcript keying

Public transcript keys should be:

Cluer bubble: keyed by visible attempt_no
Judge clue bubble: keyed by visible attempt_no
Guesser bubble: keyed by visible attempt_no
Judge guess bubble: keyed by visible attempt_no
Hidden internal retries

Hidden repairs / hidden shortlist retries must not create separate public bubbles.

Instead:

append their statuses to the existing bubble’s debug timeline
append retry reasons to debug sections
keep the bubble itself as one visible role turn
Result

For one visible attempt, transcript should look like:

Cluer bubble
Judge bubble
Guesser bubble
Judge bubble

Not:

Cluer attempt
Cluer repair
Cluer repair
Judge retry
Judge retry
etc.
5. Change rendering rule: status-only hidden messages should not become standalone dialogue turns

Right now status-only messages are renderable.

Change this behavior.

A message should be rendered as a public bubble only if one of these is true:

it is a real public turn
it is the currently active visible bubble being updated in place

Hidden internal stages must update the debug/status of the current visible bubble instead of creating new visible message rows.

Recommended approach:

add is_public_turn
_should_render_message() should use that flag
internal processing messages should stay attached to an existing visible turn
6. Rework debug layout inside each bubble

Keep debug collapsed by default, but structure it better.

New debug layout order
First: compact timeline

Show a short processing history like:

planning
drafting candidates
candidate batch parsed
hard filter completed
hidden repair 1 requested
selected candidate accepted

or for guesser:

forming hypotheses
shortlist parsed
2 repeated guesses removed
visible guess selected
canonicalization complete
Second: summary

Show compact human-readable summary fields, for example:

Cluer

selected angle
candidate count
rejected by hard filter
hidden repairs count

Guesser

shortlist size
repeats removed
selected visible guess
match status

Judge

logical verdict
llm verdict
final verdict
Third: detailed sections
Candidates
Validation
Canonicalization
Retry reasons
Fourth: raw artifacts

Put raw payloads in a nested subsection at the bottom:

raw model output
parse mode
raw parsed object dump

Do not make raw artifacts the first thing a user sees after opening Debug.

7. Bubble body formatting rules
Cluer bubble
Header: Cluer - attempt N
Status badge while running:
planning
drafting candidates
hard filtering
repair 1/3
Final body:
only the accepted visible clue text
Judge clue bubble
Header: Judge
Status badge while running:
checking rules
Final body:
Approved.
Rejected clue.
Guesser bubble
Header: Guesser - attempt N
Status badge while running:
forming hypotheses
deduping
finalizing guess
Final body:
only the selected visible guess text
Judge guess bubble
Header: Judge
Status badge while running:
verifying guess
Final body:
Correct guess confirmed.
Incorrect guess confirmed.
8. Preserve debug, but keep it UI-only

Debug data must never be fed back into:

Cluer prompts
Guesser prompts
Judge prompts

Debug is derived from:

controller state
logger events
validation results
retry metadata
raw payloads

It must remain read-only UI instrumentation.

Concrete file-level tasks
src/taboo_arena/engine/cluer_controller.py
remove or neutralize raw fallback as a source of visible clue text
malformed structured output must cause hidden retry, not visible clue promotion
add/emit a dedicated final visible_clue_text
src/taboo_arena/engine/round_session.py
ensure accepted clue stored for gameplay is the actual selected visible clue text only
ensure accepted guess stored for gameplay/judging is the actual selected visible guess only
add dedicated event fields:
visible_clue_text
visible_guess_text
hidden retries remain in logs/debug only
structured payloads must never become accepted visible utterances
src/taboo_arena/app/transcript.py
stop mapping public text from clue_text_raw / guess_text_raw
map from visible_clue_text / visible_guess_text
key public bubbles by visible attempt, not repair number
hidden retries update debug trace of existing bubble
add is_public_turn
rewrite _should_render_message() accordingly
src/taboo_arena/app/ui_transcript_panel.py
keep bubble body human-readable only
status separate from body
debug collapsed
debug shows timeline first, raw artifacts last
no raw JSON in main bubble body
new helper module recommended

Create something like:

src/taboo_arena/app/transcript_projection.py

This module should:

convert engine/logger events into public transcript view models
keep projection logic out of renderer
centralize public/debug separation
Tests to add
Structured cluer payload never appears in public bubble body
if candidate JSON is returned, public bubble must show only selected clue text
Structured guesser payload never appears in public bubble body
if shortlist JSON is returned, public bubble must show only selected visible guess
Malformed structured output triggers hidden retry
not public transcript leakage
Cluer hidden repairs do not create extra public bubbles
one public cluer bubble per visible attempt
Judge hidden/internal stages do not create extra public bubbles
one visible judge clue bubble per visible attempt
one visible judge guess bubble per visible attempt
Expanded debug still contains raw payloads
raw artifacts preserved
public transcript stays clean
Accepted clue passed to guesser is visible clue text only
not JSON
not raw fallback blob