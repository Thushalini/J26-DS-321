# Labeling Guide (Codebook) — Component 1

**Version:** 0.1 (draft, Day 1)  
**Used for:** training and testing two models: the Decision Intent Classifier (Task 1) and the Context State Classifier (Task 2).  
**Raters:** two people, labeling independently.  
**Target:** Cohen's Kappa of 0.70 or higher before a label set is trusted.

> All example messages in this guide are **invented** to explain the labels. Never copy them into your dataset.

---

## Rules for every rater

1. **Label alone.** Do not talk to the other rater, and do not look at their sheet, until both of you have finished a batch.
2. **Use only what is on the sheet.** Judge the message using the previous message shown. Do not use what you know about the project or the people.
3. **One label per row.** If two labels seem to fit, use the priority rule in Section 3.
4. **If you are not sure, still pick your best label** and put `Y` in the `unsure` column. A note in `notes` helps the later discussion. Do not leave the label blank.
5. **Keep your first answer.** After the discussion you may agree on a final label, but your original sheet must stay unchanged, because Kappa is calculated from the original answers.
6. **Respect privacy.** Sheets are anonymised (Member_A, Member_B). Do not try to work out who is who, and do not share the sheets outside the project.

---

## Task 1: Decision intent (label each message)

The sheet shows the **previous message** and the **message to label**. Label the message to label.

| Label | Meaning | Quick test |
|---|---|---|
| **proposal** | Puts forward a choice for the project that the team has not settled (dataset, target, algorithm, metric, resources, plan). | "Is the speaker offering an option for the team to accept?" |
| **agreement** | Supports a choice that was already put forward, without closing the decision. | "Is the speaker saying yes to something already on the table?" |
| **finalization** | Closes the decision. It is settled and will be treated as the plan. | "Is the speaker saying this is decided?" |
| **rejection** | Says no to an option, or withdraws support. | "Is the speaker saying no to something on the table?" |
| **other** | Anything else: facts, explanations, questions that ask for information, scheduling, status updates, small talk. | "Does this message not change the state of any decision?" |

### Examples (invented)

**proposal**
- "I suggest we use XGBoost because it works well on tabular data."
- "What about predicting churn instead of revenue?" (a question that names a specific option)
- "Maybe we could try the tourism dataset." (hedged, but still a proposal)

**agreement**
- "I agree."
- "Sounds good to me."
- "Yeah, XGBoost makes sense." 

**finalization**
- "Okay, let's finalize XGBoost."
- "Decision made: RMSE is our metric."
- "We're going with the tourism dataset. I'll update the doc."

**rejection**
- "I don't think a neural network is a good idea with this little data."
- "No, let's not use accuracy."
- "Actually, scrap that."

**other**
- "Tourism data usually has seasonal patterns." (information)
- "What is XGBoost?" (asking for facts, no option offered)
- "What time is the meeting?"
- "I'll push my code tonight." (status update)

---

## 3. Hard cases and how to decide

**Word clues (a starting point, not a rule. Context decides):**
- Closing words → *finalization*: "finalize", "decided", "locked in", "we'll go with", "let's go with".
- Endorsing words → *agreement*: "I agree", "sounds good", "makes sense", "fine by me".
- Offering words → *proposal*: "I suggest", "what about", "we could", "how about".
- Refusing words → *rejection*: "no", "I don't think", "let's not", "scrap that".

**Several labels fit in one message → use this priority:**
`finalization` > `rejection` > `proposal` > `agreement` > `other`.  
Reason: pick the label that changes the state of the decision the most.  
Example: "Yes, and let's also add lag features." contains agreement and a new proposal → label **proposal**, and note it.

**Short replies depend on the previous message:**
- "ok" after a proposal → **agreement**.
- "ok" after a status update or a fact → **other**.

**Questions:**
- Names a specific option for the team ("Should we use XGBoost?") → **proposal**.
- Asks for information only ("How does XGBoost work?") → **other**.

**Conditional agreement** ("I agree if the dataset is big enough") → **agreement**, mark `unsure`.

**Agreeing and closing at once** ("Agreed, let's lock it in") → **finalization**.

**Not about the project** → **other**.

---

## Task 2: Field state (label a field, given the evidence about it)

For this task the sheet will show **all messages or document snippets about one field up to one point in time** (for example, everything said about the target variable so far). The sheet is made after the extraction step is built.

| Label | Meaning |
|---|---|
| **missing** | Nobody has stated a value for this field. |
| **known** | One value has been stated plainly, by one person or source, and nobody has responded to it yet. |
| **uncertain** | A value was stated with doubt ("maybe", "I think", "not sure"), vaguely or only partly ("some tree-based model"), or only as a question. |
| **conflicting** | Two or more different values for the same field from different people or sources, and the team has not settled it. |
| **confirmed** | A value was stated and then agreed or finalized by someone else (or by a document the team accepts), and nothing contradicts it. |

### Rules for Task 2

- **Label the state at the end of the evidence shown**, not at the start. If a conflict was settled by a later message, the answer is *confirmed*.
- **Saying it twice yourself is not confirmation.** *confirmed* needs agreement from someone else, or a finalization, or a document the team accepts.
- **Different wording for the same thing is not a conflict** ("churn" and "customer churn"). A different choice is a conflict ("churn" and "revenue").
- If you cannot tell whether two wordings mean the same thing ("gradient boosting" and "XGBoost"), pick your best label and mark `unsure`.

### Examples (invented)

| Evidence | Label |
|---|---|
| (nothing about the metric) | missing |
| "Let's predict churn." (no replies) | known |
| "Maybe churn? I'm not sure." | uncertain |
| "Let's predict churn." / "I thought revenue." | conflicting |
| "Let's predict churn." / "Agreed, churn it is." | confirmed |

---

## How a labeling round works

1. **Pilot (50 messages).** Both raters label the same 50 messages alone.
2. **Calculate Kappa** with `python -m component1.labeling.kappa sheet_rater_a.csv sheet_rater_b.csv`.
3. **Discuss every disagreement.** Look for unclear rules, not for who was right. Write the fix into this guide and raise the version number below.
4. **Repeat the pilot on 50 new messages** until Kappa is 0.70 or higher.
5. **Label the full batch** (about 300 to 400 messages) alone, again with both raters.
6. **Report Kappa** from the original, unchanged sheets. Agreed final labels (after discussion, or decided by a third reviewer) become the answer key for training and testing.
7. **Check class balance.** Count how many messages got each label. If "finalization" is rare, say so in the report.

## Data rules

- Only use chat from people who gave **their consent** to have their messages used for this project.
- Check with your supervisor whether using your own team's chat during development needs ethics approval. The proposal says the controlled student study will get Faculty Ethics Committee approval **before** that data collection starts.
- Never commit real chat, the labeled sheets, or `speaker_key.json` to git. The `.gitignore` already blocks them.

---

## Version log

| Version | Date | What changed | Kappa after the change |
|---|---|---|---|
| 0.1 | | First draft | |