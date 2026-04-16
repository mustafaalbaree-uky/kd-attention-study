# EE599 Project Rubric
**Deep learning course · 599 section · 2-person team**

## Grading Scale
| Symbol | Label |
|--------|-------|
| D | Failing |
| C | Acceptable |
| B | Solid |
| A | Strong |
| S | Exceptional |

---

## 1. Topic Relevance & Scope — *Foundation*

| Grade | Description |
|-------|-------------|
| D | Topic unrelated to course content, or is a thesis/advisor project submitted without modification. |
| C | Loosely related to the course but lacks a clear connection to covered architectures or methods. |
| B | Clearly within course scope (CNNs, ViTs, SSL, KD, VLMs, XAI, etc.) and well-defined. |
| A | Tightly scoped, well-motivated, and directly extends or questions something from course material. |
| S | Original and technically sharp — a novel combination of course methods or creative extension of a recent paper. |

---

## 2. Technical Execution — *Core*

| Grade | Description |
|-------|-------------|
| D | Code does not run, model trained from scratch without justification, or implementation is copied wholesale. |
| C | Code runs but contains significant errors or model/dataset choice is poorly motivated. |
| B | Uses pre-trained models (Hugging Face, timm, etc.) appropriately. Code is functional, dataset is public. |
| A | Implementation is clean and deliberate. Config files present. Architectural choices are explained in the report. |
| S | Fully reproducible end-to-end. A new reader could replicate results exactly. Thoughtful ablations verify each decision. |

---

## 3. Depth of Analysis — *Highest Impact*

| Grade | Description |
|-------|-------------|
| D | No analysis beyond presenting a single accuracy number. Results are not discussed. |
| C | Some comparison made (e.g., two models) but discussion is shallow — no explanation of why one outperforms another. |
| B | Results discussed with plausible explanations. At least one failure case or limitation is identified. |
| A | Project critically examines why the model works, when it fails, and what trade-offs exist. Ablations support claims. |
| S | Analysis is the centerpiece. Surfaces non-obvious insights — edge cases, failure modes, architectural trade-offs — backed by rigorous experiments. |

---

## 4. Report Quality — *Deliverable*

| Grade | Description |
|-------|-------------|
| D | Report missing, has no structure, or reads like a README rather than a research paper. |
| C | Has most required sections but motivation is weak, experiments section is thin, or conclusion is missing. |
| B | Follows required structure: intro/motivation, method & analysis, experiments with discussion, conclusion/future work. |
| A | Reads like a polished conference paper. Each section flows logically. Figures and tables are clear and referenced. |
| S | Submission-quality. Related work is meaningfully engaged with, not just listed. Contributions are stated precisely and delivered on. |

---

## 5. Reproducibility — *Hard Requirement*

> ⚠️ Missing code, inaccessible datasets, or missing config files will **cap your grade regardless of analysis quality**.

| Grade | Description |
|-------|-------------|
| D | No code submitted, dataset is private or inaccessible, or results cannot be reproduced. |
| C | Code submitted but missing config files, resource files, or key dependencies. Dataset setup is unclear. |
| B | Full code, config files, and dataset pointers provided. A motivated reader could reproduce results with effort. |
| A | README explains setup clearly. All dependencies documented. Results match the report. |
| S | Reproduction is trivial — scripted setup, pinned dependencies, results are deterministic (seed set). |

---

## 6. Presentation — *Weeks 15–16*

| Grade | Description |
|-------|-------------|
| D | Presenter cannot answer basic questions about their own work. Slides absent or incoherent. |
| C | Covers main points but is disorganized or heavily read from slides. Q&A is uncertain. |
| B | Clear and covers all major sections. Speaker can answer questions with reasonable depth. |
| A | Engaging and well-paced. Narrative arc is clear: problem → method → results → insight. Questions handled confidently. |
| S | Feels like a real conference talk. The contribution lands with the audience. Difficult questions are welcomed and addressed. |

---

## 7. Ambition vs. Completeness — *Tie-Breaker*

| Grade | Description |
|-------|-------------|
| D | Ambitious idea attempted but nothing works. No partial results or meaningful analysis. |
| C | Project is simple but incomplete — key experiments missing or results are placeholder. |
| B | A simple, minimal version of the idea is fully implemented and analyzed. The project is self-contained. |
| A | Core idea is complete and well-analyzed, with one or two meaningful extensions beyond the minimum. |
| S | Both ambitious and complete. Extensions are well-motivated and add genuine insight, not just extra experiments. |

---

## Key Principles (Professor's Notes)

1. **Analysis is the differentiator.** Professor's exact words: *"The highest grades go to projects that do not just show a working model, but critically analyze why it works, when it fails, and what the trade-offs are."* Everything else is table stakes.

2. **Reproducibility is a hard requirement, not a bonus.** Missing code, inaccessible datasets, or missing config files will cap your grade regardless of analysis quality.

3. **Complete beats ambitious.** A fully analyzed simple project scores higher than an unfinished complex one. Start with the minimal version and extend only once the core is solid.

4. **599 section / 2-person team scope note.** You are not held to the novelty bar of a 699 student, but two people are expected to cover more ground than a solo project.
