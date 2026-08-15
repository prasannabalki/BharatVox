# BharatVox — Methods

## Building a Sovereign Multilingual Speech Foundation Model From Scratch

This document describes the methodology for designing, training, evaluating, and progressively scaling **BharatVox**, an open-source, healthcare-focused multilingual speech foundation model for Indian languages.

The purpose of `METHODS.md` is to make BharatVox development **transparent, reproducible, auditable, and open to independent verification**.

> **Methodological principle:** From scratch → Open and appropriately licensed data → Reproducible preprocessing → Reproducible training → Transparent evaluation → Progressive scaling.

---

## 1. Scope

The initial BharatVox research program focuses on:

- Hindi
- Tamil
- Telugu
- Malayalam
- Kannada
- Healthcare-oriented speech
- Indian language–English code-switching
- Real-time and streaming speech recognition
- Progressive expansion to additional Indian languages, dialects, and accents

The first milestone, **BharatVox-Tiny v0.1**, targets approximately **5–10 million parameters**. Its purpose is to validate the complete from-scratch training pipeline rather than claim production-ready clinical performance.

---

## 2. Definition of "From Scratch"

BharatVox models are initialized with **randomly initialized model weights** and trained directly on selected speech datasets.

The core BharatVox model will not initialize its weights from pretrained speech models such as Whisper, wav2vec 2.0, HuBERT, WavLM, SeamlessM4T, IndicConformer, or similar speech foundation models.

```text
Randomly Initialized Weights
          ↓
Indian Speech Data
          ↓
BharatVox Training Pipeline
          ↓
BharatVox Model
```

Training from scratch does **not** mean reimplementing every software component. BharatVox may use open-source frameworks, published architectures, algorithms, and tooling such as:

- Linux
- Python
- PyTorch
- torchaudio
- Hugging Face
- SentencePiece
- FFmpeg
- CUDA and compatible compute libraries
- Open and appropriately licensed datasets
- Published research methods

The important distinction is that BharatVox learns its core speech representations from its training data instead of inheriting them from another pretrained speech model.

> **Clarification — pretrained models as reference baselines:** The from-scratch principle applies to **initialization**, not to **evaluation**. Pretrained systems (e.g., IndicConformer, fine-tuned Whisper-small) will be used as *comparison lines* in the evaluation harness so BharatVox results are interpretable against the state of the art. They will never contribute weights, features, or distillation targets to the core model in v0.x.

---

## 3. Phased Development Roadmap

Development proceeds in explicitly sequenced phases. Each phase must complete before the next begins.

| Phase | Milestone | Scope | Purpose |
|---|---|---|---|
| **Phase 0** | Pipeline Proof-of-Run | Basics of all 5 languages + English: Hindi, Tamil, Telugu, Malayalam, Kannada, English (~15–25 hours each, ~100–150 hours total), ~2M params, CTC | Validate the entire pipeline end-to-end (data → preprocessing → multi-script tokenizer → training → per-language eval) with minimal compute |
| **Phase 1** | BharatVox-Tiny v0.1 | 5 languages + English code-switching, 5–10M params, dense CTC, full data | Multilingual from-scratch baseline with per-language reporting |
| **Phase 2** | v0.2 | Same model family | Data quality and multilingual sampling improvements |
| **Phase 3** | v0.3 | Healthcare terminology | Clinical evaluation sets and healthcare-oriented metrics |
| **Phase 4** | v0.4 | Streaming | Low-latency incremental decoding prototype |
| **Phase 5** | BharatVox-Mini / Small / MoE | 20M–300M+ | Scaling and specialization research |

**Rationale for Phase 0:** the highest-risk components in a from-scratch project are the data pipeline, multi-script text handling, and training loop — not the architecture. Running all six languages at small scale from day one surfaces script, normalization, and tokenizer defects immediately, instead of discovering them after a single-language pipeline is already frozen. Small per-language data keeps the run cheap (single GPU, days not weeks).

Phase 0 language basics:

| Language | Script | Phase 0 Data Target | Primary Source |
|---|---|---|---|
| Hindi | Devanagari | ~15–25 h | Kathbath / Common Voice |
| Tamil | Tamil | ~15–25 h | Kathbath / OpenSLR 65 |
| Telugu | Telugu | ~15–25 h | Kathbath / OpenSLR 66 |
| Malayalam | Malayalam | ~15–25 h | Kathbath / OpenSLR 63 |
| Kannada | Kannada | ~15–25 h | Kathbath / OpenSLR 79 |
| English (Indian-accented preferred) | Latin | ~15–25 h | Common Voice (en-IN) / NPTEL-type open lecture data (verify license) |

English is included at Phase 0 because code-switching is a core project goal (Section 13); the Latin script and English acoustics must be in the pipeline from the start, not bolted on later.

Phase 0 exit criteria:

- [ ] End-to-end run completes without manual intervention
- [ ] Training loss converges; greedy CTC decoding produces recognizable text in **all six languages**
- [ ] WER/CER computed automatically per language on held-out splits
- [ ] All six scripts survive normalization and tokenization without mojibake, dropped characters, or Unicode errors
- [ ] Full experiment record captured per Section 18

---

## 4. Initial Research Questions

BharatVox-Tiny v0.1 will investigate:

1. Can a compact multilingual ASR model be trained from random initialization across five Indian languages?
2. What preprocessing and normalization strategy works reliably across these scripts?
3. How should multilingual data be sampled when language resources are imbalanced?
4. What tokenizer/output representation provides a useful baseline?
5. How accurately can the model preserve healthcare-related terminology?
6. Can a small from-scratch model support an initial low-latency streaming prototype?
7. What limitations emerge before scaling the architecture?
8. How large is the gap between a small from-scratch model and pretrained reference systems, per language?

---

## 5. Dataset Methodology

### 5.1 Candidate Dataset Inventory

The following open datasets are the initial candidates. **All license terms must be re-verified from the original source before training** — license status below is preliminary and marked accordingly.

| Dataset | Source | Languages (relevant) | Approx. Hours | License (verify) | Notes |
|---|---|---|---|---|---|
| IndicVoices | AI4Bharat | All 5 target languages | Large, multi-language | CC-BY-4.0 (verify) | Natural + extempore speech; strong dialect coverage |
| Kathbath | AI4Bharat | All 5 target languages | ~1,600+ total across 12 languages | CC-BY-4.0 (verify) | Read speech, crowdsourced |
| Shrutilipi | AI4Bharat | All 5 target languages | ~6,400 total across languages | CC-BY-4.0 (verify) | Mined from All India Radio; transcript alignment quality varies — apply Section 5.3 filters |
| Common Voice | Mozilla | Hindi, Tamil, Malayalam (coverage varies) | Varies by release | CC0 | Cleanest licensing; hours limited for some languages |
| Google crowdsourced (OpenSLR 63/65/66/79) | OpenSLR | Malayalam (63), Tamil (65), Telugu (66), Kannada (79) | ~5–10 per language | CC-BY-SA 4.0 (verify) | Small but clean; note ShareAlike implications for redistribution |
| MUCS 2021 | Interspeech challenge | Hindi, Tamil, Telugu | ~40–100 per language | Research terms (verify) | Confirm commercial-use compatibility before inclusion |
| IISc-MILE | IISc | Tamil, Kannada | ~350 per language | Verify | Read speech corpora |

Known constraints:

- **License heterogeneity is a real risk.** CC0 and CC-BY are compatible with Apache-2.0 release goals; CC-BY-SA and research-only terms constrain redistribution of derived checkpoints and must be tracked per dataset. The training-set composition of any released checkpoint must be documented so downstream users can assess license posture.
- **Healthcare-domain speech in these languages effectively does not exist as open data.** See Section 14.

### 5.2 Per-Dataset Documentation

Every dataset accepted into training must be documented before use:

| Attribute | Value |
|---|---|
| Dataset | — |
| Source URL | — |
| Language(s) | — |
| Audio Hours (as measured, not as advertised) | — |
| Speakers | — |
| Sampling Rate | — |
| Domain | — |
| License (verified, with date and link) | — |
| Redistribution Allowed | — |
| Commercial Use | — |
| Healthcare Content | — |
| Included Version / Snapshot Hash | — |

Dataset provenance must be maintained throughout the project.

### 5.3 Dataset Inclusion Criteria

A dataset may be included when:

- its license permits the intended use and has been verified from the primary source;
- audio and transcripts are available in a usable format;
- language labels can be reasonably verified;
- transcript quality is sufficient for the experiment;
- privacy and consent requirements can be satisfied;
- provenance can be documented.

### 5.4 Dataset Exclusion Criteria

Samples may be excluded when:

- transcripts are missing or clearly mismatched (e.g., high character-level divergence between forced alignment and transcript — relevant for mined corpora like Shrutilipi);
- audio is corrupted or unusable;
- language labels are unreliable;
- licensing is unclear or incompatible;
- duplicate recordings are identified;
- privacy or consent conditions make use inappropriate;
- samples fail defined quality thresholds.

All exclusion rules should be documented and, where practical, automated.

---

## 6. Audio Preprocessing

The initial preprocessing pipeline will follow a reproducible sequence.

```text
Original Audio
      ↓
Format Validation
      ↓
Channel Conversion
      ↓
Resampling
      ↓
Quality Checks
      ↓
Silence Handling
      ↓
Segmentation
      ↓
Transcript Validation
      ↓
Training Manifest
```

Initial parameters:

| Parameter | Initial Plan |
|---|---|
| Sample Rate | 16 kHz |
| Channels | Mono |
| Audio Encoding | PCM / compatible training format |
| Minimum Duration | 1 s (initial; validate) |
| Maximum Duration | 20 s (initial; validate) |
| Segmentation Policy | VAD-based for long-form sources; native segments otherwise |
| Silence Policy | Trim leading/trailing silence; preserve internal pauses |

Values above are experimental defaults and should not be treated as frozen until validated in Phase 0.

The preprocessing pipeline should record enough metadata to reproduce each processed sample from its source.

---

## 7. Transcript and Text Normalization

Indian multilingual ASR requires explicit normalization policies.

The methodology will document handling of:

- Unicode normalization (NFC as default; document any exceptions per script)
- whitespace
- punctuation
- numerals (native-script digits vs. Latin digits — policy must be explicit per language)
- dates and times
- units and measurements
- abbreviations
- English words inside Indian-language speech
- transliterated words
- medical terminology
- mixed-script text
- code-switched utterances

Normalization rules must avoid unnecessarily removing clinically important information.

For example, medical expressions such as **BP**, **MRI**, **CT**, drug names, measurements, and dosage values may need to be preserved rather than treated as noise.

---

## 8. Tokenization and Output Representation

BharatVox will experimentally compare suitable multilingual output representations.

Potential approaches include:

- characters
- graphemes
- byte-level representations
- BPE
- unigram/SentencePiece
- other multilingual subword approaches

**Phase 0 default:** character-level output over the combined character inventory of all six scripts (Devanagari, Tamil, Telugu, Malayalam, Kannada, Latin) plus digits and essential punctuation — no learned tokenizer, minimal moving parts, and immediate exposure of any script-handling bug.

**Phase 1 primary candidate:** a shared multilingual SentencePiece unigram vocabulary (~1k–4k units) trained on normalized transcripts across all five scripts, with per-language coverage checks so no script is under-represented in the vocabulary.

Each tokenizer experiment should document:

```text
Tokenizer:
Vocabulary Size:
Training Corpus:
Normalization:
Special Tokens:
Language Tokens:
Unknown Token Policy:
Version:
```

Tokenizer selection should be based on measured multilingual ASR performance rather than assumptions inherited from English-centric systems.

---

## 9. Architecture

### 9.1 Phase 0 (Proof-of-Run)

| Parameter | Value |
|---|---|
| Parameter Target | ~2M |
| Encoder | Small Conformer or Squeezeformer-style |
| Features | 80-dim log-Mel filterbanks |
| Subsampling | 2× Conv2D, total 4× time reduction |
| Output Vocabulary | Combined 6-script character set (~350–450 chars) |
| Objective | CTC |
| Initialization | Random |

### 9.2 BharatVox-Tiny (Phase 1)

```text
Audio
  ↓
Feature Extraction (log-Mel)
  ↓
Convolutional Subsampling
  ↓
Speech Encoder (Conformer-style blocks)
  ↓
Hidden Speech Representation
  ↓
CTC Projection
  ↓
Output Tokens
  ↓
Transcript
```

| Parameter | BharatVox-Tiny |
|---|---|
| Parameter Target | ~5–10M |
| Architecture Type | Dense |
| Encoder Layers | 8–12 (to be frozen after Phase 0 findings) |
| Hidden Dimension | 144–192 |
| Attention Heads | 4 |
| FFN Dimension | 4× hidden |
| Vocabulary Size | Per tokenizer experiments (Section 8) |
| Dropout | 0.1 (initial) |
| Initial Objective | CTC |
| Initialization | Random |

The exact configuration is frozen in a versioned configuration file once the first baseline is selected. Ranges above are starting points, not commitments.

---

## 10. Initial CTC Baseline

CTC is the first objective because it provides a relatively simple end-to-end ASR baseline while the project validates data quality, tokenization, training stability, and multilingual behavior.

```text
Speech Encoder
      ↓
Frame Representations
      ↓
CTC Projection
      ↓
Token Sequence
```

Later BharatVox versions may compare CTC against transducer (RNN-T/Pruned Transducer) or other architectures better suited to streaming and richer speech modeling. Transducer comparison is expected no earlier than Phase 4 (streaming).

---

## 11. Training Methodology

Every official experiment records its complete training configuration.

Phase 0 / Phase 1 starting template:

```text
Optimizer: AdamW
Learning Rate: peak ~1e-3 (Phase 0), tune per run
Scheduler: Warmup (~2–5k steps) + cosine or Noam decay
Batch Size: dynamic, bucketed by duration (~60–120 s audio per GPU batch)
Precision: BF16 preferred; FP16 fallback; FP32 for debugging
Gradient Accumulation: as needed to reach effective batch
Gradient Clipping: 1.0
Training Steps / Epochs: until dev WER plateaus; record exact steps
Random Seed: fixed and recorded
Checkpoint Frequency: every N steps + best-on-dev
Validation Frequency: every N steps
Early Stopping: patience on dev WER
Augmentation: SpecAugment (time/freq masking) from Phase 1; speed perturbation optional
```

The methodology will also document:

- weight initialization;
- loss computation;
- batching and length bucketing;
- mixed precision;
- checkpointing;
- augmentation;
- distributed training when introduced;
- hardware utilization.

All values above are initial settings subject to Phase 0 validation, after which the Phase 1 configuration is frozen.

---

## 12. Multilingual Sampling

Language imbalance is expected to be a major methodological challenge.

BharatVox will compare strategies such as:

- proportional sampling;
- uniform language sampling;
- temperature-based sampling (τ sweep, e.g., 0.3 / 0.5 / 0.7 / 1.0);
- capped sampling;
- dynamically balanced sampling.

**Phase 1 default:** temperature-based sampling, with τ selected on per-language dev WER.

Results are reported separately for each language so that improvements in high-resource languages do not hide degradation in lower-resource languages. A single aggregate WER is never reported without the per-language breakdown.

---

## 13. Code-Switching

Indian clinical conversations frequently mix local languages with English.

Examples may contain terms such as:

- blood pressure
- diabetes
- insulin
- MRI
- CT scan
- cholesterol
- tablet
- injection

BharatVox should preserve naturally occurring code-switching instead of automatically treating English segments as transcript noise.

The project will progressively develop dedicated code-switching evaluation sets and report code-switched recognition performance separately.

---

## 14. Healthcare Adaptation

Healthcare specialization is introduced progressively after establishing a reliable general multilingual speech baseline.

**Honest constraint:** open healthcare-domain speech data in the five target languages is effectively nonexistent. Healthcare adaptation is therefore treated as a **data-creation effort**, not only a modeling effort. Planned tracks:

1. **Terminology test sets (Phase 3, evaluation-first):** scripted recordings of clinically important utterances (medication names, dosages, symptoms, negations) recorded under documented consent, per language. Small (hundreds of utterances per language) but sufficient for the metrics in Section 15.2.
2. **Synthetic text augmentation:** healthcare-domain text (drug names, dosage patterns, clinical phrasing) injected into tokenizer/LM-side resources where applicable. Text-only; does not compromise the from-scratch acoustic principle.
3. **Partnered collection (later phases):** consented, ethics-reviewed recordings via clinical or academic partners, with privacy and consent governance defined before any collection begins.

Potential terminology categories:

- diseases
- symptoms
- medications
- dosage expressions
- anatomy
- laboratory tests
- procedures
- measurements
- clinical abbreviations

Healthcare evaluation must consider the **importance of the error**, not only the number of words transcribed incorrectly.

For example, confusing **15 mg** with **50 mg** may produce a small change in conventional WER while representing a clinically significant transcription error.

BharatVox will therefore develop healthcare-oriented evaluation metrics alongside general ASR metrics.

---

## 15. Evaluation Methodology

### 15.1 Standard ASR Metrics

- **WER — Word Error Rate**
- **CER — Character Error Rate** (primary for agglutinative/morphologically rich targets where WER is harsh)

Results are reported separately for Hindi, Tamil, Telugu, Malayalam, and Kannada.

### 15.2 Reference Baselines (Comparison Lines, Not Initialization)

Every official evaluation table includes at least one pretrained reference system evaluated on the identical test split with identical text normalization:

| Reference System | Role |
|---|---|
| IndicConformer (or current AI4Bharat SOTA equivalent) | Indian-language pretrained reference |
| Whisper-small (zero-shot and/or fine-tuned) | Multilingual general-purpose reference |

Purpose: BharatVox WER numbers are uninterpretable in isolation. The reference line quantifies the from-scratch gap per language and per phase, and makes progress claims falsifiable. Reference systems contribute **no weights, features, or targets** to BharatVox training.

### 15.3 Healthcare-Oriented Metrics

Future evaluation will investigate:

- Medical Term Error Rate
- Medication Recognition Accuracy
- Dosage Preservation Accuracy
- Symptom Recognition Accuracy
- Clinical Entity Recall / Accuracy
- Negation Preservation
- Measurement Preservation
- Code-Switching Accuracy
- Language Identification Accuracy

### 15.4 Robustness Metrics

Evaluation should progressively cover:

- background noise;
- microphones and recording devices;
- regional accents;
- dialect variation;
- speaking rate;
- age-related voice variation where ethically and legally appropriate.

### 15.5 North-Star Question

> **Can BharatVox accurately preserve medically important information when an Indian patient speaks naturally in their preferred language?**

---

## 16. Streaming Methodology

Streaming is a long-term core capability.

A conceptual streaming pipeline is:

```text
Patient / Doctor
       ↓
Microphone
       ↓
Audio Chunks
       ↓
BharatVox Streaming Encoder
       ↓
Incremental Decoding
       ↓
Live Transcript
```

Streaming experiments should document:

```text
Chunk Size:
Context Window:
Look-Ahead:
First Output Latency:
Average Latency:
Real-Time Factor:
Peak Memory:
GPU Utilization:
CPU Utilization:
```

The first streaming prototype is intended as a research baseline, not a production clinical system.

---

## 17. Mixture-of-Experts Methodology

MoE is **not planned for BharatVox-Tiny**.

It will be investigated after dense multilingual models establish a reliable baseline.

Conceptually:

```text
             Speech Encoder
                   ↓
         Shared Representation
                   ↓
                Router
          ↙        ↓        ↘
      Expert 1  Expert 2  Expert 3 ...
          ↘        ↓        ↙
             Shared Output
                   ↓
                Decoder
```

Research questions may include:

- Do experts naturally specialize by language?
- Do experts specialize by language family?
- Can experts specialize around phonetic patterns?
- Does medical vocabulary create domain specialization?
- How does code-switching affect routing?
- How should low-resource languages be protected from expert imbalance?
- Can sparse activation reduce compute while increasing total model capacity?

BharatVox will prefer learned specialization over hard-coding one expert per language unless experiments demonstrate a clear advantage.

---

## 18. Reproducibility

Every important experiment captures:

```text
Experiment ID
Git Commit
Dataset Version
Dataset Manifest / Hash
Model Configuration
Tokenizer Version
Random Seed
Python Version
PyTorch Version
CUDA Version
GPU Model
Number of GPUs
Training Duration
Training Steps
Peak VRAM
Checkpoint
Evaluation Results (incl. reference baseline results)
Notes
```

Suggested experiment structure:

```text
experiments/
├── bv-p0-001/            # Phase 0 proof-of-run
│   ├── config.yaml
│   ├── data_manifest.json
│   ├── metrics.json
│   └── notes.md
├── bv-tiny-001/
├── bv-tiny-002/
└── bv-tiny-003/
```

Official results are linked to the exact configuration and data manifest used to produce them.

---

## 19. Negative Results and Failed Experiments

BharatVox aims to document meaningful negative results where practical.

Examples include:

- tokenizers that underperform;
- sampling strategies that harm low-resource languages;
- augmentations that reduce accuracy;
- architectures that fail to converge;
- streaming configurations with unacceptable latency;
- healthcare adaptations that cause catastrophic forgetting.

Publishing useful failures can prevent duplicated effort and improve the value of the project as an open research initiative.

---

## 20. Data, Code, and Model Licensing

BharatVox-developed code and eligible artifacts are intended to be released under the **Apache License 2.0**.

However:

> **Code license, dataset license, and model-weight distribution rights are separate considerations.**

Third-party datasets and resources retain their original licenses.

Additional operating rules:

- Each released checkpoint documents its exact training-set composition and the license of every contributing dataset.
- Datasets under ShareAlike or research-only terms are tracked separately; checkpoints trained on them are release-gated until license compatibility is confirmed.
- License verification (source link + date) is a precondition for training, not a post-hoc audit.

BharatVox will only redistribute datasets or model checkpoints when the relevant licenses, privacy requirements, consent conditions, and applicable laws permit redistribution.

Dataset provenance and licensing are treated as part of the technical methodology rather than as an administrative afterthought.

---

## 21. Healthcare Safety

BharatVox is a research and engineering project.

Clinical speech recognition errors involving medications, dosages, allergies, diagnoses, symptoms, measurements, negation, or procedures may have serious consequences.

Accordingly:

- general WER alone is insufficient for healthcare evaluation;
- clinically important information receives dedicated evaluation;
- early BharatVox releases are research models;
- models should not independently make diagnosis or treatment decisions;
- production clinical deployment requires additional validation, governance, privacy, security, and regulatory assessment.

---

## 22. Versioned Methodology

The methodology will evolve with the project while preserving the history of important decisions.

| Version / Stage | Methodological Focus |
|---|---|
| **Phase 0 Proof-of-Run** | Hindi-only ~2M pipeline validation |
| **BharatVox-Tiny v0.1** | ~5–10M dense, from-scratch multilingual ASR baseline |
| **v0.2** | Data and multilingual sampling improvements |
| **v0.3** | Healthcare terminology and clinical evaluation |
| **v0.4** | Streaming experiments |
| **BharatVox-Mini** | ~20–50M scaling experiments |
| **BharatVox-Small** | ~100–300M multi-task speech foundation modeling |
| **BharatVox-MoE** | Sparse expert routing and specialization research |
| **Future** | Additional Indian languages, dialects, accents and deployment research |

Major methodological changes are documented rather than silently replacing previous approaches.

---

## 23. First-Version Method Checklist

Before BharatVox-Tiny v0.1 is considered reproducible, the project should have:

- [ ] Phase 0 six-language proof-of-run completed and archived
- [ ] Dataset inventory and per-source license verification (link + date)
- [ ] Data provenance manifests
- [ ] Audio preprocessing specification
- [ ] Transcript normalization specification
- [ ] Tokenizer/output-unit specification
- [ ] Frozen BharatVox-Tiny architecture
- [ ] Random initialization documented
- [ ] Reproducible training configuration
- [ ] Per-language train/dev/test splits
- [ ] WER and CER evaluation per language
- [ ] Reference baseline systems evaluated on identical splits
- [ ] Initial healthcare terminology test set
- [ ] Code-switching evaluation samples
- [ ] Initial streaming experiment
- [ ] Hardware and software environment recorded
- [ ] Experiment logs and checkpoints versioned
- [ ] Limitations documented
- [ ] Eligible artifacts prepared for open release

---

## 24. Methodology Principle

BharatVox should not be judged solely by model size.

The project should prioritize:

**Data quality → Reproducibility → Multilingual fairness → Clinical information preservation → Efficient architecture → Responsible scaling**

The objective is to establish a transparent technical path from a few-million-parameter ASR baseline toward a sovereign multilingual speech foundation model designed around India's languages and healthcare needs.
