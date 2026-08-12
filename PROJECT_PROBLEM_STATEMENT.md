# Rupture Prediction from Psychotherapy Sessions

## Problem statement

The goal of this project is to predict the presence of a therapeutic rupture
within one-minute segments of psychotherapy sessions.

The source data consists of video recordings of patients and therapists across
multiple sessions. The dataset includes multiple patients and therapists, and a
patient or therapist may appear in more than one recording. Some segments have
limited or no visible face information because of occlusion, camera position,
tracking failure, or poor video quality. Audio quality also varies, and some
segments may have missing, noisy, or unusable speech.

Rupture prediction is currently formulated as a binary classification task for
each one-minute segment:

- `1`: rupture is present according to the selected annotation rule.
- `0`: rupture is not present according to the selected annotation rule.

The exact conversion from rupture ratings to binary labels, including the
rating columns and threshold, must be recorded with every experiment. Existing
binary experiments use a threshold of rating greater than `1`.

## Available modalities

The intended model is multimodal and may use:

- **Video:** frame-level facial representations of the patient, including
  facial embeddings, action units, landmarks, head pose, gaze, expression
  features, and face-detection or quality indicators.
- **Audio:** acoustic and prosodic features such as pitch, energy, speaking
  rate, pauses, voice quality, turn-taking, and other temporal speech features.
- **Transcript:** text produced from the session audio, represented with
  utterance-level or contextual language embeddings.

Each modality should preserve temporal order within the one-minute segment.
Modality-presence masks and quality indicators should be retained so the model
can distinguish a genuine zero-valued feature from missing or unreliable input.

## Proposed modeling approach

Frame-level video features, time-aligned audio features, and transcript
representations will be encoded and fused into a temporal sequence. Candidate
sequence models include:

- Temporal Convolutional Network (TCN)
- Gated Recurrent Unit (GRU)
- Transformer

The sequence model will output a rupture probability for each one-minute
segment. Modality-specific encoders and late or intermediate fusion may be
compared after unimodal baselines are established.

## Current focus

The current focus is the video modality, specifically the construction and
evaluation of useful frame-level facial representations for the patient.

Work in this stage includes:

- detecting and tracking the patient's face;
- extracting per-frame facial features or embeddings;
- retaining frame order and timestamps;
- representing face visibility, detection confidence, and video quality;
- defining behavior for frames or segments with no detected face; and
- testing whether these representations contain generalizable rupture signal.

OpenFace and Py-Feat have already been used to extract facial features.
TCN and GRU sequence models have been trained on these temporal features.
Initial experiments show rapid overfitting and weak out-of-group
generalization, so simple baselines and strictly grouped evaluation remain
essential.

## Data-quality handling

Poor-quality or missing observations must not be silently treated as valid
facial measurements.

- Preserve face-detection success and quality fields for every frame.
- Use an explicit observation mask when facial features are missing.
- Apply imputation only after the train split is defined, using statistics
  calculated from training data.
- Report the number of missing frames, low-quality frames, and fully missing
  segments for every split.
- Evaluate whether missingness or quality alone predicts the label, since this
  could create a dataset artifact rather than a clinically meaningful signal.
- Compare results with and without low-quality or no-face segments.

Bad audio and unreliable transcripts should eventually be handled in the same
way, with modality-availability and quality masks rather than silent removal or
zero filling.

## Evaluation requirements

Splits must be group-wise so the same patient does not appear in both training
and evaluation data. Therapist-level and session-level leakage should also be
checked. Depending on the research question, evaluation should use patient,
therapist, patient-therapist dyad, or session as the grouping unit.

Every experiment should report:

- the binary-label definition and threshold;
- train, validation, and test segment counts;
- unique patient, therapist, session, and video counts per split;
- class prevalence per split;
- face/audio/transcript availability and quality per split;
- balanced accuracy, precision, recall, specificity, F1, AUROC, and average
  precision;
- confusion matrices and predicted-probability distributions;
- results from train-prior, pooled-summary ridge or elastic-net, and other
  appropriate simple baselines; and
- variation across repeated group splits or leave-one-group-out evaluation.

When comparing datasets or feature extractors, use matched group counts,
matched split sizes, the same label rule, and the same evaluation protocol.
Model selection must use validation data only; the test set is reserved for the
final evaluation.

## Primary research questions

1. Do frame-level facial representations contain rupture-related signal that
   generalizes to unseen patients?
2. Which video representation is most useful: handcrafted facial features,
   learned facial embeddings, or a combination?
3. How much performance is explained by missingness, recording quality,
   patient identity, therapist identity, or session-level artifacts?
4. Does temporal modeling improve over pooled one-minute summary features?
5. Do audio and transcript modalities add generalizable signal beyond video?
6. Which multimodal fusion strategy remains robust when one or more modalities
   are missing or poor quality?

## Context for future prompts

Use the following as the default project context unless a prompt explicitly
overrides it:

> We are developing a binary classifier for therapeutic rupture in one-minute
> psychotherapy-session segments. The recordings contain multiple patients,
> therapists, sessions, and videos. The same person may occur in multiple
> recordings, so all validation and test evaluation must use leakage-resistant
> group-wise splits, normally by patient and, when appropriate, by therapist,
> dyad, or session. The final system is multimodal: patient facial features or
> embeddings from video, acoustic and prosodic audio features, and transcript
> embeddings. The current focus is frame-level patient video representation.
> OpenFace and Py-Feat features have been tested with TCN and GRU models.
> Existing results show early overfitting and weak out-of-group performance.
> Missing faces, poor video, bad audio, and unreliable transcripts must be
> represented explicitly with quality and availability masks. Experiments must
> compare against simple baselines, report class prevalence and group counts,
> and use repeated or leave-one-group-out evaluation where feasible. Dataset
> comparisons must match patient counts, segment counts, label definitions,
> splits, and metrics.

