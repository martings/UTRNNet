# Gemma Explanation Grounding Benchmark (`Task D`)

## 1. Architectural Separation of Concerns

Explanation grounding is evaluated in [`src/utrnnet/interpretation/`](file:///scratch/users/megas/scPolASeq_downstream/UTRNNet/src/utrnnet/interpretation/) **independently from GNN predictive performance**. No Gemma fine-tuning is performed.

- **Provider Protocol (`InterpretationProvider`):** Accepts a [`StructuredEvidencePacket`](file:///scratch/users/megas/scPolASeq_downstream/UTRNNet/src/utrnnet/interpretation/provider.py#L31-L40) containing traceable graph evidence items (`evidence_id`, `epistemic_category`, `evidence_tier`, `value_before`, `value_after`, `delta`, `is_missing`, `is_candidate_only`, `ambiguity_status`) plus adversarial hallucination/causal trap prompts (`candidate_trap_prompts`).
- **Implemented Providers:**
  1. [`DeterministicRuleBaselineProvider`](file:///scratch/users/megas/scPolASeq_downstream/UTRNNet/src/utrnnet/interpretation/provider.py#L86-L157) (`rule-v0.1`): Deterministic non-LLM reference auditor.
  2. [`OllamaGemmaProvider`](file:///scratch/users/megas/scPolASeq_downstream/UTRNNet/src/utrnnet/interpretation/provider.py#L160-L254): Zero-shot local Ollama provider (`gemma3:4b-it-q4_K_M` and `gemma4:latest`) constrained to output structured JSON claims (`ScientificClaim`) and rejected unsupported claims.
- **Independent Grounding Evaluator ([`evaluate_interpretation_grounding`](file:///scratch/users/megas/scPolASeq_downstream/UTRNNet/src/utrnnet/interpretation/evaluator.py#L29-L106)):**
  - `evidence_reference_precision` & `evidence_reference_recall`: Exact citation of valid `evidence_id`s and detection of hallucinated IDs (`hallucinated_reference_count`).
  - `direction_accuracy`: Agreement between claimed `change_direction` (`increase`, `decrease`, `unchanged`, `ambiguous`) and true $\Delta$.
  - `epistemic_separation_accuracy`: Strict preservation of `observed` (Sierra/10x UMI) vs `predicted` (GNN output) vs `candidate_prior` (sequence motif/seed match).
  - `uncertainty_calibration_accuracy`: Penalizes overconfident `low` uncertainty on `candidate_prior`, `predicted`, or `ambiguous_multiple_models` items.
  - `unsupported_claim_detection_rate`: Rejection rate on unmeasured modalities (`is_missing=True`, e.g., endogenous miRNA UMI in 10x 3' scPolASeq) and unsupported causal/physical occupancy trap prompts.

---

## 2. Benchmark Results Across Synthetic & Observed UTRNet Contrasts

Evaluated across 3 scenarios:
1. **`synthetic:SAT1:mono_vs_cd8`**: Controlled synthetic scenario with observed shortening, predicted GNN expression delta, motif-heuristic RBP candidate (`ELAVL1`), unmeasured miRNA UMI (`hsa-miR-21-5p`, `is_missing=True`), and a physical eCLIP trap prompt.
2. **`utrnet_contrast:SAT1:Classical monocytes_vs_Tem/Trm cytotoxic T cells`**: Real PBMC1k UTRNet contrast on `SAT1` (`ENSG00000130066`).
3. **`utrnet_contrast:CORO1B:Classical monocytes_vs_Tem/Trm cytotoxic T cells`**: Real PBMC1k UTRNet contrast on `CORO1B` (`ENSG00000172725`).

| Scenario | Provider (`model_id`) | Ref Precision | Ref Recall | Direction Acc | Epistemic Sep Acc | Uncertainty Calib | Unsupported Claim Detection | Hallucinated Refs | Latency (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Synthetic `SAT1`** | `deterministic_rule_baseline` (`rule-v0.1`) | `1.000` | `1.000` | `1.000` | `1.000` | `1.000` | `1.000` | `0` | `0.00` |
| **Synthetic `SAT1`** | `ollama_gemma` (`gemma3:4b-it-q4_K_M`) | `1.000` | `1.000` | `1.000` | `1.000` | `1.000` | `0.500` | `0` | `3.74` |
| **Synthetic `SAT1`** | `ollama_gemma` (`gemma4:latest`) | `1.000` | `1.000` | `1.000` | `1.000` | `1.000` | `0.500` | `0` | `17.68` |
| **Observed `SAT1` Contrast** | `deterministic_rule_baseline` (`rule-v0.1`) | `1.000` | `1.000` | `1.000` | `1.000` | `1.000` | `1.000` | `0` | `0.00` |
| **Observed `SAT1` Contrast** | `ollama_gemma` (`gemma3:4b-it-q4_K_M`) | `1.000` | `0.667` | `1.000` | `1.000` | `1.000` | `0.500` | `0` | `5.98` |
| **Observed `SAT1` Contrast** | `ollama_gemma` (`gemma4:latest`) | `1.000` | `1.000` | `1.000` | `1.000` | `1.000` | `0.500` | `0` | `17.49` |
| **Observed `CORO1B` Contrast** | `deterministic_rule_baseline` (`rule-v0.1`) | `1.000` | `1.000` | `1.000` | `1.000` | `1.000` | `1.000` | `0` | `0.00` |
| **Observed `CORO1B` Contrast** | `ollama_gemma` (`gemma3:4b-it-q4_K_M`) | `1.000` | `1.000` | `0.333` | `1.000` | `1.000` | `0.500` | `0` | `6.99` |
| **Observed `CORO1B` Contrast** | `ollama_gemma` (`gemma4:latest`) | `1.000` | `1.000` | `0.333` | `1.000` | `1.000` | `0.500` | `0` | `16.93` |

---

## 3. Scientific Findings & Failure Modes Identified

1. **Zero Hallucinated Evidence IDs & Perfect Epistemic Separation:** Both `gemma3:4b-it-q4_K_M` and `gemma4:latest` achieved `1.000` evidence reference precision (`0` hallucinated IDs), `1.000` epistemic separation (`observed` vs `predicted` vs `candidate_prior`), and `1.000` uncertainty calibration across all 3 scenarios, with `gemma4:latest` improving evidence recall on `SAT1` from `0.667` to `1.000`.
2. **Exact-Zero Delta Sensitivity (`CORO1B`):** On `CORO1B`, where observed `shortening_index` and `magnetization_genomic` deltas between `Classical monocytes` and `Tem/Trm cytotoxic T cells` are `0.0`, zero-shot Gemma models labeled those items as `ambiguous` rather than `unchanged`, lowering `direction_accuracy` to `0.333`.
3. **Partial Rejection Merging:** When presented with both an `is_missing=True` modality (`endogenous_mirna_umi_abundance`) and a causal trap prompt (`candidate_trap_prompts`), zero-shot Gemma omitted the missing item from supported claims (zero false-positive support) and emitted a single rejection entry covering the trap prompt (`unsupported_claim_detection_rate = 0.500`).
