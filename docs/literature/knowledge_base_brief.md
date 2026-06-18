# Knowledge base brief for STED fiber segmentation papers

**Document role:** scientific scope and synthesis brief
**Status:** working specification
**Companion specification:** `knowledge_base_extraction_schema.md`
**Schema version:** `1.2.0`

## 1. Goal

Build an auditable knowledge base from an eight-paper core corpus on filament segmentation, topology preservation, structural prediction, crossing resolution, tracing, annotation, and evaluation.

The knowledge base will support later decisions about a U-Net-like system for segmenting, extracting, and skeletonizing fibers in STED microscopy images. It must make it possible to distinguish:

1. facts and results explicitly reported by a paper;
2. synthesis statements made by a review;
3. project-specific interpretations or transfer judgments;
4. information that was not reported or could not be verified.

The knowledge base is evidence for model design, not a predetermined model specification. It must preserve contradictions, limitations, and uncertainty.

## 2. Scope and authority

The core corpus is limited to the eight papers listed in `Papers/manifest.yaml`, including `Togo26`.

The manifest is the authoritative source for:

- stable cite keys;
- exact PDF filenames;
- canonical titles;
- DOI or other persistent identifier;
- publication year;
- document type;
- source checksum.

Cite keys are stable identifiers only. They must not be interpreted as authoritative publication years. In particular, legacy keys such as `Shi20b` or `Tet18` may differ from the final publication year; the verified year in the manifest must be used.

When sources disagree, use the following priority:

1. the paper PDF for methods, experiments, and reported results;
2. the publisher or DOI metadata for bibliographic metadata;
3. the review for review-level synthesis;
4. agent inference, clearly labelled as inference.

Do not silently reconcile contradictory information. Record the disagreement and its sources.

## 3. Corpus roles

| Cite key | Short title | Primary role in this knowledge base |
|:---|:---|:---|
| `Liu18` | Densely Connected Stacked U-network for Filament Segmentation in Microscopy Images | Microscopy-specific segmentation baseline; annotation workflow; training recipe |
| `Shi20b` | clDice | Topology-preserving objective for thin connected structures |
| `Tet18` | DeepVesselNet | Multitask prediction of segmentation, centerline, and bifurcation targets |
| `Liu20` | Quantifying Actin Filaments Using Keypoint Detection Techniques and a Fast Marching Algorithm | Junction and endpoint prediction with graph-oriented postprocessing |
| `Liu19` | Intersection to Overpass | Orientation-aware treatment of crossings and instance continuity |
| `Xu15` | SOAX | Classical tracing method; possible annotation, weak-label, and baseline tool |
| `Ozd21` | Automated and semi-automated enhancement, segmentation and tracing of cytoskeletal networks in microscopic images: A review | Field taxonomy, recurring bottlenecks, annotation burden, and evaluation framing |
| `Togo26` | ToFiE | Topology-aware 3D reconstruction, parameter sensitivity, and graph recovery from dense microscopy data |

These descriptions define extraction priorities, not conclusions that the papers are required to support.

## 4. Required artifacts

Write all generated artifacts under `knowledge_base/`.

| Artifact | Purpose | Minimum content |
|:---|:---|:---|
| `paper_cards.jsonl` | Structured source records | Exactly one record per manifest entry, with provenance and evidence links |
| `synthesis_claims.jsonl` | Structured cross-paper claims | Evidence-neutral findings, question answers, hypothesis assessments, and corpus limitations |
| `comparison_table.csv` | Cross-paper comparison | One row per paper using normalized fields derived from the JSONL |
| `synthesis.md` | Evidence-grounded synthesis | Answers to the synthesis questions, disagreements, gaps, and design implications |
| `kb_index.md` | Human-readable entry point | Corpus list, schema version, artifact map, extraction status, and validation instructions |

The paper cards are the source for paper-level content. Structured synthesis claims reference card evidence IDs. The CSV and narrative synthesis must be generated from these sources rather than maintained independently.

## 5. Information to extract

Each paper card should cover the following areas when applicable:

| Area | Content |
|:---|:---|
| Identity and provenance | Cite key, title, verified year, PDF filename, checksum, extraction status |
| Scope | Paper type, domain, imaging modality, dimensionality, task level |
| Inputs | Image or volume representation, patching, preprocessing |
| Architecture or algorithm | Family, backbone or classical method, components, downsampling, heads, supervision |
| Outputs | Mask, centerline, junctions, endpoints, bifurcations, orientation layers, traces, graphs |
| Objectives | Loss functions, weighting, class-imbalance handling |
| Training | Optimizer, learning rate, batch size, epochs, augmentation, pretraining, fine-tuning |
| Data and annotation | Real or synthetic data, splits, weak labels, manual correction, derived targets |
| Postprocessing | Skeletonization, tracing, pairing, voting, fast marching, graph reconstruction |
| Evaluation | Metrics, baselines, validation design, topology-sensitive evaluation |
| Findings | Author-stated contributions, experimental results, author-stated limitations |
| Project assessment | Relevance, transfer risks, assumptions, and possible use in a STED pipeline |
| Evidence | Page-linked evidence for every nontrivial reported fact or paper-attributed claim |

Use the missing-value token `not_reported` only after checking the relevant method, experiment, appendix, figures, and tables.

## 6. Evidence and claim rules

### 6.1 Claim classes

Every substantive statement must be assigned one of these classes:

- `paper_claim`: a claim made by the paper's authors;
- `reported_method`: a method or implementation detail explicitly described in the paper;
- `reported_result`: an experimental result explicitly reported in the paper;
- `author_stated_limitation`: a limitation explicitly acknowledged by the authors;
- `review_synthesis`: a statement synthesized by the review;
- `agent_inference`: a conclusion or transfer judgment made for this project.

A sentence must not be presented as a paper finding merely because it appears plausible from the method.

### 6.2 Evidence requirements

- Every nontrivial reported field must link to one or more evidence spans.
- Evidence should identify the source PDF, PDF page, printed page when available, section, and figure or table where relevant.
- Use a short direct quotation only when wording matters. Otherwise use a close paraphrase.
- Do not use abstracts alone for detailed architecture, training, or evaluation fields when the full paper provides more precise evidence.
- Hyperparameters, dataset sizes, splits, and numerical results must not be inferred.
- If a field is ambiguous, record the ambiguity and use a lower confidence level.
- Review-level statements must not be treated as direct experimental evidence from the papers discussed by the review.
- Transfer judgments to STED are `agent_inference` unless the source directly evaluates STED or an explicitly comparable setting.

### 6.3 Negative and conflicting evidence

Record:

- failed or weak results;
- reported sensitivity to parameters;
- methods that apply only to special cases;
- differences between 2D and 3D settings;
- conflicts between papers;
- claims that cannot be verified from the available PDF.

The synthesis must not omit evidence merely because it contradicts a working hypothesis.

## 7. Normalization principles

Controlled vocabularies exist to support comparisons, not to overwrite paper terminology.

For normalized fields:

- preserve the paper's terminology in a corresponding `raw_value` or notes field;
- use `other` when no controlled term is accurate;
- explain `other` in `raw_value`;
- use `not_reported` only for information absent from the source;
- do not choose the nearest available term when it changes the technical meaning.

The complete controlled vocabularies and escape-hatch conventions are defined in `knowledge_base_extraction_schema.md`.

## 8. Paper-specific extraction priorities

### `Liu18`

Prioritize:

- exact structure of the densely connected stacked U-Net;
- number of stages, channels, skip connections, cross-connections, and intermediate supervision;
- patch size, optimizer, learning rate, epochs, batch size, and augmentation;
- dataset composition and train, validation, and test splits;
- how SOAX-derived or other preliminary labels were corrected;
- topology- or continuity-sensitive evaluation;
- limitations of predicting only a semantic mask.

Do not assume that this is the best baseline until the evidence is compared with the rest of the corpus.

### `Shi20b`

Prioritize:

- mathematical definition of clDice and soft-clDice;
- topology assumptions and guarantees stated by the authors;
- how soft-clDice is combined with region-based losses;
- meaning and cost of iterative soft skeletonization;
- tested backbones, tasks, and dimensionalities;
- failure cases and sensitivity to structure thickness or radius variation;
- distinction between a loss contribution and a backbone contribution.

### `Tet18`

Prioritize:

- exact architecture and the rationale for its spatial-resolution choices;
- segmentation, centerline, and bifurcation outputs;
- loss design and class-imbalance handling;
- patch size and dimensionality;
- synthetic pretraining and real-data fine-tuning;
- evidence for or against transfer from vessels to microscopy fibers.

Do not assume that every bifurcation-specific design transfers directly to actin or other STED fibers.

### `Liu20`

Prioritize:

- definitions of endpoint and junction targets;
- heatmap and offset targets;
- synthetic-data generation;
- keypoint-detection architecture;
- integration of segmentation, keypoints, fast marching, and graph construction;
- the failure modes of naive skeletonization that the method addresses;
- dependence on accurate keypoint detection.

### `Liu19`

Prioritize:

- orientation-aware output decomposition;
- orientation bins or layers and their construction;
- synthetic-data procedure;
- merged reconstruction and overlap-reduction objective;
- terminus-pairing criteria;
- assumptions about crossings and filament continuity;
- situations in which the method is preferable to semantic segmentation;
- conditions under which orientation discretization may fail.

### `Xu15`

Prioritize:

- required inputs and produced outputs;
- initialization and evolution of stretching open active contours;
- junction configuration;
- user parameters and sensitivities;
- failure modes in dense, noisy, or crossing-rich networks;
- possible use for annotation assistance, weak labels, correction, tracing, or baseline evaluation.

Treat SOAX as a classical method. Do not describe it as a neural architecture.

### `Ozd21`

Prioritize:

- taxonomy of enhancement, segmentation, and tracing methods;
- recurring bottlenecks such as low signal-to-noise ratio, density, intersections, annotation burden, and skeleton artifacts;
- distinctions between supervised, unsupervised, classical, and deep-learning approaches;
- evaluation when topology and morphology matter;
- relevance to super-resolution and STED-like data;
- the review date and the resulting limits on coverage of newer methods.

Statements that summarize underlying literature must be labelled `review_synthesis`. Facts about the review's own scope or method may be labelled `reported_method`, and its authors' interpretations may be labelled `paper_claim`.

### `Togo26`

Evaluate, without assuming effectiveness:

- the full 3D reconstruction workflow and its required inputs;
- how topology and graph entities are represented;
- preprocessing, persistence, tracing, junction, and cleanup stages;
- validation on synthetic and experimental data;
- reported parameter sensitivity and failure modes;
- transfer limits between confocal collagen data and STED cytoskeletal data;
- the implications of preprint status for confidence and reproducibility.

## 9. Cross-paper synthesis questions

`synthesis.md` must explicitly answer:

1. Which architecture families are credible first baselines for STED fiber segmentation, and what evidence supports each?
2. Which training objectives improve connectivity without requiring a different backbone?
3. What evidence supports auxiliary centerline, junction, endpoint, or bifurcation supervision?
4. Which approaches are most useful when labelled STED data are scarce?
5. Which methods address crossings specifically rather than general semantic segmentation?
6. Which classical tools remain useful for annotation, weak labels, tracing, or evaluation?
7. Which conclusions are supported by direct microscopy evidence, adjacent-domain evidence, review synthesis, or project inference?
8. Which methods produce a skeleton or graph directly, and which require postprocessing?
9. Which evaluation metrics capture pixel accuracy, centerline quality, connectivity, branch structure, and instance continuity?
10. What important questions remain unanswered by this manifest-defined corpus?

Each answer must include:

- supporting and contradicting papers;
- evidence class;
- confidence;
- limitations on transfer to STED;
- a concise implication for model design.

## 10. Working hypotheses to test

The following are provisional hypotheses. First summarize the evidence without reference to these hypotheses. Then assess each hypothesis as `supported`, `partially_supported`, `contradicted`, or `insufficient_evidence`.

| ID | Working hypothesis |
|:---|:---|
| `H1` | A U-Net-like encoder-decoder is a reasonable first semantic-segmentation baseline for 2D STED fibers. |
| `H2` | Adding a topology-aware objective such as a Dice–clDice combination can improve connectivity without replacing the backbone. |
| `H3` | Auxiliary centerline or branch-related supervision can improve structural fidelity beyond mask-only training. |
| `H4` | Junction and endpoint prediction becomes useful when the final scientific output is a graph or branch statistics rather than only a mask. |
| `H5` | Orientation-aware decomposition is most valuable when crossings are a dominant and measurable failure mode. |
| `H6` | Classical tracing may be useful for annotation assistance or baselines, but its reliability depends on image quality and parameter tuning. |
| `H7` | Annotation burden and topology-sensitive evaluation are major constraints for the project. |

For each hypothesis, `synthesis.md` must record one of:

- `supported`;
- `partially_supported`;
- `contradicted`;
- `insufficient_evidence`.

## 11. Comparison table

`comparison_table.csv` must contain one row per paper and, at minimum:

| Column | Meaning |
|:---|:---|
| `cite_key` | Stable manifest identifier |
| `study_type` | Primary research or review |
| `method_paradigms` | Deep learning, classical, composite, or not applicable |
| `task_focus` | Main technical problem |
| `architecture_family` | Normalized family or `other` |
| `dimensionality` | 2D, 2.5D, 3D, mixed, or not reported |
| `main_targets` | Predicted or reconstructed structures |
| `losses` | Main training objectives |
| `data_strategy` | Real, synthetic, weak-label, or mixed strategy |
| `postprocessing` | Main downstream processing |
| `evaluation_focus` | Pixel, topology, centerline, branch, or instance evaluation |
| `direct_sted_evidence` | Whether STED was evaluated directly |
| `best_use_case` | Agent assessment, explicitly labelled |
| `main_transfer_risk` | Main limitation for use in this project |
| `extraction_status` | Draft, reviewed, or verified |

Free-text columns derived from agent assessment must not be presented as paper claims.

## 12. Quality gates

A paper card is incomplete when:

- its source PDF or checksum does not match the manifest;
- a required field is absent rather than explicitly populated or marked `not_reported`;
- a reported fact has no evidence link;
- an inference is not labelled;
- a review statement is represented as direct experimental evidence;
- normalized terminology has replaced an important raw term;
- the card contains values copied from an example rather than verified from the PDF.

The complete knowledge base is not ready for model-design decisions until:

- every manifest-listed card passes schema validation;
- the CSV is regenerated from the JSONL;
- the synthesis answers all required questions;
- contradictions and insufficient evidence are visible;
- the extraction status of each paper is shown in `kb_index.md`;
- the validator exits successfully.

## 13. Expected synthesis structure

`synthesis.md` should present the corpus as a design space, organized by how each approach seeks to preserve or recover fiber structure:

- backbone and multiscale representation;
- region and topology-aware objectives;
- auxiliary structural prediction;
- crossing-aware decomposition;
- keypoint and graph-based reconstruction;
- classical tracing and annotation support;
- evaluation and annotation constraints.

After presenting evidence-neutral findings and hypothesis assessments, the final section may translate the evidence into ranked design options, not a single mandatory architecture. Each option must state its evidence base, confidence, expected benefit, implementation cost, and main risk.

## 14. References

The bibliography below is a convenience copy. `Papers/manifest.yaml` remains authoritative for verified metadata.

- `[Liu18]` Y. Liu *et al.*, “Densely Connected Stacked U-network for Filament Segmentation in Microscopy Images,” DOI: `10.1007/978-3-030-11024-6_30`.
- `[Shi20b]` S. Shit *et al.*, “clDice — a Novel Topology-Preserving Loss Function for Tubular Structure Segmentation,” DOI: `10.1109/CVPR46437.2021.01629`.
- `[Tet18]` G. Tetteh *et al.*, “DeepVesselNet: Vessel Segmentation, Centerline Prediction, and Bifurcation Detection in 3-D Angiographic Volumes,” DOI: `10.3389/fnins.2020.592352`.
- `[Liu20]` Y. Liu *et al.*, “Quantifying Actin Filaments in Microscopic Images Using Keypoint Detection Techniques and a Fast Marching Algorithm,” DOI: `10.1109/ICIP40778.2020.9191337`.
- `[Liu19]` Y. Liu *et al.*, “Intersection to Overpass: Instance Segmentation on Filamentous Structures with an Orientation-Aware Neural Network and Terminus Pairing Algorithm,” DOI: `10.1109/CVPRW.2019.00021`.
- `[Xu15]` T. Xu *et al.*, “SOAX: A software for quantification of 3D biopolymer networks,” DOI: `10.1038/srep09081`.
- `[Ozd21]` B. Özdemir and R. Reski, “Automated and semi-automated enhancement, segmentation and tracing of cytoskeletal networks in microscopic images: A review,” DOI: `10.1016/j.csbj.2021.04.019`.
- `[Togo26]` R. Togo *et al.*, “ToFiE, a Topology-aware Fiber Extraction workflow for 3D reconstruction of dense and heterogeneous biological fiber networks from microscopy images,” arXiv: `2604.18230`.
