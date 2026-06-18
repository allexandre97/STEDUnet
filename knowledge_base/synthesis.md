# Literature synthesis

## Corpus and evidence scope

This synthesis is derived from the eight-paper core corpus (8 manifest-listed PDFs) and structured evidence spans. Paper-reported content, review synthesis, and project inference remain separately labelled. The working hypotheses are evaluated after the evidence-neutral findings.

## Extraction status

| Cite key | Status | Publication status | Direct STED evidence |
|:--|:--|:--|:--|
| `Liu18` | `reviewed` | `published` | `no` |
| `Shi20b` | `reviewed` | `published` | `no` |
| `Tet18` | `reviewed` | `published` | `no` |
| `Liu20` | `reviewed` | `published` | `no` |
| `Liu19` | `reviewed` | `published` | `no` |
| `Xu15` | `reviewed` | `published` | `no` |
| `Ozd21` | `reviewed` | `published` | `no` |
| `Togo26` | `reviewed` | `preprint` | `no` |

## Evidence-neutral findings

### F01: No primary STED evaluation in the manifest corpus

**Claim:** None of the primary-research papers directly evaluates the proposed method on STED data; the only STED-specific statements in this corpus come from the 2021 review.

**Evidence class:** `agent_inference`

**Supporting papers:** `Liu18`, `Shi20b`, `Tet18`, `Liu20`, `Liu19`, `Xu15`, `Togo26`, `Ozd21`

**Supporting evidence:** `Liu18-E001`, `Shi20b-E004`, `Tet18-E001`, `Liu20-E002`, `Liu19-E005`, `Xu15-E001`, `Togo26-E006`, `Ozd21-E005`

**Contradicting or limiting papers:** None

**Limiting evidence:** None

**Confidence:** `high`

**Implication for the STED project:** All architecture, loss, tracing, and graph-recovery choices require validation on project STED data before being treated as established.

### F02: Connectivity and filament identity are distinct objectives

**Claim:** Region connectivity, centerline continuity, graph junction recovery, and instance identity at crossings are treated as different technical problems across the corpus.

**Evidence class:** `agent_inference`

**Supporting papers:** `Shi20b`, `Tet18`, `Liu20`, `Liu19`, `Togo26`

**Supporting evidence:** `Shi20b-E002`, `Tet18-E001`, `Liu20-E004`, `Liu19-E001`, `Togo26-E004`

**Contradicting or limiting papers:** None

**Limiting evidence:** None

**Confidence:** `high`

**Implication for the STED project:** Evaluation and target design must state whether the project needs a mask, connected centerline, graph, or individual filament traces.

### F03: Labels are produced through heterogeneous processes

**Claim:** The corpus uses manual correction, classical seed labels, procedural synthetic data, derived structural labels, and partial annotation; these sources have different bias and uncertainty.

**Evidence class:** `agent_inference`

**Supporting papers:** `Liu18`, `Tet18`, `Liu20`, `Liu19`, `Ozd21`

**Supporting evidence:** `Liu18-E002`, `Tet18-E003`, `Liu20-E002`, `Liu19-E003`, `Ozd21-E006`

**Contradicting or limiting papers:** None

**Limiting evidence:** None

**Confidence:** `high`

**Implication for the STED project:** Project labels should record their origin and should not mix automatically derived and manually verified targets without provenance.

### F04: Dimensionality changes the evidence base

**Claim:** The closest microscopy semantic and instance segmentation studies are 2D, whereas direct 3D graph reconstruction evidence comes from vessels or confocal collagen.

**Evidence class:** `agent_inference`

**Supporting papers:** `Liu18`, `Liu19`, `Tet18`, `Togo26`

**Supporting evidence:** `Liu18-E004`, `Liu19-E002`, `Tet18-E001`, `Togo26-E001`

**Contradicting or limiting papers:** None

**Limiting evidence:** None

**Confidence:** `high`

**Implication for the STED project:** The project must decide and document whether its immediate task is 2D image segmentation, volumetric reconstruction, or both before comparing methods.


## Cross-paper synthesis questions

### Q01: Credible first segmentation baselines

**Claim:** The stacked U-Net study is the closest primary evidence for a 2D microscopy-filament mask baseline; FCN and 3D topology workflows are adjacent-domain or different-output alternatives rather than equivalent baselines.

**Evidence class:** `agent_inference`

**Supporting papers:** `Liu18`

**Supporting evidence:** `Liu18-E001`, `Liu18-E003`, `Liu18-E006`

**Contradicting or limiting papers:** `Tet18`, `Togo26`

**Limiting evidence:** `Tet18-E001`, `Togo26-E001`

**Confidence:** `medium`

**Implication for the STED project:** Begin later model comparison with a simple 2D encoder-decoder only if the project target is a mask; do not infer that the stacked design is optimal.

### Q02: Connectivity-aware training objectives

**Claim:** Soft-clDice is direct evidence that a connectivity-oriented term can be combined with a region loss without changing the backbone, but gains vary by dataset, loss pairing, and alpha and are not STED-specific.

**Evidence class:** `agent_inference`

**Supporting papers:** `Shi20b`

**Supporting evidence:** `Shi20b-E003`, `Shi20b-E004`, `Shi20b-E005`

**Contradicting or limiting papers:** `Shi20b`

**Limiting evidence:** `Shi20b-E006`, `Shi20b-E008`

**Confidence:** `medium`

**Implication for the STED project:** Treat Dice plus soft-clDice as an experiment to compare against a region-only loss, with dataset-specific alpha and skeleton-iteration tuning.

### Q03: Auxiliary structural supervision

**Claim:** Centerline, bifurcation, junction, and endpoint targets have supporting evidence for structural outputs, but the evidence is split between 3D angiography and a small actin quantification study.

**Evidence class:** `agent_inference`

**Supporting papers:** `Tet18`, `Liu20`

**Supporting evidence:** `Tet18-E001`, `Tet18-E007`, `Liu20-E003`, `Liu20-E004`

**Contradicting or limiting papers:** `Tet18`, `Liu20`

**Limiting evidence:** `Tet18-E008`, `Liu20-E005`

**Confidence:** `medium`

**Implication for the STED project:** Add auxiliary heads only when labels and downstream measurements justify them; compare against the mask-only baseline.

### Q04: Strategies for scarce labelled data

**Claim:** The corpus supports testing semi-automatic correction, procedural structural labels, and synthetic pretraining, while showing that synthetic transfer may reduce convergence more than final accuracy.

**Evidence class:** `agent_inference`

**Supporting papers:** `Liu18`, `Liu19`, `Liu20`, `Tet18`

**Supporting evidence:** `Liu18-E002`, `Liu19-E003`, `Liu20-E002`, `Tet18-E003`

**Contradicting or limiting papers:** `Tet18`, `Ozd21`

**Limiting evidence:** `Tet18-E006`, `Ozd21-E006`

**Confidence:** `medium`

**Implication for the STED project:** Measure label-source bias and real-STED validation separately from synthetic-data training performance.

### Q05: Methods that address crossings specifically

**Claim:** The orientation-aware Liu19 method directly targets instance identity through crossings; junction detection and topology-aware masks address related structure but do not by themselves resolve which filament continues through a crossing.

**Evidence class:** `agent_inference`

**Supporting papers:** `Liu19`

**Supporting evidence:** `Liu19-E001`, `Liu19-E002`

**Contradicting or limiting papers:** `Liu19`, `Liu20`, `Shi20b`

**Limiting evidence:** `Liu19-E007`, `Liu20-E004`, `Shi20b-E002`

**Confidence:** `high`

**Implication for the STED project:** Use a crossing-aware method only if instance-continuity errors are explicitly labelled and measured.

### Q06: Continuing roles for classical tools

**Claim:** SOAX and ToFiE remain relevant as tracing or graph baselines and possible annotation aids, but both require parameter validation and should not be treated as automatic ground truth.

**Evidence class:** `agent_inference`

**Supporting papers:** `Xu15`, `Togo26`

**Supporting evidence:** `Xu15-E001`, `Xu15-E002`, `Togo26-E004`, `Togo26-E007`

**Contradicting or limiting papers:** `Xu15`, `Togo26`

**Limiting evidence:** `Xu15-E003`, `Xu15-E005`, `Togo26-E008`, `Togo26-E009`

**Confidence:** `high`

**Implication for the STED project:** Store classical outputs as provenance-labelled candidate annotations and quantify manual correction before reuse.

### Q07: Strength of evidence by domain

**Claim:** Microscopy-primary evidence exists for confocal filament masks, actin keypoints, microtubule instances, active-contour tracing, and collagen graphs; clDice and DeepVesselNet contribute adjacent-domain evidence, while Ozd21 is secondary synthesis.

**Evidence class:** `agent_inference`

**Supporting papers:** `Liu18`, `Liu20`, `Liu19`, `Xu15`, `Togo26`, `Shi20b`, `Tet18`, `Ozd21`

**Supporting evidence:** `Liu18-E001`, `Liu20-E001`, `Liu19-E001`, `Xu15-E001`, `Togo26-E001`, `Shi20b-E004`, `Tet18-E001`, `Ozd21-E001`

**Contradicting or limiting papers:** None

**Limiting evidence:** None

**Confidence:** `high`

**Implication for the STED project:** Recommendations must state whether they rely on direct microscopy, adjacent-domain experiments, review synthesis, or project inference.

### Q08: Direct structural outputs versus postprocessing

**Claim:** SOAX and ToFiE directly produce centerline or graph structures, and Liu20 reconstructs filament statistics through keypoints and fast marching; Liu18 and clDice produce masks that still require structural postprocessing for graphs or instances.

**Evidence class:** `agent_inference`

**Supporting papers:** `Xu15`, `Togo26`, `Liu20`

**Supporting evidence:** `Xu15-E002`, `Togo26-E004`, `Liu20-E004`

**Contradicting or limiting papers:** `Liu18`, `Shi20b`

**Limiting evidence:** `Liu18-E003`, `Shi20b-E001`

**Confidence:** `high`

**Implication for the STED project:** Define the required final artifact before choosing whether graph reconstruction belongs inside or after the learned model.

### Q09: Evaluation coverage

**Claim:** The corpus collectively covers pixel overlap, skeleton overlap, topology, graph similarity, junction recall, bifurcation error, and instance matching, but no single common benchmark spans all categories.

**Evidence class:** `agent_inference`

**Supporting papers:** `Liu18`, `Shi20b`, `Tet18`, `Liu19`, `Xu15`, `Togo26`

**Supporting evidence:** `Liu18-E005`, `Shi20b-E005`, `Tet18-E007`, `Liu19-E005`, `Xu15-E003`, `Togo26-E005`

**Contradicting or limiting papers:** None

**Limiting evidence:** None

**Confidence:** `high`

**Implication for the STED project:** Use a metric suite aligned to masks, centerlines, connectivity, junctions, and instances instead of relying on Dice alone.

### Q10: Questions left unresolved

**Claim:** The corpus does not establish performance on the project's STED distribution, the best label protocol, the reliability of crossing annotations, or the tradeoff between 2D segmentation and 3D graph reconstruction.

**Evidence class:** `agent_inference`

**Supporting papers:** `Ozd21`, `Liu19`, `Togo26`

**Supporting evidence:** `Ozd21-E005`, `Ozd21-E006`, `Liu19-E007`, `Togo26-E009`

**Contradicting or limiting papers:** None

**Limiting evidence:** None

**Confidence:** `high`

**Implication for the STED project:** Resolve these questions with dataset characterization and baseline experiments before architecture selection.


## Working-hypothesis assessment

### H01: H1: U-Net-like 2D baseline

**Claim:** A U-Net-like encoder-decoder is a reasonable first 2D mask baseline, based mainly on confocal microscopy evidence rather than STED evidence.

**Evidence class:** `agent_inference`

**Supporting papers:** `Liu18`

**Supporting evidence:** `Liu18-E001`, `Liu18-E003`, `Liu18-E006`

**Contradicting or limiting papers:** `Ozd21`

**Limiting evidence:** `Ozd21-E005`

**Confidence:** `medium`

**Hypothesis status:** `partially_supported`

**Implication for the STED project:** Test a compact U-Net-like baseline without treating the DCS architecture as predetermined.

### H02: H2: Dice and clDice combination

**Claim:** Combining region overlap with soft-clDice can improve connectivity on tested tubular datasets without replacing the backbone, but improvements are not uniform across every alpha and transfer to STED remains unverified.

**Evidence class:** `agent_inference`

**Supporting papers:** `Shi20b`

**Supporting evidence:** `Shi20b-E003`, `Shi20b-E004`, `Shi20b-E005`

**Contradicting or limiting papers:** `Shi20b`

**Limiting evidence:** `Shi20b-E006`, `Shi20b-E008`

**Confidence:** `medium`

**Hypothesis status:** `partially_supported`

**Implication for the STED project:** Compare region-only and hybrid objectives under identical training and architecture settings.

### H03: H3: Auxiliary structural supervision

**Claim:** Auxiliary structural targets are plausible for improving structural fidelity, but the corpus does not isolate their effect on STED fiber segmentation.

**Evidence class:** `agent_inference`

**Supporting papers:** `Tet18`, `Liu20`

**Supporting evidence:** `Tet18-E001`, `Tet18-E007`, `Liu20-E003`

**Contradicting or limiting papers:** `Tet18`, `Liu20`

**Limiting evidence:** `Tet18-E008`, `Liu20-E005`

**Confidence:** `low`

**Hypothesis status:** `partially_supported`

**Implication for the STED project:** Add structural heads only as controlled ablations with reliable labels.

### H04: H4: Junction and endpoint prediction for graph outputs

**Claim:** Junction and endpoint prediction is useful when the downstream output is filament count, length, or a graph, but evidence is limited to a small actin study and adjacent vessel work.

**Evidence class:** `agent_inference`

**Supporting papers:** `Liu20`, `Tet18`

**Supporting evidence:** `Liu20-E003`, `Liu20-E004`, `Tet18-E007`

**Contradicting or limiting papers:** `Liu20`

**Limiting evidence:** `Liu20-E005`

**Confidence:** `medium`

**Hypothesis status:** `partially_supported`

**Implication for the STED project:** Tie keypoint supervision to explicit graph-level acceptance metrics.

### H05: H5: Orientation-aware crossing treatment

**Claim:** Orientation-aware decomposition is specifically relevant to crossing identity, but its value depends on crossing prevalence and its heuristic failure modes.

**Evidence class:** `agent_inference`

**Supporting papers:** `Liu19`

**Supporting evidence:** `Liu19-E001`, `Liu19-E002`

**Contradicting or limiting papers:** `Liu19`

**Limiting evidence:** `Liu19-E003`, `Liu19-E007`

**Confidence:** `medium`

**Hypothesis status:** `partially_supported`

**Implication for the STED project:** Measure crossing-specific errors before investing in orientation branches or pairing logic.

### H06: H6: Classical tracing for annotation and baselines

**Claim:** Classical tracing is supported as a baseline and editable candidate-label source, while reliability remains dependent on parameters, SNR, and manual correction.

**Evidence class:** `agent_inference`

**Supporting papers:** `Xu15`, `Togo26`

**Supporting evidence:** `Xu15-E001`, `Xu15-E002`, `Togo26-E004`

**Contradicting or limiting papers:** `Xu15`, `Togo26`

**Limiting evidence:** `Xu15-E003`, `Xu15-E005`, `Togo26-E008`

**Confidence:** `high`

**Hypothesis status:** `partially_supported`

**Implication for the STED project:** Evaluate correction effort and systematic label bias before using classical outputs for supervision.

### H07: H7: Annotation and topology-sensitive evaluation constraints

**Claim:** The corpus consistently identifies annotation burden and structure-sensitive evaluation as important constraints, but their magnitude for this STED dataset is not yet measured.

**Evidence class:** `agent_inference`

**Supporting papers:** `Ozd21`, `Liu18`, `Liu19`, `Togo26`

**Supporting evidence:** `Ozd21-E006`, `Liu18-E002`, `Liu19-E005`, `Togo26-E006`

**Contradicting or limiting papers:** None

**Limiting evidence:** None

**Confidence:** `high`

**Hypothesis status:** `partially_supported`

**Implication for the STED project:** Budget annotation review and define topology/instance metrics before model development.


## Contradictions, unresolved questions, and corpus limitations

### L01: Corpus has no direct primary STED benchmark

**Claim:** Review-level descriptions of STED studies cannot replace primary-paper extraction or project-specific benchmarking.

**Evidence class:** `agent_inference`

**Supporting papers:** `Ozd21`

**Supporting evidence:** `Ozd21-E005`

**Contradicting or limiting papers:** None

**Limiting evidence:** None

**Confidence:** `high`

**Implication for the STED project:** Add primary STED sources or project experiments before making STED-specific performance claims.

### L02: Tasks and domains are heterogeneous

**Claim:** Comparisons span 2D and 3D, masks and graphs, microscopy and angiography, learned and classical methods, so reported metrics are not directly rankable across papers.

**Evidence class:** `agent_inference`

**Supporting papers:** `Liu18`, `Tet18`, `Liu19`, `Togo26`

**Supporting evidence:** `Liu18-E005`, `Tet18-E001`, `Liu19-E005`, `Togo26-E005`

**Contradicting or limiting papers:** None

**Limiting evidence:** None

**Confidence:** `high`

**Implication for the STED project:** Use the knowledge base to define experiments, not to construct a cross-paper performance leaderboard.

### L03: Independent verification remains outstanding

**Claim:** All eight core-corpus cards completed the documented PDF second pass and are marked reviewed, but none has independent human verification and Togo26 additionally remains a preprint.

**Evidence class:** `agent_inference`

**Supporting papers:** `Togo26`

**Supporting evidence:** `Togo26-E001`

**Contradicting or limiting papers:** None

**Limiting evidence:** None

**Confidence:** `high`

**Implication for the STED project:** Use the reviewed knowledge base for auditable hypothesis formation, while retaining human verification before irreversible model or experimental decisions.


## Design-option boundary

These findings identify evidence and transfer risks; they do not select or implement a final model architecture.
