# CaptionLens research progress — references and source notes

Research window: five most recent complete years (2021–2025). The survey is scoped, not systematic. Paper metadata, contributions and dataset roles were checked against the linked official publisher, proceedings, project or institutional page. Public datasets are not treated as IITH resources.

## Literature

- **2021 — CLIPScore: A Reference-free Evaluation Metric for Image Captioning** (Jack Hessel; Ari Holtzman; Maxwell Forbes; Ronan Le Bras; Yejin Choi). EMNLP 2021. Dataset(s): COCO; Flickr30K; nocaps; Conceptual Captions (metric evaluation corpora). [Official source](https://aclanthology.org/2021.emnlp-main.595/).
- **2021 — Attentive Contextual Network for Image Captioning** (Prudviraj Jeripothula; Chalavadi Vishnu; C. Krishna Mohan). IJCNN 2021. Dataset(s): MS COCO. [Official source](https://doi.org/10.1109/IJCNN52387.2021.9533970).
- **2022 — M-FFN: Multi-Scale Feature Fusion Network for Image Captioning** (Prudviraj Jeripothula; Chalavadi Vishnu; C. Krishna Mohan). Applied Intelligence 2022. Dataset(s): MS COCO. [Official source](https://people.iith.ac.in/ckm/assets/pdfs/j_4.pdf).
- **2022 — BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation** (Junnan Li; Dongxu Li; Caiming Xiong; Steven Hoi). ICML 2022. Dataset(s): COCO; Flickr30K; NoCaps (captioning evaluation/fine-tuning settings). [Official source](https://proceedings.mlr.press/v162/li22n.html).
- **2023 — BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models** (Junnan Li; Dongxu Li; Silvio Savarese; Steven Hoi). ICML 2023. Dataset(s): COCO (image captioning evaluation); multiple VLM task datasets. [Official source](https://proceedings.mlr.press/v202/li23q.html).
- **2024 — ALOHa: A New Measure for Hallucination in Captioning Models** (Suzanne Petryk; David M. Chan; Anish Kachinthaya; Haodi Zou; John Canny; Joseph E. Gonzalez; Trevor Darrell). NAACL 2024. Dataset(s): HAT (annotated MS COCO subset); NoCaps; NoCaps-FOIL. [Official source](https://aclanthology.org/2024.naacl-short.30/).
- **2024 — Mitigating Open-Vocabulary Caption Hallucinations** (Assaf Ben-Kish; Moran Yanuka; Morris Alper; Raja Giryes; Hadar Averbuch-Elor). EMNLP 2024. Dataset(s): OpenCHAIR (author-created); MS COCO and Flickr30K test examples. [Official source](https://aclanthology.org/2024.emnlp-main.1263/).
- **2024 — DOCCI: Descriptions of Connected and Contrasting Images** (Yasumasa Onoe; Sunayana Rane; Zachary Berger; Yonatan Bitton; Jaemin Cho; Roopal Garg; Alexander Ku; Zarana Parekh; Jordi Pont-Tuset; Garrett Tanzer; Su Wang; Jason Baldridge). ECCV 2024. Dataset(s): DOCCI (15k images; human descriptions; dataset contribution). [Official source](https://google.github.io/docci/).
- **2024 — DENEB: A Hallucination-Robust Automatic Evaluation Metric for Image Captioning** (Kazuki Matsuda; Yuiga Wada; Komei Sugiura). ACCV 2024. Dataset(s): Nebula (author-created); FOIL; Composite; Flickr8K-Expert/CF; PASCAL-50S. [Official source](https://openaccess.thecvf.com/content/ACCV2024/html/Matsuda_DENEB_A_Hallucination-Robust_Automatic_Evaluation_Metric_for_Image_Captioning_ACCV_2024_paper.html).
- **2025 — Panoptic Captioning: Seeking An Equivalency Bridge for Image and Text** (Kun-Yu Lin; Hongjun Wang; Weining Ren; Kai Han). NeurIPS 2025. Dataset(s): Paper-introduced panoptic caption data and human-curated test set. [Official source](https://proceedings.neurips.cc/paper_files/paper/2025/hash/7116cda41d75d580bae15d9e484a8466-Abstract-Conference.html).

## Dataset sources

- **MS COCO Captions** — Microsoft / COCO Consortium; 330K+ images overall; caption subset depends on release/split. [Official source](https://cocodataset.org/#home). Availability note: Public download; images retain source licenses; follow COCO terms.
- **Flickr8k** — University of Illinois / Flickr image collection; 8,092 images; 5 captions each (commonly reported). [Official source](https://github.com/awsaf49/flickr-dataset). Availability note: Images sourced from Flickr; check original rights and dataset terms.
- **Flickr30K** — University of Illinois / Flickr image collection; 31,783 images; 5 captions each. [Official source](https://shannon.cs.illinois.edu/DenotationGraph/). Availability note: Access terms apply; images remain subject to source rights.
- **nocaps** — Georgia Tech / Allen Institute / collaborators; Open Images images; 15,100 images; 166,100 human captions. [Official source](https://nocaps.org/). Availability note: Images under original Open Images license; review official terms.
- **Conceptual Captions 3M** — Google Research; ~3.3M training pairs; ~16K validation. [Official source](https://github.com/google-research-datasets/conceptual-captions). Availability note: Image URLs can be unavailable; review dataset terms and source-image rights.
- **TextCaps** — Facebook AI Research / collaborators; Open Images images; 28,408 images; 145,329 captions in paper. [Official source](https://textvqa.org/textcaps/). Availability note: Dataset access through official page; images retain Open Images terms.
- **DOCCI** — Google Research and collaborators; 15,000 images; descriptions average 136 words. [Official source](https://google.github.io/docci/). Availability note: Official page states CC BY 4.0 for annotations/images.
- **Sydney-Captions / UCM-Captions / RSICD** — Remote-sensing captioning research community; not IITH-hosted in verified sources; Varies by dataset; use each source paper/release for exact counts. [Official source](https://people.iith.ac.in/ckm/assets/pdfs/conf_56.pdf). Availability note: Third-party datasets; obtain and follow each release’s access and licensing terms.
- **IITH-hosted image-captioning dataset** — IIT Hyderabad; Not applicable. [Official source](https://people.iith.ac.in/ckm/index.html). Availability note: No dataset identified; IITH papers and general vision datasets are not evidence of IITH dataset ownership.

## CaptionLens implementation sources

- Repository implementation: [README](../README.md), [architecture](../docs/architecture.md), [research methodology](../docs/research-methodology.md), [dataset evaluation](../docs/dataset-evaluation.md), [API](../docs/api-reference.md), [fine-tuning](../docs/fine-tuning.md).
- Verified code modules: app.py, model_pipeline.py, caption_metrics.py, evaluate.py, dataset_evaluation.py, database.py, fine_tune_lora.py.

## Scope and limitations

- IITH affiliation is not dataset ownership. Official IITH faculty materials identify captioning research; the scoped search did not verify an IITH-hosted or IITH-maintained captioning dataset. The remote-sensing paper uses Sydney-Captions and UCM-Captions; its affiliations do not change those datasets’ provenance.
- BLIP, BLIP-2, CLIPScore, ALOHa, MOCHa, DENEB and DOCCI use different tasks, model settings and metrics. Their results are not a common leaderboard.
- CaptionLens has no recorded benchmark runs. Faster R-CNN detections are independent of caption text and are not phrase grounding. The CHAIR-inspired evaluator is not canonical CHAIR. BLEU-4/ROUGE-L and manifest evaluation are utilities, not live API features. LoRA training code exists, but no run or API adapter loading is documented.
