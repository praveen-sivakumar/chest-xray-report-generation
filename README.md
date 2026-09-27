# Human-Centred Chest X-Ray Report Generation with Vision-Language Models

My MSc thesis project. It generates a written radiology report from a pair of chest X-rays (frontal and lateral), explains what the model looked at with Grad-CAM, and puts a clinician in charge of the final report through an editable Streamlit interface.

The project follows a **Value Sensitive Design** approach, so values like transparency, trust, accountability and human oversight shaped the design from the start rather than being added at the end.

**Thesis:** *Value Sensitive Design Approach for Human-Centered Automated Chest X-Ray Report Generation Using Large Language Models and Computer Vision*
MSc in Human-Centred Artificial Intelligence, Technological University Dublin (THES H6011, 30 credits)
Supervisor: Abdelsalam Busalim (TU Dublin) · Industry partner: Junke Xu (CeADAR)

## Why this matters

Radiologists are under heavy workload, and writing reports is a big part of it. AI models can now write fluent reports from images, but fluent isn't the same as trustworthy. Many systems produce text that reads well yet isn't clearly tied to what's in the image, and they give doctors no easy way to check or correct the output. Under the EU AI Act, a system like this counts as high-risk AI, so transparency and real human oversight are requirements, not extras.

## Research questions

1. How can Value Sensitive Design guide the development of an AI radiology report generation system?
2. How can language models be fine-tuned for chest X-ray report generation?
3. How far can explainability and a clinician-centred interface improve transparency, usability and trust in AI-generated reports?

## Results at a glance

- **Model comparison:** R2GenCMN reached a mean ROUGE-L of **0.32** on the test set, against **0.10** for ViT + GPT-2
- **Report quality:** R2GenCMN wrote clear, well-structured reports. ViT + GPT-2 produced broken words, fragmented sentences and made-up content
- **Training length:** the 5-epoch R2GenCMN model beat the 10-epoch one (ROUGE-L 0.352 vs 0.332) and produced more varied reports, so it was kept as the final model
- **Explainability:** Grad-CAM heatmaps for both X-ray views mostly focused on the lung fields and heart
- **Human in the loop:** a Streamlit app where a clinician uploads the two X-rays, sees the AI draft next to the Grad-CAM heatmaps, edits the report and saves both versions

## Dataset

[Indiana University Chest X-Ray (IU X-Ray)](https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university), a public, de-identified dataset.

- 7,466 images and 3,851 reports, with frontal and lateral views
- Kept only studies with both a frontal and a lateral image and a non-empty report
- Built the report text from the *Findings* and *Impression* sections, then cleaned it (lowercase, extra spaces removed, medical terms kept)
- Split by patient study (UID) so no study appears in more than one split
- Reports are mostly short, with a median of about 34 words, and a large share of them describe normal findings

## Method

### 1. Two models compared

| | ViT + GPT-2 | R2GenCMN |
|---|---|---|
| Image encoder | Vision Transformer (`vit-base-patch16-224-in21k`) | ResNet-101 |
| Text decoder | GPT-2 | Transformer with a cross-modal memory network |
| How the two views are used | Frontal and lateral combined into one image | Each view encoded separately, then fused |
| Built for | General image captioning | Medical report generation |

ViT + GPT-2 was fine-tuned with Hugging Face `VisionEncoderDecoderModel` for 5 epochs (AdamW, lr 5e-5, batch size 8, mixed precision). R2GenCMN was trained from its [original implementation](https://github.com/zhjohnchan/R2GenCMN) for 5 epochs on the two-view IU X-Ray data, running on an NVIDIA L40S GPU.

### 2. Evaluation

- **Quantitative:** ROUGE-L F1 against the real reports, plus how many of the generated reports were unique and how often the most common report repeated
- **Qualitative:** reading the outputs for clinical sense, anatomical correctness and whether they follow normal radiology report style

### 3. Hyperparameter tuning (R2GenCMN)

Five focused runs of 3 epochs each, changing one setting at a time:

- Learning rate: 1e-4, 5e-5, 1e-5
- Dropout: 0.1, 0.3
- Batch size: 4, 8

Each run was scored on ROUGE-L and on a "balanced score" that also rewards varied output, so a model that writes the same report every time can't win. The best balance came from **lr 1e-4, dropout 0.1, batch size 8**.

### 4. Longer training

The chosen setup was trained for 10 epochs and compared with the 5-epoch model on the same 44 test studies:

| Model | ROUGE-L | Unique reports (of 44) | Most common report used |
|---|---|---|---|
| **R2GenCMN, 5 epochs** | **0.352** | **5** | 17 times |
| R2GenCMN, 10 epochs | 0.332 | 3 | 29 times |

Longer training improved a few individual reports, but overall the model leaned even harder on one "normal" template. More training didn't mean better generalisation, so the 5-epoch model was kept.

### 5. Explainability

Grad-CAM was applied to the last convolutional layer of R2GenCMN's ResNet-101 image encoder. Separate heatmaps for the frontal and lateral views are overlaid on the X-rays. The aim isn't to prove the diagnosis is right, but to show whether the model is looking at sensible areas like the lungs and heart.

### 6. Streamlit app (`app.py`)

1. Upload a frontal and a lateral chest X-ray
2. The model generates a draft report
3. Grad-CAM heatmaps are shown for both views
4. The clinician edits the report in the app
5. The heatmaps, the AI draft and the edited report are saved together, so there's a record of what the AI wrote and what the clinician changed

## Human-centred design and regulation

- **Value Sensitive Design:** a stakeholder survey on trust, transparency, explainability, oversight and accountability, used to inform design decisions, plus an ethical matrix
- **EU AI Act:** treated as a high-risk system, with the design built around transparency, traceability and human oversight
- **GDPR:** de-identified public data, data minimisation and use limited to research
- **ALTAI:** transparency through Grad-CAM, human agency through required clinician review, and accountability through traceable inputs, outputs and model settings
- **UN SDGs:** 3 (good health and well-being), 9 (industry, innovation and infrastructure) and 16 (peace, justice and strong institutions)

The system is built to support radiologists, not to make decisions on its own.

## Limitations

- IU X-Ray is small, and most of its reports describe normal findings, which pushes models towards repeating a "normal" template
- ROUGE-L measures word overlap, not whether a report is clinically correct
- The dataset has no demographic information, so fairness across patient groups couldn't be checked
- Grad-CAM shows where the image encoder is looking, not which part of the image led to which sentence

## Future work

- Train and test on larger datasets such as MIMIC-CXR, and test across datasets
- Reduce template repetition with better decoding (nucleus sampling, contrastive decoding) or reinforcement learning with clinical rewards
- Try stronger image encoders such as Swin Transformer or ConvNeXt
- Explanations at the word level, linking each part of the report to a region of the image
- Grow the Streamlit app into a fuller decision-support tool with feedback loops, audit trails and PACS integration

## Tech stack

Python, PyTorch, Hugging Face Transformers, ViT, GPT-2, R2GenCMN, ResNet-101, torchvision, Grad-CAM, Streamlit, OpenCV, pandas, NumPy, Matplotlib, GPU training on an HPC cluster

## Files

| File | What it does |
|---|---|
| `00_sanity_check.py` | Checks the train/val/test CSVs load and the image paths exist |
| `sanity.txt` | Output of the sanity check |
| `01_Data_Exploration.ipynb` | Loads IU X-Ray, pairs frontal and lateral views, cleans reports, report length analysis, splits |
| `02_ViT_GPT2.ipynb` | Builds and fine-tunes the ViT + GPT-2 model |
| `03_ViT_GPT2_inference.ipynb` | Generates ViT + GPT-2 reports for the 44 test studies |
| `04_R2GenCMN.ipynb` | Sets up R2GenCMN, builds the two-view annotation file, smoke test and 5-epoch training |
| `05_r2gencmn_inference.ipynb` | Generates R2GenCMN reports for the 44 test studies |
| `06_Vit_GPT2_vs_R2GenCMN.ipynb` | Compares both models on ROUGE-L, diversity, length and example outputs |
| `07_R2GenCMN_hyperparameter_tuning.ipynb` | Hyperparameter runs and the 10-epoch training run |
| `08_Best_Model.ipynb` | Compares the 5-epoch and 10-epoch models and picks the final one |
| `09_Predictions.ipynb` | Runs the final model on test studies |
| `10_XAI.ipynb` | Grad-CAM for both views, with a small saved gallery |
| `app.py` | Streamlit app for report generation, Grad-CAM and clinician editing |
| `X00230952_Praveen_Sivakumar_Thesis_Report.pdf` | The full thesis |

The IU X-Ray images, the R2GenCMN source code and the trained model checkpoints (about 950 MB each) are not included here. The notebooks point to paths on the university HPC cluster where they were stored.

## Acknowledgements

- R2GenCMN: Chen et al., *Cross-modal Memory Networks for Radiology Report Generation* ([code](https://github.com/zhjohnchan/R2GenCMN))
- IU X-Ray dataset: Demner-Fushman et al., *Preparing a collection of radiology examinations for distribution and retrieval*, JAMIA
