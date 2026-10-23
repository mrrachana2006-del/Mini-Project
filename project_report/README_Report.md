# Anvaya Nova | Engineering Project Report

**Project:** AI-Based Agricultural Produce Quality Grading, Price Estimation and Value-Addition Recommendation
**Evidence cutoff:** 23 October 2026

This folder contains the complete engineering report in DOCX and PDF, plus the figures used in both documents.

## Key verified implementation facts

- Two separate EfficientNet-B0 image classifiers: fresh/rotten and visible grade A/B/C.
- The project includes image resolution/brightness/sharpness checks and command-line inference.
- The existing evaluation scripts were executed using saved checkpoints; no retraining was performed for this report.
- Fresh/rotten test set: 300 images; accuracy 95.33%, weighted precision 95.57%, weighted recall 95.33%, weighted F1 95.38%.
- Grade A/B/C test set: 79 images; accuracy 82.28%, weighted precision 83.27%, weighted recall 82.28%, weighted F1 82.35%.
- Price estimation, farmer-to-business/consumer connections, and value-addition recommendations are proposals, not implemented features.
- The models assess image appearance; they do not establish internal spoilage or food safety.

## Files

- `Complete_Project_Report.docx` - editable report.
- `Complete_Project_Report.pdf` - paginated report.
- `Figures/` - system architecture, data-flow, and inference-flow diagrams.

## Before submission

Fill in [STUDENT NAME], [USN], [GUIDE NAME], [DEPARTMENT NAME], [COLLEGE NAME], [ACADEMIC YEAR], and certificate/declaration dates and signatures. Add dataset source, attribution, license, collection and permission details; these were not documented in the repository. Check the institution's required formatting and citation style.

Evaluation values above are the observed output of `src/evaluate.py` and `src/evaluate_grading.py` on the available project test folders. See the report for confusion matrices, class counts, methodology, assumptions, limitations, citations, and proposed features.
