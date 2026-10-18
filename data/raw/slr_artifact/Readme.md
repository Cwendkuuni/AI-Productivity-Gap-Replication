# Replication Package for Systematic Literature Review (SLR)
This repository contains the replication package for a systematic literature review (SLR) on `The Impact of AI-Assistants on Software Developer Productivity`. It includes screening records, inclusion decisions, and exported raw data to ensure transparency and reproducibility.

### Pre-review Mapping and Control Paper Validation

Before executing the main SLR protocol, we conducted a **pre-review mapping phase** to define the scope, validate the inclusion and exclusion criteria, and verify the search string’s coverage. The flow chart diagram of the pre-review mapping can be found in .\ReviewPlanning - FlowChart.pdf

This phase included the following steps:

1. **Definition of Research Questions and Criteria**  
   The authors collaboratively defined the three main research questions (RQ1-RQ3) and drafted the inclusion/exclusion criteria according to Kitchenham and Charters’ SLR guidelines.

2. **Identification of Control Papers**  
   A preliminary manual search in Google Scholar retrieved **35 entries**, screened in full text.  
   Of these, **17 studies** were retained as *control papers* to test and refine the search string and ensure it captured all relevant studies. These steps were not part of main protocol of the SLR.

3. **Iterative Search Query Formulation**  
   Five refinement meetings were held to iteratively adjust the query syntax, Boolean logic, and keyword combinations across databases (IEEE, ACM, Web of Science, and ScienceDirect).  
   Each iteration was validated against the 17 control papers to ensure 100% retrieval consistency.

4. **Protocol Refinement**  
   The authors discussed and resolved borderline cases, to ensure shared interpretation of criteria before the main search.

5. **Transition to Main SLR Execution**  
   Once the search query successfully retrieved all control studies, the team proceeded with the full-scale database search and subsequent PRISMA-guided screening.


### Inclusion Exclusion criteria:

**Inclusion (IC)**:
- **(+) IC1**: The paper investigates the effect of AI or LLMs on software engineering developers' productivity.
- **(+) IC2**: The paper is in English.
- **(+) IC3**: The paper has an accessible full text and was published in 2014 or later.

**Exclusion (EC)**:
- **(-) EC1 (Out of Scope)**: The paper does not focus on software engineering or does not explore the impact of AI or LLMs on developers' productivity.
- **(-) EC2 (Out of Focus)**: The paper mentions the impact of AI or LLM on developers' productivity without it being one of the topics of the study.
- **(-) EC3 (Publication Type)**: The paper belongs to any of the following categories: secondary studies; work-in-progress, extended abstracts, posters, tool demos, editorials, or grey literature; studies published in books, theses, workshop, monographs, keynotes, panels, doctoral symposia, or any other venues without a formal peer-review process.
- **(-) EC4 (Length)**: The paper is a short publication with fewer than four pages.

## Contents

### Excel Files

- **SLR_Screening_Title_abstract.xlsx**  
  Contains all records retrieved from the databases during the title and abstract screening phase. Each entry includes: `Title`, `Author`, `Source`, `Decision`, and `Reason`.
  - The `Decision` column indicates inclusion or exclusion.

- **SLR_Screening_Full_Text.xlsx**  
  Records assessed during full-text screening for inclusion exclusion. Along with the justification of the crietria from

- **SLR_Snowballing.xlsx**  
  Documents studies identified via backward and forward snowballing, along with the source included from.

- **SLR_all_included_studies.xlsx**  
  The final list of included studies after all SLR phases. Each row represents a primary study with metadata:
  - Strategy
  - Procedure
  - Data analysis type (qualitative, quantitative)
  - Research focus (formative vs summative)
  
### Ris Files

- **Ris Files.zip/**  
  This folder contains the original `.ris` files exported from the digital libraries (e.g., IEEE Xplore, ACM Digital Library, Web of Science and Science Direct). These are the raw metadata records of the studies retrieved before preprocessing and screening.

