# Open Source Developer Productivity - Late 2025 Follow-on Study Results

**See announcement blog post [here](https://metr.org/blog/2026-02-24-uplift-update/).**

This repository contains data and starter code for METR's expanded open-source developer productivity study which occured during DATE.

This experiment was a follow-up to the [early 2025 study on OS developer productivity](https://metr.org/blog/2025-07-10-early-2025-ai-experienced-os-dev-study/) using the same methodology. Due to changes in expectations of AI usefulness, we believe these new results are subject to increased selection that renders the results hard to interpret. 

__Please interpret these results with a major grain of salt, due to this increased selection, and see the blog post for further detail.__

# Data specification

`data.csv` contains all valid issues from the expanded study, including both developers from the original study as well as newly recruited developers.

The data includes:
- `id`: a unique, randomly generated ID for each issue.
- `developerId`: an integer ID for each developer.
- `treatmentAssignment`: either "AI-allowed" or "AI-disallowed" - issues are randomly assigned.
- `status`: what state the issue is in (`incomplete`, `complete`, or `ditched`). Note that invalid issues from cheating developers are filtered out of this dataset.
- `priorExperience`: Prior experience with this type of task. Input prior to randomization:
    - 1: Never done this task before
    - 2: Have seen this task done but never done it myself
    - 3: Have attempted this task once before
    - 4: Have done this task multiple times before
    - 5: I am an expert at this type of task
- `estimatedTimeWithAI`: how long (in minutes) the developer expects the issue to take with AI access (input before randomization).
- `estimatedTimeWithoutAI`: how long (in minutes) the developer expects the issue to take without AI access (input before randomization).
- `createdAt`: when the issue was randomized.
- `lastModified`: when the issue was last modified.
- `initialImplementationTime`: time (in minutes) to complete the initial implementation.
- `postReviewImplementationTime`: time (in minutes) to address post-review changes.
- `estimatedTimeOtherTreatment`: developer's post-completion estimate of how long it would have taken them to complete this issue in the other treatment would have taken (in minutes).
- `perceivedEffort`: Rate the difficulty from 1 (very easy) to 5 (very hard). Input after randomization.
    - 1: Minimal effort: extremely easy to implement, required very little effort or concentration.
    - 2: Below-average effort: easy to solve, required less effort than average.
    - 3: Average effort: required typical effort to implement, not notably different from other issues.
    - 4: Above-average effort: hard to solve, required more effort and concentration than average.
    - 5: Maximum effort: extremely difficult to solve, required very heavy effort and concentration.
- `isPanelDeveloper`: whether this issue was completed by a developer from the original panel study (True) or a newly recruited developer (False). (For clarity: all issues completed by panel developers were _new_ issues, not from the original study).

# Running the regression

First, install relevant dependencies:
```
pip install pandas numpy statsmodels scipy
```

Then, run the regression:
```
python regression.py --input-data data.csv
```

This runs three regressions: one on all developers, one on panel developers only, and one on new developers only. Each outputs the estimated speedup and a 95% confidence interval.

# Regression Details

The regression estimates the effect of AI treatment on developer productivity using a log-linear specification:

```
log(totalImplementationTime) ~ ai_treatment + log(estimatedTimeWithoutAI)
```
