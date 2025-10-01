### Huawei DIGIX — Music-App User Retention

Goal: Predict whether a user will be active at 1, 2, 3, 7, 14, 30 days in the future.
Approach: Train six binary LightGBM models (one per horizon) with 5-fold CV, using last-30-day device activity and basic user attributes. Average fold probabilities and export a single submission.csv.

## Summary:

Features: day_1..day_30 recent activity + gender, age, device, city, is_vip.

Models: LGBMClassifier(num_leaves=31, learning_rate=0.1, n_estimators=500, objective='binary') with early stopping on AUC.

Output: submission.csv columns → device_id, label_1d, label_2d, label_3d, label_7d, label_14d, label_30d.

#Data

The baseline requires device activity and user info. 

Included in repo (small/medium):

data/1_device_active.csv – device-level daily activity for 60 days (device_id, days, day_1 … day_60).

data/2_user_info.csv – user meta (pipe-separated: device_id|gender|age|device|city|is_vip|topics). The script drops topics.

data/5_artist_info.csv (optional) – pipe-separated artist meta (not used in baseline).


## How it works (high level)

#Load & merge.

Read device activity (drop days) and pipe-separated user info.

Build six training frames by merging the past-30-day features with the future label for each horizon (day_31, day_32, day_33, day_37, day_44, day_60).

Build the test frame by renaming recent activity columns so they match the past-30-day schema (day_31..60 → day_1..30) and merging user features.

#Categoricals.

Treat gender, age, device, city, is_vip and the activity fields day_1..day_30 as categorical features for LightGBM.

Train 6 models with 5-fold CV.

KFold(5, shuffle=True, random_state=42), early stopping on AUC, average fold probabilities.

#Export.

Concatenate horizon probabilities into submission.csv:
device_id, label_1d, label_2d, label_3d, label_7d, label_14d, label_30d.

Includes a helper to downcast dtypes and trim memory.

##Reproducibility

Random state: 42

CV: 5 folds (shuffled)

Metric: AUC for early stopping (per fold)
