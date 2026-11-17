# Data

The repository includes no event logs. Put every log you use here as `<name>.csv` with the columns

| Column | Meaning |
|---|---|
| `case` | case id |
| `event` | activity label |
| `completeTime` | timestamp of the event |

The file name without `.csv` is the dataset name used by all scripts. Preprocessed versions are cached in `Preprocessed/` on first use.

| Dataset name | Log in the paper | Source |
|---|---|---|
| `DomesticDeclarations` | BPI20-DD, BPI Challenge 2020 domestic declarations | BPI Challenge 2020 (4TU.ResearchData, https://data.4tu.nl) |
| `InternationalDeclarations` | BPI20-ID | BPI Challenge 2020 (4TU.ResearchData) |
| `RequestForPayment` | BPI20-RFP | BPI Challenge 2020 (4TU.ResearchData) |
| `PrepaidTravelCost` | pretraining source for `RequestForPayment` | BPI Challenge 2020 (4TU.ResearchData) |
| `BPIC2015_Recurrent` | BPI15-REC | BPI Challenge 2015 with recurring drift, see the paper for the exact version |
| `BPIC2017` | BPI17 | BPI Challenge 2017, 4TU.ResearchData |
| `IRO5000`, `ORI5000`, `ROI5000`, `OIR5000`, `RIO5000` | synthetic logs with recurring drifts | synthetic logs with five recurring drift segments, see the paper for the source |

Convert the original XES files to csv with the three columns above (for example with pm4py `convert_to_dataframe` and renaming `case:concept:name`, `concept:name`, `time:timestamp`).
The ground truth drift points of the logs that have them are in `Setup/drifts.py`. They only matter for the oracle trigger and the drift recovery analysis.
