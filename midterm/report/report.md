# Midterm: TV programme recommendation on sweet.tv

<div class="meta">Aleksei Kosychev · HSE, LLM4Rec · Kaggle <code>sweettv-tv-program-recommender</code>, MAP@5 ·
Repository: <a href="https://github.com/ak1232320/rec-sys-hw1/tree/main/midterm">github.com/ak1232320/rec-sys-hw1/tree/main/midterm</a></div>

**Result.** The designated submission, `submissions/submission_habit.csv` (Kaggle ref 57498272), returned
**public 0.42168** (private 0.41476, Fig. 1). It replaced my first-round final, a LambdaRank hybrid at public
0.32279, after the public scores showed that the answer key counts **watched airings**, not seconds watched.
The final model multiplies each user's rate of watching a programme when it airs by how often it airs in weeks
31–42. It has no learned parameters and runs in about 25 s. The 80% "watched" rule and the idea of the model
came from material shared with me (§7, Disclosure); the implementation and every number here are my own runs.

<div class="fig kaggle">
<img src="figures/kaggle_score.png">
<p><em>Fig. 1. Returned score of the designated submission (competition page, Leaderboard tab): public 0.42168,
private 0.41476. All five submissions, refs and dates: <code>submissions/kaggle_scores.json</code>.</em></p>
</div>

| Submission | Kaggle ref | Public | Private | Hold-out MAP@5 (its own answer key) |
|---|---|---|---|---|
| **Habit × schedule (final)** | 57498272 | **0.42168** | **0.41476** | 0.3997 (watched airings) |
| LambdaRank hybrid (first round) | 57496713 | 0.32279 | 0.29662 | 0.3470 (seconds) |
| Channel-hour exposure | 57496711 | 0.29113 | 0.26829 | 0.3240 (seconds) |
| PPR on the co-view graph | 57496709 | 0.29038 | 0.25733 | 0.3076 (seconds) |
| Popularity (total seconds) | 57496705 | 0.28361 | 0.27447 | 0.1811 (seconds) |

## 1. Data and the join

The log has 5,821,987 sessions of 4,914 users on 202 channels in weeks 11–30. A session's `duraton` is the
length of the viewing interval (median 396 s, max 342,145 s ≈ 95 h). The schedule has 3,444,007 airings on 405
channels. All 1,260 test users appear in the log.

**Join rule.** A session (user, channel, start, stop) is matched to every airing on the same channel
(`vsetv_id` = `channel_id`) whose interval [start, start + duration) overlaps it. Each match carries the
overlap in seconds, so a session that spans three airings credits each with its own share. The join runs per
channel with two binary searches (`src/intervals.py`, 8 unit cases). It produces 7,752,424 matches, and 94.1%
of logged seconds fall on some airing; 5 log channels have no schedule.

**`tv_show_id = 0`.** Only 10.8% of schedule rows in weeks 11–20 carry a programme id (11.1% over all three
files, 16.5% of airtime). Unlabelled airings stay in the join, so their seconds are not credited to a
neighbouring labelled airing. The channel-hour habit feature also uses them as context. They are never items.
Labelled airings still receive 53.1% of all logged seconds: viewers concentrate on what is labelled.

**What counts as watched.** A user's *coverage* of an airing is their overlap seconds summed over all their
sessions, divided by the airing's length and capped at 1. The airing counts as **watched at coverage ≥ 0.8**.
The answer key is each user's five programmes with the most watched airings. This is the organisers' rule as
relayed in the shared material. I could not read it on the Kaggle page itself, but §4 shows the public scores
agree with it. The median airing runs 100 min and the median single overlap covers 11.6% of it, so the rule
keeps deliberate viewing only: 24.9% of user–airing pairs in weeks 21–30 qualify. My first round used an
absolute cut instead: an overlap counted from 60 s, which drops channel-zapping (2.7% of overlaps, 0.07% of
seconds). That round also counted preference in seconds.

One defect surfaced on the way. Reading the log with pandas' *pyarrow* engine routes the 20-digit user ids,
which exceed 2⁶³, through float64. It silently rewrote 1,982 of the first 1,999 ids, and only 4 of 1,260 test
users could be found. The C parser is exact, and `tests/check_pipeline.py` compares 300,000 parsed ids with
the raw text.

## 2. Protocol

**Split.** Hold-out: history in weeks 11–20, answer in weeks 21–30. These are the two training slices of the
schedule, and the 10-week horizon is close to the test's 12. 4,599 users have both a history and an answer.
MAP@5 is reported for all of them and for the **cohort** of 1,230 who are also in the Kaggle file; the cohort
are heavier viewers (median 1,377 sessions). The hold-out is used only for selection. The first round's ranker
learned on an inner split, weeks 11–15 → 16–20. The final run uses history 11–30 and the schedule of weeks
31–42. Checks 5 and 6 verify the time boundary: the hold-out features and scores do not change when every log
row after week 20 is deleted.

**Candidates and repeats.** Every programme with a labelled airing in the target weeks is a candidate: 6,376
on the hold-out and 6,642 in the test. Already-watched programmes are allowed for every method, since 54.9% of
hold-out answer pairs are programmes the user had already watched.

**Five ids and the fallback.** Each method scores all candidates. The list is sorted by score, with ties broken
by popularity and then by id, and the first five are taken; scores from different methods are never mixed.
Popularity means the number of users who have the programme in their own history top-5. Because it breaks
every tie, a zero score (no evidence) falls back to it in the same way in every method. In the test, 2 users
have no labelled history at all. The final model scores no programme for 5 users and fewer than five for 28.
MAP@5 follows Kaggle's `ml_metrics` definition and is checked against its reference cases.

## 3. Methods on one hold-out

<!-- METHODS_TABLE -->
<p class="tcap"><em>Table 1. MAP@5 on the hold-out. The competition key is the top-5 by watched airings. The
seconds key is the first round's answer (top-5 by seconds), with each method re-selected for it. "&lt;5
scored" is the number of cohort users with fewer than five non-zero scores (filled by popularity). Time is for
scoring 4,599 users on an Intel i5-13420H.</em></p>

- **CBF**: TF-IDF over metadata tokens (main channel, category, three genres, director and actors, title
  words). The user profile is the seconds-weighted mean of the user's programmes; candidates are scored by
  cosine. A profile in log-minutes scored only 0.021: a long tail of once-watched films, each with three
  genres and a director, outvoted the few programmes a person lives on. Weighting by seconds lifted it to
  0.197. Of the metadata fields, the task's own (category, genres, people) reach 0.115; the channel adds 0.043
  and title words a further 0.039.
- **Association rules**: X → Y over baskets of programmes watched for at least a few minutes, with confidence
  summed over the rules a user's basket fires. The rule Y → Y is excluded, so this method can only recommend
  through co-viewing, never through the user's own repeat habit.
- **MF**: implicit ALS (Hu, Koren & Volinsky) fits a binary "watched" target and uses time only as confidence,
  so a sampled film and the daily news both get target 1. It worked only once positives were restricted to
  habits (≥ 20 h), and that setting leaves 801 of the 1,230 cohort users with no positive at all: their lists
  are popularity. PureSVD reconstructs the magnitudes and does better.
- **PPR** runs on the programme co-view graph (top-50 neighbours) and restarts from the user's own programmes,
  so α moves the ranking from the user's history (α → 0) towards what their programmes' audiences watch.
- **Channel-hour exposure** (first round) multiplies the user's seconds per (channel, half-hour of the week)
  by each programme's minutes in that slot in the target schedule.

Under the competition key, none of the four course methods clearly beats popularity (PPR 0.305 vs 0.304), for
two reasons. They consume seconds, which is the wrong currency for this key: the same history scores 0.263
counted in seconds and 0.376 counted in watched airings (Fig. 2b). They also ignore the target-week schedule.

<div class="fig wide2">
<img src="figures/fig_methods_probe.png">
<p><em>Fig. 2. (a) Table 1's competition-key column for the cohort. (b) One ranking (each user's own
history), counted five ways, scored under answer keys counted the same five ways. Bold marks the best ranking
in each column. Each key is best served by its own count, and counts of airings or days lift plain history to
0.45.</em></p>
</div>

## 4. Finding the answer key, and the final model

**First round.** Everything counted seconds. The hybrid generated candidates from the user's own programmes,
each method's top-N and the exposure feature (70.5% of answer pairs reached the pool), then re-ranked them
with LambdaRank (LightGBM). On the hold-out it scored 0.347 against 0.324 for the best single signal, and
0.32279 on Kaggle. The probe was popularity: 0.181 on the hold-out but 0.284 on Kaggle, almost level with the
personal methods. I re-labelled the hold-out under other plausible keys and scored fixed rankings against each
(Fig. 2b; `experiments/probe_target.py`, `probe_matrix.py`). Count-based keys reproduce the public pattern:
with a views key, popularity lands at 0.284 on the hold-out. A ranking counted the same way as the key wins its
column. The key therefore counts airings, not seconds. The shared material then supplied the exact rule
(≥ 80% coverage).

**Habit × schedule** (`src/habit.py`), for a user *u*, programme *s* and channel *c*:
*q* = watched airings of *s* on *c* in the history weeks, each weighted 0.5<sup>age / 10 weeks</sup>;
*e* = airings of *s* on *c* on days when *u* appears in the log at all, with the same weights;
rate = min(1, *q* / (*e* + 30)); score(*u*, *s*) = Σ<sub>*c*</sub> rate × (airings of *s* on *c* in the
target weeks). The top five are taken, and empty slots go to popularity. The schedule factor makes the model
forward-looking: a programme that is off air, or rarely broadcast, in the target weeks cannot rank.

**What carries it** (cohort MAP@5): full model 0.3997. Without the target-week schedule it falls to 0.187;
without the prior, 0.367; without recency weighting, 0.397. Without the active-day correction it scores
0.4001: the June logging gap that motivates the correction lies outside my hold-out history, so this check can
neither confirm nor refute it. A sweep of half-life {5, 10, 20} weeks × prior {10, 30, 100} stays within
0.393–0.3997, with the best at the submitted 10 / 30. One choice was my own: the description specifies
recency weights for *q* only, and I applied the same weights to *e*.

## 5. Two users

Both users are from the hold-out cohort: the users at the 90th and 10th percentiles of watched airings in
weeks 11–20. The rule is fixed in `experiments/two_users.py` and was set before the lists were seen. Ids in
**bold ✓** are in the user's answer for weeks 21–30, and <sup>n</sup> marks a programme the user had not
watched to 80% before. Titles and categories are looked up from the schedule as they appear in one of the programme's listings (an
id groups listings whose titles vary, e.g. *ТСН* and *ТСН. Неделя*).

<!-- TWO_USERS -->

## 6. Which strategy would I select for this platform, and why?

**Habit × schedule as the default "your programmes this week" list, with popularity filling empty slots, and a
co-view method (association rules) as a separate "viewers like you also watch" row.**

- **Validation and test.** Habit × schedule scores 0.400 on the hold-out cohort, against 0.305 for the best
  course method and 0.304 for popularity. On Kaggle it reached 0.422, against 0.284 for popularity. Its
  hyperparameters are flat (0.393–0.400 across the sweep), so the score does not hinge on tuning.
- **Familiar preferences, not discovery.** All five of its ids are programmes each user already watched. For
  the heavy viewer it found *Факты. День* and *Сегодня* but missed *ТСН*. Popularity and PPR caught *ТСН*: the
  user had watched it to the end only twice in weeks 11–20, and it entered their top-5 in weeks 21–30. For the light
  viewer, four of the five answer programmes were ones they had never watched to the end (three without a
  single minute-long overlap in their history), and the model could return only the user's own five. Association rules found *Мир наизнанку*, which the user had not watched before and did watch in
  weeks 21–30. Co-viewing reaches where habit has no data. That is one user, not evidence that the rules'
  other unfamiliar picks are useful.
- **How the methods differ.** Association rules capture directional co-viewing and, by construction, nothing
  of the user's own habit. MF compresses taste into factors and, with ALS, loses the difference between habit
  and sample. CBF knows what kind of programme a user watches but not which one, so its lists drift to
  same-category shows (0.197). It is still the only method that can place a programme nobody has watched.
  PPR's diffusion pulls the list towards what its neighbours' audiences watch; at the selected α = 0.85 it
  ends level with popularity. Popularity does well because ten programmes make up 45.9% of all answer rows:
  five news and morning programmes, three children's cartoons and two documentary-style series.
- **Cost and experience.** The final model is a few joins and group-bys: about 25 s for 1,260 users, no
  training, and each item can be explained ("you watch it 3 times in 5 when it airs; it airs 60 times in the
  coming weeks"). The LambdaRank hybrid took 3–8 min end to end in my runs and scored lower here. The lists it
  produces repeat each household's linear-TV habits, with no exploration. That is what the metric
  rewards, but a discovery row would need its own evidence, which this data cannot give.

**Limitations.** Only 10.8% of schedule rows carry an id, so 47% of logged seconds can never become an
interaction. The answer-key rule was inferred from public scores and taken from shared material, not read from
the organisers. The history (March–July 2020, a lockdown period) and the test (August–October) are different
seasons, and the hold-out cannot show how much that costs. The log covers this platform only: absence from it
does not mean a user never saw a programme. Children's cartoons among the most common answers suggest that
one account is often a household, not a person. 65 cohort users receive fewer than five scored programmes, and
the four course methods were evaluated on seconds input. Re-counting their input in watched airings is the
obvious next step, and it is untested. No online test or business outcome is claimed.

## 7. Verification, reproduction and records

**What I personally verified.** On the competition page I confirmed that the designated submission returned
public 0.42168 (Fig. 1 is my own screenshot). I opened `submissions/submission_habit.csv` and checked its
layout: a header, then a user id and five programme ids per row. On my machine I ran
`python tests/check_pipeline.py`, and every check passed; among them are the format check of all 1,260 rows
and both leakage checks. I also looked up programme 6500479 from Table 2 in the raw weeks 21–30 schedule
(`export_arh_21-30-final.csv`) and found the title *Мир наизнанку* and the category *Познавательное*, as the
table shows.

**Reproduction.** `python src/ingest.py` (about 1 min) → `python src/predict.py` (about 25 s) writes the
designated file. `--first-submission` rebuilds the four first-round files. `tests/check_pipeline.py --repro`
runs 9 checks: unit cases, exact ids, join bounds, time order, leakage for both rounds, file format, the 80%
rule, and a byte-identical rebuild of all five files sent to Kaggle. Environment: Windows 11, Python 3.12.10,
12 logical CPUs, 15.7 GB RAM; pandas 2.3.3, NumPy 2.3.5, SciPy 1.17.0, scikit-learn 1.8.0, implicit 0.7.3,
LightGBM 4.6.0. Seed 20260112 for ALS, SVD and LightGBM; the final model is deterministic.

**Disclosure.** *Shared material:* a classmate's write-up of their solution (repository
`Marrrik/sweettv-recsys`) gave the 80% rule and the idea of the final model, with its half-life and prior. I
implemented it from that description. I read the repository's README once, to find where the rule came from,
and did not read or copy its code. *AI assistance:* the work was done with Claude Code, and the unmodified
session export is submitted with this report. *Libraries:* as listed above, plus Matplotlib. *Data:* Kaggle
competition files, not redistributed.
