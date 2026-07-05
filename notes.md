# Project Notes

## Objective

This project studies narrative communities in FIFA World Cup 2026 YouTube comments. The aim is not a basic sentiment result such as "fans of winning teams are happier." Instead, the project asks how YouTube users cluster around recurring narratives such as matches, underdogs, controversies, rivalries, tactical blame, national pride, and star-player debates.

## Research Questions

1. Do World Cup 2026 YouTube discussions form participation communities around specific teams, matches, or broader narratives?
2. Which narratives dominate detected communities, and how are they expressed through entities, keywords, sentiment, and emotions?
3. Which users, videos, matches, teams, players, or controversy-related entities bridge different narrative communities?

## Data Collection Process

The collection pipeline has three steps.

1. Candidate discovery: the script searches YouTube using predefined World Cup event and narrative query groups from `config/youtube_queries.json` rather than one broad generic query.
2. Automatic eligibility filtering: discovered videos are deduplicated by `video_id`, enriched with basic YouTube metadata, and written into one compact sampling file with either `eligible` or `excluded` status.
3. Comment collection: the script collects capped top-level comments only from rows marked `eligible`.

This design is simple enough for a short presentation, but still reproducible and defensible. It captures multiple World Cup discussion spaces instead of depending on one narrow search phrase.

## Output Files

- `metadata/video_sampling_pilot.csv`
  Compact video sampling file. It contains both eligible and excluded videos, the final eligibility decision, one exclusion reason if excluded, and one `video_context_type` label for later interpretation.
- `data/raw/youtube_comments_pilot.csv`
  One row per collected top-level comment. It does not repeat video metadata; comments can be joined back to the sampling file through `video_id`.
- `outputs/tables/data_collection_summary_pilot.csv`
  Small key-value summary containing video counts, comment counts, user-overlap diagnostics, category and context counts, and exclusion reasons.

## Eligibility Rules

- `below_min_comments`: videos with fewer than 100 public comments are excluded because they are too sparse for useful network or content analysis.
- `comments_unavailable`: videos with disabled or unavailable comments are excluded because comments are the main unit of analysis.
- `non_public`: non-public videos are excluded so the corpus uses accessible public content.
- `live_or_scheduled`: live or scheduled broadcasts are excluded because the project focuses on standard video comment sections, not live chat.
- `likely_short`: videos shorter than 90 seconds are excluded as a transparent proxy for short-form content.
- `clearly_non_english`: clearly non-English videos are excluded because the planned text-analysis tools are English-oriented.
- `irrelevant_gaming_or_betting`: clearly gaming, simulation, betting, fantasy, or similar content is excluded because it is not real World Cup 2026 discussion.
- `missing_required_metadata`: videos missing key metadata needed for filtering are excluded for reproducibility.

## Video Context Type

Each video is assigned one compact `video_context_type` using simple rule-based heuristics from the title, description if available, primary category, and search categories. This is not a machine-learning model.

The rule order is intentionally simple:

1. controversy terms such as VAR, referee, penalty, controversy, or robbed
2. highlight terms such as highlights, goals, or full highlights
3. reaction terms such as post-match reaction, fan reaction, or instant reaction
4. underdog terms such as upset, shock, or surprise
5. star-player terms such as legacy, GOAT, hat-trick, or obvious player names
6. tactical terms such as analysis, manager, formation, or why a team lost
7. rivalry or national-pride terms
8. group draw, roster, squad, or players-to-watch terms
9. watchalong or live-reaction terms
10. preview or background terms
11. otherwise `other_world_cup_discussion` or `uncertain`

Allowed values are:

- `match_highlights`
- `post_match_reaction`
- `var_referee_controversy`
- `upset_underdog`
- `star_player_legacy`
- `tactical_analysis`
- `rivalry_national_pride`
- `preview_background`
- `group_draw_roster`
- `watchalong_reaction`
- `other_world_cup_discussion`
- `uncertain`

`video_context_type` is video-level metadata only. It is useful later for diagnostics and community interpretation, but it does not mean every comment under the video expresses that same narrative.

## Comment Collection Choices

The script collects top-level comments only. It stores `reply_count`, but not reply text.

This is a deliberate simplification:

- top-level comments are usually standalone reactions and are enough for user-video participation and co-commenting networks;
- reply text would be more useful for direct interaction analysis, but adds complexity that is unnecessary for this project stage;
- the 200-comment cap stops very large videos from dominating the pilot corpus;
- YouTube relevance ordering captures prominent platform-ranked discussion, even though that ranking is not neutral.

## Preprocessing

Before preprocessing, the script runs quick checks on video and comment counts, duplicate IDs, user overlap across videos, category and context balance, and video language metadata.

Raw comments are deduplicated by `comment_id`. The script preserves `original_text` for possible qualitative inspection and creates a lightly cleaned `clean_text` by decoding HTML entities, removing URLs, replacing line breaks, collapsing whitespace, and stripping the final text. Punctuation, capitalization, and negation are kept.

Very short comments are removed with `min_chars = 15` because they usually add little semantic information. Likely non-English comments are also removed because the later text-analysis tools are English-oriented. This language filter is a rule-based heuristic, not a full language-detection model: it uses ASCII or Latin-letter ratio, common English function words, football terms, and token count. It is transparent but imperfect.

Users are pseudonymized with a stable hashed `author_id`. The same YouTube channel always maps to the same anonymous ID, which preserves repeat participation for later network analysis without using raw usernames or channel IDs in the processed dataset.

Compact video metadata is merged through `video_id`. The preprocessing outputs are `data/processed/comments_processed.csv` and `outputs/tables/preprocessing_summary.csv`.

Current preprocessing result:

- 22,594 raw comments
- 19,152 processed comments
- 16,505 processed users
- 134 processed videos
- 1,207 users with at least 2 videos
- 298 users with at least 3 videos

The repeat-user count is sufficient to proceed with the user-video and user-user co-commenting network construction.

## Network Construction

The network step builds two related participation structures from `data/processed/comments_processed.csv`.

The user-video network is the direct bipartite participation structure in the data. One edge means an anonymized user commented on a specific video discussion space. This tells us where participation happened before any projection or community detection.

The user-user co-commenting network is a projection of that bipartite structure. Two users are connected when they both commented on at least one of the same videos. The edge weight is the number of shared videos. This is shared participation, not direct interaction, friendship, agreement, or reply behavior.

This is the most defensible network representation for the project because we do not have follower or friendship data, and we collected top-level comments rather than full reply conversations. Co-commenting therefore captures repeated presence in the same World Cup discussion spaces without overstating what the data can show.

Edge-weight thresholds are diagnostic at this stage:

- weight >= 1 is broad and inclusive but may contain many one-off overlaps from popular videos
- weight >= 2 captures stronger repeated co-participation
- weight >= 3 is stricter and useful as a robustness check, but it may remove many users

The goal of this step is not final interpretation yet. It only builds the networks and compares threshold diagnostics so we can later choose a suitable community-detection input.

Outputs from this step:

- `data/processed/user_video_edges.csv`
- `data/processed/user_network_edges.csv`
- `data/processed/user_network_nodes.csv`
- `outputs/tables/network_summary.csv`
- `outputs/figures/user_participation_distribution.png`
- `outputs/figures/network_threshold_diagnostics.png`

## Community Detection

### Methodology

Community detection is run after constructing the user-user co-commenting network. The main graph uses edge weight >= 2, where edge weight means the number of videos two users both commented on. In practice, weight >= 2 means two anonymized users shared at least two World Cup video discussion spaces. This threshold was chosen because weight >= 1 was too noisy, while weight >= 3 retained too few users.

Louvain community detection is then applied to the weighted graph. Louvain groups users whose repeated co-commenting connections are denser within the group than outside it.

### Interpretation

The detected groups are participation communities. They represent anonymized users who repeatedly appear in similar World Cup video discussion spaces. They do not represent friendship, direct replies, agreement, or support for the same team. Communities are first profiled with video metadata before any text analysis.

### Results

The current main graph contains 931 users and 4,133 repeated co-commenting edges. Louvain detects 71 communities with modularity = 0.5505. The largest community contains 273 users and the median community size is 2. There are 64 communities with fewer than 10 users, while 6 communities have at least 20 users. Those 6 larger communities cover 76.37% of users in the main graph.

### Insights

The community structure has a clear long tail. A few larger communities contain most of the interpretable structure, while many small communities are peripheral and should not be deeply interpreted. The six larger communities are therefore the main focus for later content analysis.

The preliminary video-metadata profiles already suggest different discussion spaces, including match highlights, post-match reactions, VAR or referee controversy, star-player debate, and tactical blame. Later content analysis will validate and refine these tentative labels.

### Figures

- `community_size_distribution.png` shows the top communities plus the grouped long tail
- `community_size_long_tail.png` shows all detected communities and makes the long-tail structure visible
- `community_context_profile.png` shows video-context composition of the main communities
- `community_category_profile.png` shows sampling-category composition of the main communities
- `community_network_overview.png` provides an illustrative view of the main graph
- `community_highlight_C1.png` to `community_highlight_C5.png` highlight the five largest communities inside the same network layout

Network layout positions in the overview and highlight figures are visual aids only and should not be overinterpreted as exact distances.

Outputs from this step:

- `data/processed/user_communities.csv`
- `data/processed/community_summary.csv`
- `data/processed/community_video_profiles.csv`
- `outputs/tables/community_detection_summary.csv`
- `outputs/figures/community_size_distribution.png`
- `outputs/figures/community_size_long_tail.png`
- `outputs/figures/community_context_profile.png`
- `outputs/figures/community_category_profile.png`
- `outputs/figures/community_network_overview.png`
- `outputs/figures/community_highlight_C1.png`
- `outputs/figures/community_highlight_C2.png`
- `outputs/figures/community_highlight_C3.png`
- `outputs/figures/community_highlight_C4.png`
- `outputs/figures/community_highlight_C5.png`

## Content Analysis

### Objective

Content analysis profiles the detected participation communities using the actual comment text. Community detection shows which users cluster together structurally through repeated co-commenting, while content analysis helps explain what those communities are talking about. In this project, the main interpretive focus is on the six larger communities because they cover most users in the main graph and provide the clearest basis for comparison.

This step mainly supports RQ2: "Which narratives dominate the detected communities, and how are they expressed through entities, keywords, sentiment, and emotions?" The goal is not to replace the network analysis, but to complement it by adding substantive evidence about the themes, actors, controversies, and tones that characterize each community.

### Input Data

The content-analysis script combines three processed inputs:

- `data/processed/comments_processed.csv`
- `data/processed/user_communities.csv`
- `data/processed/community_summary.csv`

Comments are linked to communities through `author_id`, so each comment inherits the detected community membership of its author. This means the analysis profiles shared participation spaces rather than direct user-to-user conversations.

Only pseudonymized user identifiers are used in this step. The workflow does not rely on raw usernames or channel IDs for interpretation.

### TF-IDF Terms

TF-IDF is used to identify distinctive unigrams and bigrams for each community. This is preferable to raw word frequency because raw counts would overemphasize generic football vocabulary that appears across many comments. TF-IDF instead highlights terms that are especially characteristic of one community relative to the others, which helps identify narrative signals rather than just common tournament language.

Using both unigrams and bigrams is important because some narratives appear most clearly as short phrases rather than single words. Examples include expressions such as `offside decision`, `Steve Clarke`, or `World Cup`. These terms help identify the vocabulary around a community, but they are not treated as full topics by themselves. They are used as supporting evidence for later narrative interpretation.

### Entity Extraction and Normalization

The course discussed NER, NEL, and NED, and this project applies those ideas in a limited, report-oriented form. Entity extraction uses a hybrid approach:

1. a football/domain dictionary;
2. optional spaCy NER if `en_core_web_sm` is available.

spaCy NER is the broader generic layer. It tries to detect entities such as people, places, countries, organizations, and events. So it can help catch names or countries outside our dictionary. Then our normalization logic should merge aliases where possible. 

So the dictionary captures football-specific entities and known aliases; spaCy adds general-purpose named entities such as people, organizations, and geopolitical entities. The final entity table normalized overlapping mentions into canonical forms. 

The domain dictionary is necessary because generic NER may miss football-specific concepts that matter for interpretation. Terms such as `VAR`, `offside`, `penalty`, `disallowed goal`, `goalkeeper`, and `referee` can be central narrative objects in World Cup discussion even when a general-purpose model does not reliably mark them as named entities.

Entity normalization then maps multiple surface forms to canonical labels so that comparable mentions are counted together. For example:

- `Ronaldo`, `CR7`, `Cristiano Ronaldo` -> `Cristiano Ronaldo`
- `Messi`, `Leo Messi`, `Lionel Messi` -> `Lionel Messi`
- `Mbappe`, `Kylian Mbappe` -> `Kylian Mbappe`
- `ref`, `refs`, `referee` -> `Referee`
- `keeper`, `goalkeeper` -> `Goalkeeper`
- `offside`, `off-side` -> `Offside`

This normalization is inspired by NEL/NED because surface forms are mapped to canonical entities. However, it is not full external entity linking. The project does not use Wikidata, DBpedia, or other external entity IDs. That choice is deliberate because the aim is community-level narrative profiling, not the construction of a knowledge-base-linked entity graph.

### VADER Sentiment

Sentiment is measured with VADER through the `vaderSentiment` package. Each comment receives a compound polarity score and is then classified using the standard thresholds:

- `compound >= 0.05`: positive
- `compound <= -0.05`: negative
- otherwise: neutral

VADER is justified here because it is designed for short English social-media-style text, is transparent to explain, and provides an interpretable indicator of tone at the community level. In this project, sentiment is used as supporting evidence rather than the main narrative result.

VADER uses lexical sentiment plus rules for things that often matter in short online comments, including punctuation, capitalization, degree modifiers, and negation. This is useful for YouTube comments because people often use capitalization, punctuation, emojis, and emphasis. 

This measure still has clear limits. VADER may misread sarcasm, football slang, irony, or context-specific meanings, especially in emotionally charged match discussions.

### NRCLex Emotion Analysis

Emotion analysis uses `NRCLex`, which is based on the NRC Emotion Lexicon or EmoLex. It scores comments across eight emotion categories:

- anger
- joy
- sadness
- fear
- surprise
- trust
- disgust
- anticipation

NRCLex uses the NRC Emotion lexicon, which follows a basic-emotion lexicon approach related to frameworks such as Plutchik's eight basic emotions. In this project, the scores are treated as approximate lexical signals of emotional tone, not as exact psychological states. 

Emotion is useful because World Cup discussions are often emotionally rich in ways that simple positive or negative polarity does not capture well. Sentiment can show whether a community is broadly more positive or negative, while emotion categories help distinguish patterns such as anger, joy, surprise, disappointment, or anticipation.

Positive and negative polarity labels from NRC are not used because VADER already handles sentiment polarity. NRCLex is also lexicon-based rather than a deep contextual model, so its emotion labels should be treated as approximate comparative signals across communities, not exact psychological states.

### Representative Comments

Representative comments are selected for manual qualitative inspection from the main communities. In practice, comments are chosen using practical criteria such as reasonable length, the presence of distinctive terms or normalized entities, and in some cases like count.

These comments are included without usernames and are used as illustrative examples rather than statistical proof. Their main role is to help validate whether the automatically suggested community labels and narrative cues are plausible when checked against real comment text.

### Outputs

The default content-analysis outputs are intentionally compact.

Default CSV outputs:

- `data/processed/community_content_summary.csv`: one row per community, including sentiment shares, emotion profiles, top terms, top normalized entities, and representative comments.
- `outputs/tables/content_analysis_summary.csv`: a compact method and run summary that records which tools were used, including `vaderSentiment`, `NRCLex`, dictionary normalization, and optional spaCy.

Default figures:

- `outputs/figures/community_sentiment_profile.png`: positive, neutral, and negative shares by main community.
- `outputs/figures/community_emotion_profile.png`: normalized emotion profile by main community.
- `outputs/figures/community_top_terms.png`: distinctive TF-IDF terms for each main community.
- `outputs/figures/community_entity_profile.png`: normalized entity mention intensity across main communities.

Optional detail tables can also be written with `--write-detail-tables`. These include detailed comment, term, entity, and representative-comment CSV outputs, but they are not default outputs because the project keeps the folder structure deliberately simple.

### Caveats

- TF-IDF terms are indicators of distinctive vocabulary, not complete topics.
- VADER and NRCLex are lexicon-based methods and may miss sarcasm, irony, slang, and football-specific context.
- Sentiment and emotion should therefore be treated as approximate supporting evidence.
- Entity extraction is not perfect, and spaCy remains a generic model that may miss domain-specific football concepts.
- Dictionary normalization depends on manually defined aliases and may not capture every relevant surface form.
- Full NEL/NED is not performed, so normalized entities are not linked to an external knowledge base.
- Representative comments are illustrative rather than statistical evidence.
- Final narrative interpretation should triangulate video metadata, community structure, TF-IDF terms, normalized entities, sentiment and emotion profiles, and representative comments.

### Insights

The communities seem to have different narrative centers. 

```
C1:
messi, congratulations, japan, best, great, norway, love, watching
```

This sounds like a more positive/general highlights or player-appreciation community. The words 'congratulations', 'best', 'great', and 'love' suggest celebratory reactions. 

```
C2:
portugal, england, ronaldo, croatia, germany, paraguay, win, penalty
```

This looks like a mixed high-profile post-match/star-player community, especially around Portugal, Ronaldo, England, Croatia, Germany, and penalty discussions. 

```
C3:
ball, croatia, touch, offside, portugal, hair, var, ronaldo
```

This is very clearly a VAR/offside controversy community. The terms 'touch', 'offside' 'VAR', 'ball', and even 'hair' suggest a very specific dispute about whether someone touched the ball or whether the offside decision was correct (Portugal vs. Croatia, decisive situation). 

```
C4:
germany, england, german, var, kane, just, foul
```

This looks like a Germany/England/VAR/foul discussion space. It is probably a controversy or post-match debate community, but we need representative comments to label it properly. 

```
C5:
scotland, clarke, fans, scottish, steve, steve clarke, manager, better
```

This one is extremely clear. It's a Scotland/Steve Clark/manager criticism or tactical blame community. 

```
C6:
foul, robbed, var, germany, right, keeper, ghana, goalkeeper
```

This is also clear: a referee/VAR injustice community. The words 'foul', 'robbed', 'VAR', 'keeper' and 'goalkeeper' suggest a controversy around a specific goalkeeping/foul incident.

So TF-IDF is already helping answer RQ2: 

```
Different communities are characterized by different narrative vocabularies: player praise, post-match debate, VAR/offside controversy, Germany/England disputes, Scotland manager criticism, and robbery/foul controversy. 
```

The sentiment chart shows that all communities are mixed, but there are differences: C1 and C5 look more positive, C3 and C6 look more negative, C2 and C4 are somewhere in the middle. VADER is supporting evidence, it helps say that controversy-oriented communities show more negative tone than celebratory/highlight-oriented communities. 

The emotion profile is not as sharply different as the TF-IDF or entity chart. The distributions look relatively similar across communities. This can happen because NRCLex is lexicon-based and many football comments contain overlapping emotional language. Words like 'great', 'trust', 'fear', 'surprise', and 'anger' may appear across many communities in different contexts. 

C3 and C6 have relatively stronger anger-related components. C1 and C5 seem more positive/trust/joy oriented. 

Emotion profiles were less sharply separated than the keyword and entity profiles, but controversy-oriented communities showed stronger anger-related signals, while highlight/player-appreciation communities showed more positive affective language. 

The main insight from NER is that the communities are not just separated by sentiment. They are separated by different narrative objects: some communities revolve around teams, others around players, and others around referee/VAR concepts. 

C1 has relatively generic entities like 'World Cup', 'Japan', 'Lionel Messi', 'Argentina'. This fits the idea that C1 is a broader highlights/general-tournament viewing community. It looks more like a mixed audience reacting to match highlights, players, and tournament moments. 

C2 has visible mentions of Portugal, Cristiano Ronaldo, Croatia, Germany, England, Penalty. This supports the earlier idea that C2 is a post-match reaction/star-player debate community. The presence of Cristiano Ronaldo, Portugal, Croatia, and Penalty suggests that this community is likely centered around high-profile match drama, especially involving portugal/ronaldo and controversial match moments. 

C3 is one of the clearest ones. It strongly features Portugal, Croatia, Offside, VAR, Referee, Penalty, Cristiano Ronaldo. This directly supports the VAR/referee controversy interpretation. The entity heatmap confirms what the TF-IDF terms already suggested. So C3 is a community organized around a concrete controversy narrative. 

C4 strongly features Germany, England, Harry Kane, VAR, Referee, Goalkeeper. This suggests that C4 is centered around a Germany/England-related controversy or post-match debate. 

C5 is less strongly in this heatmap. It might be because the most relevant C5 terms are things like Scotland, Clarke, Steve Clarke, Manager, Fans. If those are not among the top 20 entities shown, they may be present in the detailed summary but excluded from the heatmap because other entities have other higher overall counts. 

C6 shows mentions around Germany, VAR, Referee, Goalkeeper, France, Penalty and, together with the TF-IDF terms: foul, robbed, var, germany, keeper, goalkeeper; this supports a ref/goalkeeper/foul controversy narrative. 

The representative comments strongly support the interpretations suggested by TF-IDF, entities, sentiment, and emotions. 

C1 has positive terms like: messi, congratulations, japan, best, great. And the representative comments are very celebratory. They mention players scoring, "what a day of football", "beautiful football", "huge love", and congratulations to Argentina/Messi. 

So C1 is not just large; it seems like a broad, positive, match-highlightss/tournament-appreciation community. 

C2 has: portugal, england, ronaldo, croatia, germany. The comments talk about FIFA favoritism, Portugal, Ronaldo, Croatia being robbed, penalties, and Germany squad debates. So C2 looks like a post-match debate community, with a mix of star-player discussion and controversy/fairness arguments. 

C3 is extremely clear. Terms and entities include: ball, croatia, touch, offside, portugal -- Croatia, Portugal, FIFA, Offside, Referee. The representative comments talk about whether the ball was touched, VAR, the referee, Croatia deserving a penalty, and claims that the decision was unfair or rigged. 

C4 has: germany, england, german, var, like -- Germany, England, VAR, Goalkeeper, Harry Kane. The comments include arguments about whether a Germany incident was a foul, criticism of VAR, Germany losing identity, and comparisons to Kane/England decisions. So C4 seems like a Germany/England-related controversy and post-match debate community. 

C5 is very clear: scotland, clarke, fans, scottish, steve -- Scotland, Steve Clarke, Brazil, World Cup, Haiti. The comments discuss Steve Clarke, Scotland's tactics, manager responsibility, and Scotland fans. This looks like a tactical blame/national-team criticism community. 

C6 has: foul, robbed, var, germany, right -- Germany, VAR, Goalkeeper, Foul, Portugal. The comments talk about missed penalties, VAR, corrupt/incompetent officials, the Paraguay goalkeeper, simulation, and Germany being robbed. So C6 is another controversy community, probably focused on a diffrent referee/goalkeeper/foul narrative than C3. 

The communities are not just different in tone, they are organized around different narrative anchors. 

Representative comments were used as a qualitative validation step. 

## Later Analysis Plan

- user-video participation network
- user-user co-commenting network
- Louvain community detection
- centrality and bridge analysis
- VADER sentiment
- emotion analysis
- TF-IDF keywords
- named entities
- manual narrative labels

## Caveats

- The dataset is not a random sample of all World Cup YouTube discourse.
- YouTube search and relevance ranking are platform-mediated.
- The dataset is English-focused.
- Automatic filters and context labels are transparent but imperfect.
- Top-level comments do not capture full conversation threads.
- Co-commenting later means shared participation, not direct interaction.
- If categories or context types are imbalanced, later analysis should rely on normalized measures rather than raw counts alone.
