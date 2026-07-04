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
