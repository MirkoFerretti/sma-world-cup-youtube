# Project Notes

## Objective

This project studies narrative communities in FIFA World Cup 2026 YouTube comments. The aim is not a basic sentiment result such as "fans of winning teams are happier." Instead, the project asks how YouTube users cluster around recurring narratives such as matches, underdogs, controversies, rivalries, tactical blame, national pride, and star-player debates.

## Research Questions

1. Do World Cup 2026 YouTube discussions form participation communities around specific teams, matches, or broader narratives?
2. Which narratives dominate detected communities, and how are they expressed through entities, keywords, sentiment, and emotions?
3. Which users, videos, matches, teams, players, or controversy-related entities bridge different narrative communities?

## Project Status / Reproducibility Note

The collected YouTube dataset is now frozen for this project. The later scripts operate on this frozen dataset and its derived outputs rather than on a live, constantly changing YouTube sample.

Do not rerun:

- `python src/collect_youtube.py discover`
- `python src/collect_youtube.py filter-eligible`
- `python src/collect_youtube.py collect`

unless you intentionally want to restart the collection process. YouTube search results, metadata, and comments can change over time, so rerunning collection would likely change the sample and would require rerunning preprocessing, network construction, community detection, content analysis, and bridge analysis.

The project should now focus on interpretation, report writing, and presentation using the existing frozen dataset and analysis outputs.

## Data Collection Process

The collection pipeline has three steps.

1. Candidate discovery: the script searches YouTube using predefined World Cup event and narrative query groups from `config/youtube_queries.json` rather than one broad generic query.
2. Automatic eligibility filtering: discovered videos are deduplicated by `video_id`, enriched with basic YouTube metadata, and written into one compact sampling file with either `eligible` or `excluded` status.
3. Comment collection: the script collects capped top-level comments only from rows marked `eligible`.

This design is simple enough for a short presentation, but still reproducible and defensible. It captures multiple World Cup discussion spaces instead of depending on one narrow search phrase.

The project does not try to collect every World Cup video. It uses query groups to build a balanced corpus of different narrative contexts: highlights, reactions, controversies, upsets, star-player debates, and tactical/rivalry discussions. 

The project studies narrative communities, so the dataset needs to include different kinds of situations where narratives are likely to emerge: match events, post-match reactions, controversies, underdog stories, star-player debate, tactical criticism, and blame. 

Match highlights capture broad, high-visibility match discussions. These videos are useful because they attract general tournament audiences and give us a baseline of participation around concrete matches. 

Post match reactions capture pundit, fan, and media reactions after matches, which can be important for interpretation, praise, criticism, blame, and debate. 

Upsets underdogs capture unexpected results, smaller teams, and surprise performances. They are useful because upset narratives generate surprise, national pride, disappointment, and cross-community attention. 

VAR-referee controversies capture disputed calld, VAR decisions, penalties, offside decisions, disallowed goals, and criticism. Controversy often creates emotionally intense discussion and can attract users from different narrative communities. 

Start player-related queries capture debates around major players and other star figures, which can often cut across teams, countries, and fan communities. 

The blame and tactical query captures tactical analysis, manager criticism, rivalry narratives, national-team failrue, and blame attribution. 

The design is narrative-oriented, it is meant to cover several types of World Cup discussion spaces rather than only one type of content. This makes the process systematic and transparent, but not statistically representative of all YouTube discussion around the World Cup. YouTube search is platform-mediated, so the discovered videos depend on YouTube's ranking algorithm, metadata, timing, and exact query wording. 

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

Raw comments are deduplicated by `comment_id`. The script preserves `original_text` for possible qualitative inspection and creates a lightly cleaned `clean_text` by decoding HTML entities, removing URLs, replacing line breaks, collapsing whitespace, and stripping the final text. Punctuation, capitalization, and negation are kept because they will be used for sentiment analysis using VADER. 

Very short comments are removed with `min_chars = 15` because they usually add little semantic information. Likely non-English comments are also removed because the later text-analysis tools are English-oriented. VADER, English TF-IDF stopword handling, spaCy's English model, and NRCLex are all more appropriate for english text. This language filter is a rule-based heuristic, not a full language-detection model: it uses ASCII or Latin-letter ratio, common English function words, football terms, and token count. It is transparent but imperfect.

Users are pseudonymized with a stable hashed `author_id`. The same YouTube channel always maps to the same anonymous ID, which preserves repeat participation for later network analysis without using raw usernames or channel IDs in the processed dataset.

Compact video metadata is merged through `video_id`. The preprocessing outputs are `data/processed/comments_processed.csv` and `outputs/tables/preprocessing_summary.csv`.

Current preprocessing result:

- 22,594 raw comments
- 19,152 processed comments
- 16,505 processed users
- 134 processed videos
- 1,869 short comments removed
- 1,573 likely non-English comments removed
- 1,441 raw users with at least 2 videos
- 382 raw users with at least 3 videos
- 134 eligible videos with comments
- 0 eligible videos without comments

The repeat-user count is sufficient to proceed with the user-video and user-user co-commenting network construction.

The preprocessing step prepared the collected YouTube comments for network and content analysis while keeping the text as close as possible to the original comments. Before applying any cleaning, we run basic checks on the collected dataset, including the number of videos and comments, duplicate IDs, user overlap across videos, category and context balance, and video-level language metadata. 

Preprocessing is intentionally light. Cleaning only removes obvious noise: HTML entities are encoded, URLs are removed, line breaks are replaced, repeated whitespace is collapsed, and leading/trailing whitespace is stripped. 

We don't remove punctuation, stop words, lemmatize words, or lowercase the text because later methods such as VADER can use punctuation, capitalization, negation, and emphasis as meaningful signals. 



## Network Construction

### Objective

The purpose of the network step is to move from individual comments to participation structures. The raw data is comment-level: an anonymized user commented on a video. Network construction makes that participation structure explicit and prepares the data for community detection.

### User-Video Network

The first structure is a user-video network. This is a bipartite network, meaning it contains two different types of nodes: users and videos. An edge means that an anonymized user commented on a specific video.

In simple terms, a video is treated as a discussion space, and a user-video edge means that the user entered that discussion space by commenting. This is the direct participation structure in the data before users are projected onto one another.

### User-User Co-Commenting Network

The second structure is a user-user co-commenting network created by projecting the user-video network. Two users are connected if they commented on at least one same video. The edge weight is the number of videos they both commented on:

- weight = 1 means they shared one video discussion
- weight = 2 means they shared two video discussions
- weight = 3 means they shared three video discussions

This is shared participation, not direct interaction. The network captures repeated presence in the same World Cup discussion spaces, not direct communication, friendship, agreement, replies, or conversation between users.

### Why This Network Is Defensible

This representation is the most defensible choice for the project because the dataset does not include follower or friendship data, and the collection step captured top-level comments rather than full reply conversations. Co-commenting therefore fits the research goal well: the project studies shared participation in World Cup narrative spaces rather than interpersonal ties.

### Edge Weight Thresholds

Edge-weight thresholds are used to filter edges, not users directly. A user who comments on two videos does not automatically remain in the weight >= 2 network. To remain in that thresholded graph, the user must share at least two of the same videos with at least one other user. This is why the weight >= 2 graph contains far fewer users than the broader preflight count of repeat commenters.

The threshold comparison suggests three different trade-offs:

- weight >= 1 keeps all one-off overlaps. It retained 16,505 nodes and 1,395,315 edges, formed one giant component, and had an average degree of about 169 with a median degree of 171. This suggests that the graph is dominated by weak one-off overlaps from popular videos.
- weight >= 2 requires repeated co-participation. It retained 931 users and 4,133 edges, with 58 connected components and a largest component of 721 users, or 77.44% of the graph. The average degree was about 8.88, the median degree was 5, and the maximum degree was 228. This produces a much cleaner graph while still preserving enough structure for interpretation.
- weight >= 3 is stricter. It retained only 160 users and 687 edges, with 12 connected components and a largest component of 107 users, or 66.87% of the graph. This makes it useful as a robustness check, but too restrictive for the main analysis.

### Resulting Choice

Weight >= 2 was selected as the main graph for community detection because it balances noise reduction, repeated co-participation, and sufficient network size. It removes much of the weak overlap that dominates the weight >= 1 graph without shrinking the network as severely as the weight >= 3 alternative.

### Outputs

Outputs from this step:

- `data/processed/user_video_edges.csv`
- `data/processed/user_network_edges.csv`
- `data/processed/user_network_nodes.csv`
- `outputs/tables/network_summary.csv`
- `outputs/figures/user_participation_distribution.png`
- `outputs/figures/network_threshold_diagnostics.png`

### Caveats

- Co-commenting is not direct interaction.
- High degree does not necessarily mean influence.
- Large popular videos can create many weak edges.
- Network results depend on the sampled videos and the comment cap.
- The threshold choice is methodological and should be justified rather than treated as objectively perfect.

## Community Detection

### Objective

After building the user-user co-commenting network, community detection identifies groups of users who are more densely connected to one another than to the rest of the graph. In this project, those groups are interpreted as participation communities. The goal is to identify clusters of users who repeatedly appear in similar World Cup video discussion spaces.

### Methodology

The main graph uses edge weight >= 2. Louvain community detection is applied to this weighted graph. Louvain is a community-detection method that searches for partitions in which connections are denser within groups than between groups.

Edge weights are kept in the model so that stronger repeated co-participation matters more than weaker links. Communities are labelled `C1`, `C2`, `C3`, and so on by size for readability, where `C1` is the largest community. These are display labels, not inherent community names. The narrative meaning of each community is interpreted later using video metadata and content analysis.

### Interpretation

A community in this project means a group of anonymized users with similar participation patterns. It does not mean that users are friends, agree with one another, reply to one another, or support the same team. It means that they repeatedly participate in overlapping World Cup discussion spaces.

The network gives the structure; video metadata and comment content give the meaning.

### Results

The main graph contains 931 users and 4,133 repeated co-commenting edges. Louvain detects 71 communities with modularity = 0.5505. The largest community contains 273 users and the median community size is 2 users. There are 64 communities with fewer than 10 users, while 6 communities have at least 20 users. Those 6 larger communities cover 76.37% of users in the main graph.

Modularity is a summary measure of how clearly the network separates into communities. A value of 0.5505 suggests a meaningful community structure rather than a completely random grouping, although it should not be treated as proof that the partition is uniquely correct.

### Long-Tail Community Structure

The community size distribution has a clear long-tail pattern. A few large communities contain most of the interpretable structure, while many small communities are peripheral. This supports focusing the later content analysis on the six communities with at least 20 users: `C1` with 273 users, `C2` with 194, `C3` with 141, `C4` with 46, `C5` with 35, and `C6` with 22.

The figure `community_size_long_tail.png` makes this structure visible and helps show why the large communities receive the most detailed later interpretation.

### Community Profiles From Video Metadata

Before any text analysis, the communities were profiled using video metadata, including primary sampling category, video context type, top videos, and top channels. These metadata profiles already suggest that different communities are associated with different kinds of discussion spaces.

- `C1` is strongly associated with match highlights and appears to represent a broad match-highlights or general tournament viewing community.
- `C2` is associated with post-match reactions and star-player legacy contexts, suggesting a post-match debate or star-player discussion space.
- `C3` is strongly associated with VAR or referee controversy, suggesting a controversy-oriented community.
- `C4` appears more mixed, with VAR or referee controversy, post-match reaction, and other World Cup discussion elements.
- `C5` is strongly associated with tactical analysis and tactical blame or rivalry contexts, suggesting a manager or tactical criticism community.
- `C6` is also heavily associated with VAR or referee controversy, suggesting another controversy-oriented community, possibly linked to a different controversy or match context.

These are preliminary interpretations based on video metadata. Later content analysis is used to validate and refine them.

### Network Visualizations

The full community network overview shows the largest connected component of the weight >= 2 graph. Nodes are users, edges are repeated co-commenting links, and colors show the main communities. Small communities are shown in gray. The highlighted `C1` to `C5` figures show where each large community sits inside the same overall layout.

The layout is force-directed and illustrative. Spatial positions should not be overinterpreted as exact distances. The value of these visualizations is that they show separation and overlap between participation communities, not that they provide a literal map of social distance.

### Insights

The community structure is not homogeneous. The network separates into several large participation communities plus many smaller peripheral groups. This already suggests that World Cup YouTube discussion is structured rather than fully diffuse.

The six larger communities contain most of the interpretable structure and are therefore the main focus for later content analysis.

The preliminary video profiles already suggest different discussion spaces, including match highlights, post-match or star-player debate, VAR or referee controversy, tactical blame, and mixed national-team debate. These metadata-based patterns provide an early answer to RQ1 before the comments themselves are analyzed in depth.

The network visualization also suggests that some communities are more compact, while others appear positioned closer to other groups. This does not prove bridging by itself, but it prepares the ground for the later bridge analysis.

Community detection therefore helps answer RQ1 by showing that World Cup YouTube discussions form detectable participation communities based on repeated co-commenting patterns.

### Outputs

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

### Caveats

- Communities are participation communities, not friendship groups.
- Community detection depends on the edge threshold.
- Louvain is heuristic, so the result should be interpreted as useful structure rather than the only possible partition.
- Small communities are not deeply interpreted because they contain limited evidence.
- Video metadata profiles are preliminary; content analysis is needed for narrative labels.
- Network layout positions are illustrative and should not be overinterpreted.

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

## Bridge Analysis

### Objective

Bridge analysis addresses RQ3: "Which users, videos, matches, teams, players, or controversy-related entities bridge different narrative communities?" After identifying participation communities and profiling their content, the next step is to look for users, videos, and entities that connect multiple communities. This helps explain not only how discussion spaces separate, but also what links them across the sampled World Cup YouTube corpus.

### Bridge Users

Bridge users are anonymized users who occupy connecting positions in the main co-commenting network. This analysis uses the same main graph as community detection, with edge weight >= 2, so the bridge results stay aligned with the threshold used for the main participation structure.

The main metric is betweenness centrality. Betweenness centrality is high when a node lies on many shortest paths between other nodes in the graph. In this project, high betweenness suggests cross-community participation or a structurally connecting role between different parts of the repeated co-commenting network.

This does not mean real-world influence, friendship, or opinion leadership. The users remain pseudonymized, and the metric should be interpreted as a structural indicator inside this sampled participation network rather than as evidence of personal influence.

### Bridge Videos

Bridge videos are videos that attract users from multiple communities. For each video, the script computes the distribution of participating users by community and then summarizes how mixed that participation is.

Videos with higher community diversity and enough participating users receive higher bridge scores. This is important because videos function as shared discussion spaces in the project design. A video can connect narratives even when users do not directly reply to one another.

### Bridge Entities

Bridge entities are normalized entities that appear across multiple communities. These may include teams, players, countries, or controversy concepts. Examples in this project could include `VAR`, `Referee`, `Penalty`, `Germany`, `Portugal`, `Croatia`, or `Cristiano Ronaldo`.

Generic entities such as `World Cup` and `FIFA` may also appear broadly. They are still kept in the data, but they should be interpreted differently from more specific narrative bridges. Broad tournament entities indicate general tournament reference points, while more specific bridge entities help show which actors, teams, or controversy concepts travel across community boundaries.

### Metrics

Bridge users are summarized with betweenness centrality together with degree and weighted degree on the thresholded co-commenting graph.

Bridge videos are summarized with the number of communities represented, the number of main communities represented, the dominant community share, community entropy, and a bridge-video score. Entropy is used here as a simple diversity measure: high entropy means participation is spread more evenly across communities, while low entropy means one community dominates. The bridge-video score combines that diversity signal with the total number of participating community users so that a bridge video should be both mixed and substantively attended.

bridge_video_score = normalized community entropy * log1p(total community users on video)

A video gets a high bridge score if its commenters come from several different communities and if enough community users commented on it. So it rewards both diversity across communities and enough participation volume. 

For each video, look at the communities of the users who commented on that video:

```
Video A:
C1: 30 users
C2: 10 users
C3: 10 users
Total = 50 community users
```

Then it computes the distribution:

```
C1 = 60%
C2 = 20%
C3 = 20%
```

A video is more bridging when the users are spread across several communities instead of coming almost entirely from one community. So we use the community entropy, which is just a diversity score:

```
C1 = 95%
C2 = 5%
```

This means one community dominates. Not much of a bridge.

High entropy would be:

```
C1 = 35%
C2 = 30%
C3 = 20%
C4 = 15%
```

This means several communities are participating, which is more bridge-like. We use normalized entropy, so it's between 0 and 1, where 0 means that one community dominates completely, and 1 means that participation is evenly spread across represented communities. 

The log(1 + total community users on the video) is the second metric. A video with 3 users from 3 communities is diverse, but not very important. A video with 40 users from several communities is more meaningful. But we use the log instead of raw user count cause, otherwise, large videos would dominate too much. 

And the final formula is:

bridge_video_score = community_entropy * log1p(total_community_users)

A problem is that some top bridge videos are dominated by small communities. For example, the top bridge video is the "REACTION to Carlos Queiroz and Jose Mourinho criticizing the 48-team World Cup format", which has 8 communities represented, and the main community represented is 1. So technically it is a bridge video because it attracts many different communities, but most of those are small peripheral communities, not the main C1-C6 communities. It's not wrong, but for the report it may be less interesting than videos that connect multiple main communities (i would keep it though, we'll just include other top bridges too for the report). 

Other relevant ones are:

```
Germany dumped out by Paraguay → 5 main communities
Germany and Netherlands knocked out → 4 main communities
VAR is trash / Germany robbed → 3 main communities
Klinsmann slams Germany elimination → 3 main communities
Croatia disallowed goal vs Portugal → 2 main communities
Panama v England post-match analysis → 2 main communities
```

Bridge entities are summarized with the number of communities in which they appear, the number of main communities in which they appear, community entropy when a spread measure can be computed, and a bridge-entity score. When detailed entity counts are available, the score can use both diversity and mention volume. When the compact default community summary is used, the score is interpreted more conservatively as cross-community presence.

The bridge entity score is more conservative than the video score: 

bridge_entity_method = compact top_entities presence; score uses detailed mention counts when available, otherwise main-community presence

It's basically: "in how many main communities does this entity appear?"

### Outputs

- `data/processed/bridge_users.csv`
- `data/processed/bridge_videos.csv`
- `data/processed/bridge_entities.csv`
- `outputs/tables/bridge_analysis_summary.csv`
- `outputs/figures/bridge_videos.png`
- `outputs/figures/bridge_entities.png`

### Insights

The bridge videos suggest something very interesting: cross-community bridging is mainly created by controversy, elimination, and pundit-reaction videos. The strongest examples are the ones for the Germany/Paraguay elimination, like "TRAVESTY! Germany dumped out by Paraguay", which connects communities:

```
C4:13
C2:4
C5:4
C1:3
C6:1
plus smaller communities
```

Germany's elimination appears to be a cross-community discussion event, attracting not just the Germany/VAR community but also the post-match debate, tactical blame, highlights, and controversy-oriented users. 

"VAR is TRASH. Germany robbed!!!" connects C6:11, C4:7, C2:4 and is a very clean bridge between controversy communities. VAR/referee controversy around Germany acts as a connection between multiple narrative communities. 

"HEARTBREAK! Croatia fans react to late disallowed goal vs. Portugal" connects C2:30 and C3:10. The Portugal-Croatia disallowed goal links the post-match/star-player debate community with the VAR/offside controversy community. 

Videos like:

```
Panama v England post-match analysis
England v DR Congo post-match
Simon Jordan attacks England players
```

connect multiple communities, usually C2/C4 and some smaller groups. Pundit-reaction videos about England act as shared discussion spaces across otherwise separate communities. 

So videos become bridges when they combine high-stakes matches, controversy, elimination, or pundit debate. 

<br>

Germany and Portugal are the strongest bridge entities, they appear in 3 main communities each. This makes sense because they are both tied to several narratives:

```
Germany → elimination, VAR, England/Germany debates, Paraguay upset
Portugal → Ronaldo, Croatia, penalty/offside controversy
```

VAR and Goalkeeper are specific controversy bridges that appear in 2 main communities. 

World Cup and FIFA appear across communities, but they are generic. They indicate shared context, but they are less narratively specific than Germany, Portugal, VAR, or Goalkeepers. 

Cristiano Ronaldo appears in many communities overall, but only 1 main community, so he seems like a community-specific narrative anchor, especially for C2, rather than a bridge across the main communities. 

Not every frequent entity is a bridge. Some entities define one main community strongly rather than connecting several communities. Same for Lionel Messi, Harry Kane, Steve Clarke, and Offside. 

The top bridge user has 

```
community = C3
degree = 44
weighted_degree = 94
betweenness = 0.2519
videos_commented = 9
```

Which means that this anonymized user sits in a structurally connecting position inside the co-commenting graph. Bridge users identify cross-community participation patterns but because users are anonymized and we do not observe direct interaction, they should be interpreted as structural connectors only. 

<br>

Bridge users are a small set of anonymized users that occupy high-betweenness positions in the repeated co-commenting network, meaning they structurally connect otherwise separated participation communities. 

The strongest interpretable bridge videos are reaction/controversy videos, especially:

```
Germany dumped out by Paraguay
Germany robbed / VAR is trash
Croatia disallowed goal vs Portugal
England post-match analysis/reaction videos
```

The strongest bridge entities include 

```
Germany
Portugal
England
Croatia
VAR
Goalkeeper
World Cup
```

but the interpretation differs: 

```
Germany / Portugal / England / Croatia = team or national narrative bridges
VAR / Goalkeeper / Referee / Penalty = controversy bridges
World Cup / FIFA = broad contextual bridges
Cristiano Ronaldo / Messi / Steve Clarke = more community-specific anchors
```

Unlike content analysis, where small communities were not deeply interpreted because they contain limited semantic evidence, bridge analysis considers all detected communities. This is because bridge objects are interesting precisely when they connect the core communities to the long tail of smaller peripheral groups. 

Bridge analysis was applied to all detected communities, not only the six large ones. This is useful because bridges can connect major narrative communities to each other, but they can also connect the core discussion space to smaller peripheral clusters. The top bridge videos therefore reveal two patterns: some videos, such as Germany elimination or VAR controversy videos, connect multiple main communities; others connect one or two main communities with many small communities, acting as broader discussion hubs in the long tail of the network.

### Caveats

- Bridge users are structural connectors, not necessarily influencers.
- Bridge videos represent shared participation spaces, not direct user interaction.
- Bridge entities show cross-community mentions, not shared stance or agreement.
- Results depend on the sampled YouTube videos, the English-focused comment corpus, entity normalization choices, and the chosen weight >= 2 network threshold.
- Bridge analysis should be interpreted together with community detection and content analysis rather than as a standalone final explanation.

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