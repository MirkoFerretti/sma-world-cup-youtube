"""Helpers for entity extraction and football-specific matching."""


def load_spacy_model(model_name: str = "en_core_web_sm"):
    """Load the spaCy model used for entity extraction."""
    # TODO: Load a spaCy pipeline and handle missing model setup clearly.
    pass


def extract_named_entities(comments_df, text_column: str = "comment_text"):
    """Extract named entities from comments with spaCy."""
    # TODO: Run spaCy NER over comments and return structured entity data.
    pass


def match_custom_football_entities(comments_df, dictionary_path=None):
    """Match optional football-specific entities from a custom dictionary."""
    # TODO: Add rule-based matching for teams, players, or competitions if needed.
    pass


def combine_entity_sources(spacy_entities_df, custom_entities_df=None):
    """Merge general NER output with optional custom entity matches."""
    # TODO: Combine entity sources into one consistent dataset.
    pass
