"""Helpers for building project network structures."""


def build_user_video_bipartite_graph(comments_df):
    """Build a bipartite graph linking users to videos."""
    # TODO: Create a user-video participation graph from cleaned comments.
    pass


def build_user_user_projection(comments_df, minimum_edge_weight: int = 1):
    """Project shared video participation into a user-user network."""
    # TODO: Build a co-commenting network with configurable edge thresholds.
    pass


def build_network_a(comments_df):
    """Build Network A with the lower co-commenting threshold."""
    # TODO: Keep users with at least two videos and edges with weight >= 1.
    pass


def build_network_b(comments_df):
    """Build Network B with the stricter co-commenting threshold."""
    # TODO: Keep the same user pool, require weight >= 2, and drop isolates.
    pass


def build_entity_cooccurrence_graph(entity_df):
    """Build an entity co-occurrence graph from extracted entities."""
    # TODO: Connect entities that appear together within the same comment or unit.
    pass
