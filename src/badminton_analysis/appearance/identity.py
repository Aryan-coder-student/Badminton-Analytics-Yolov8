"""Resolve anonymous players from accepted local-track associations."""

from badminton_analysis.appearance.models import (
    AssociationCandidate,
    PlayerIdentity,
    TrackAssociationProfile,
)
from badminton_analysis.appearance.utils import has_temporal_overlap


def resolve_player_identities(
    profiles: list[TrackAssociationProfile],
    candidates: list[AssociationCandidate],
) -> dict[int, PlayerIdentity]:
    """Map every raw local track ID to a deterministic logical player."""
    profile_by_track_id = {profile.track_id: profile for profile in profiles}
    components_by_track_id = {
        track_id: {track_id} for track_id in profile_by_track_id
    }
    for candidate in _accepted_candidates(candidates):
        _merge_candidate_components(
            candidate,
            components_by_track_id,
            profile_by_track_id,
        )
    return _create_identity_mapping(
        components_by_track_id,
        profile_by_track_id,
    )


def propagate_local_player_ids(
    identities_by_track_id: dict[int, PlayerIdentity],
    accepted_player_ids_by_track_id: dict[int, str],
) -> dict[int, str]:
    """Label linked fragments only when their accepted IDs do not conflict."""
    resolved = accepted_player_ids_by_track_id.copy()
    for identity in set(identities_by_track_id.values()):
        accepted_ids = {
            resolved[track_id]
            for track_id in identity.local_track_ids
            if track_id in resolved
        }
        if len(accepted_ids) == 1:
            player_id = next(iter(accepted_ids))
            resolved.update(
                {track_id: player_id for track_id in identity.local_track_ids},
            )
    return resolved


def _accepted_candidates(
    candidates: list[AssociationCandidate],
) -> list[AssociationCandidate]:
    return sorted(
        (
            candidate
            for candidate in candidates
            if candidate.likely_association
            and candidate.temporal_valid
            and not candidate.temporal_overlap
        ),
        key=lambda candidate: (
            -candidate.mean_visual_similarity,
            candidate.court_distance_meters,
            candidate.frame_gap,
            candidate.predecessor_track_id,
            candidate.successor_track_id,
        ),
    )


def _merge_candidate_components(
    candidate: AssociationCandidate,
    components_by_track_id: dict[int, set[int]],
    profile_by_track_id: dict[int, TrackAssociationProfile],
) -> None:
    predecessor_tracks = components_by_track_id.get(candidate.predecessor_track_id)
    successor_tracks = components_by_track_id.get(candidate.successor_track_id)
    if predecessor_tracks is None or successor_tracks is None:
        return
    if predecessor_tracks is successor_tracks:
        return
    if _components_overlap(
        predecessor_tracks,
        successor_tracks,
        profile_by_track_id,
    ):
        return

    merged_tracks = predecessor_tracks | successor_tracks
    for track_id in merged_tracks:
        components_by_track_id[track_id] = merged_tracks


def _components_overlap(
    first_track_ids: set[int],
    second_track_ids: set[int],
    profile_by_track_id: dict[int, TrackAssociationProfile],
) -> bool:
    return any(
        has_temporal_overlap(
            profile_by_track_id[first_track_id].first_frame,
            profile_by_track_id[first_track_id].last_frame,
            profile_by_track_id[second_track_id].first_frame,
            profile_by_track_id[second_track_id].last_frame,
        )
        for first_track_id in first_track_ids
        for second_track_id in second_track_ids
    )


def _create_identity_mapping(
    components_by_track_id: dict[int, set[int]],
    profile_by_track_id: dict[int, TrackAssociationProfile],
) -> dict[int, PlayerIdentity]:
    components = {
        frozenset(component) for component in components_by_track_id.values()
    }
    ordered_components = sorted(
        components,
        key=lambda component: (
            min(profile_by_track_id[track_id].first_frame for track_id in component),
            min(component),
        ),
    )
    identities_by_track_id: dict[int, PlayerIdentity] = {}
    for player_number, track_ids in enumerate(ordered_components, start=1):
        identity = _create_player_identity(
            f"P{player_number}",
            track_ids,
            profile_by_track_id,
        )
        identities_by_track_id.update(
            {track_id: identity for track_id in track_ids},
        )
    return identities_by_track_id


def _create_player_identity(
    player_id: str,
    track_ids: frozenset[int],
    profile_by_track_id: dict[int, TrackAssociationProfile],
) -> PlayerIdentity:
    profiles = [profile_by_track_id[track_id] for track_id in track_ids]
    return PlayerIdentity(
        player_id,
        track_ids,
        min(profile.first_frame for profile in profiles),
        max(profile.last_frame for profile in profiles),
    )
