import unittest

import numpy as np

from badminton_analysis.appearance.identity import (
    propagate_local_player_ids,
    resolve_player_identities,
)
from badminton_analysis.appearance.models import (
    AssociationCandidate,
    TrackAssociationProfile,
)


def create_profile(
    track_id: int,
    first_frame: int,
    last_frame: int,
) -> TrackAssociationProfile:
    descriptor = np.array([1.0, 0.0], dtype=np.float32)
    position = np.array([0.0, 0.0], dtype=np.float32)
    return TrackAssociationProfile(
        track_id,
        first_frame,
        last_frame,
        position,
        position,
        descriptor,
        descriptor,
    )


def create_candidate(
    predecessor_track_id: int,
    successor_track_id: int,
    *,
    likely_association: bool = True,
    temporal_valid: bool = True,
    temporal_overlap: bool = False,
) -> AssociationCandidate:
    return AssociationCandidate(
        predecessor_track_id,
        successor_track_id,
        0.95,
        0.95,
        3,
        0.2,
        temporal_overlap,
        temporal_valid,
        likely_association,
    )


class PlayerIdentityTest(unittest.TestCase):
    def test_tracks_two_and_five_map_to_one_player(self) -> None:
        profiles = [create_profile(2, 10, 20), create_profile(5, 25, 35)]

        identities = resolve_player_identities(
            profiles,
            [create_candidate(2, 5)],
        )

        self.assertEqual(identities[2].player_id, "P1")
        self.assertIs(identities[2], identities[5])
        self.assertEqual(identities[2].local_track_ids, frozenset({2, 5}))

    def test_unrelated_tracks_remain_separate(self) -> None:
        profiles = [
            create_profile(1, 10, 20),
            create_profile(2, 10, 20),
            create_profile(5, 25, 35),
        ]

        identities = resolve_player_identities(
            profiles,
            [create_candidate(2, 5)],
        )

        self.assertEqual(identities[1].player_id, "P1")
        self.assertEqual(identities[2].player_id, "P2")
        self.assertEqual(identities[5].player_id, "P2")

    def test_chained_associations_resolve_to_one_player(self) -> None:
        profiles = [
            create_profile(2, 10, 20),
            create_profile(5, 25, 35),
            create_profile(8, 40, 50),
        ]

        identities = resolve_player_identities(
            profiles,
            [create_candidate(2, 5), create_candidate(5, 8)],
        )

        self.assertEqual({identities[track_id].player_id for track_id in (2, 5, 8)}, {"P1"})
        self.assertEqual(identities[2].local_track_ids, frozenset({2, 5, 8}))

    def test_rejected_and_overlapping_associations_do_not_merge(self) -> None:
        profiles = [
            create_profile(2, 10, 20),
            create_profile(5, 15, 25),
            create_profile(8, 30, 40),
        ]
        candidates = [
            create_candidate(2, 5, temporal_valid=False, temporal_overlap=True),
            create_candidate(2, 8, likely_association=False),
        ]

        identities = resolve_player_identities(profiles, candidates)

        self.assertEqual(
            {identity.local_track_ids for identity in identities.values()},
            {frozenset({2}), frozenset({5}), frozenset({8})},
        )

    def test_player_numbers_are_deterministic_by_first_appearance(self) -> None:
        profiles = [
            create_profile(4, 10, 30),
            create_profile(5, 40, 50),
            create_profile(2, 10, 30),
            create_profile(3, 10, 30),
            create_profile(1, 10, 30),
        ]

        identities = resolve_player_identities(
            profiles,
            [create_candidate(2, 5)],
        )

        self.assertEqual(identities[1].player_id, "P1")
        self.assertEqual(identities[2].player_id, "P2")
        self.assertEqual(identities[5].player_id, "P2")
        self.assertEqual(identities[3].player_id, "P3")
        self.assertEqual(identities[4].player_id, "P4")

    def test_raw_track_ids_remain_unchanged(self) -> None:
        track_two = create_profile(2, 10, 20)
        track_five = create_profile(5, 25, 35)
        candidate = create_candidate(2, 5)

        identities = resolve_player_identities(
            [track_two, track_five],
            [candidate],
        )

        self.assertEqual((track_two.track_id, track_five.track_id), (2, 5))
        self.assertEqual(
            (candidate.predecessor_track_id, candidate.successor_track_id),
            (2, 5),
        )
        self.assertEqual(set(identities), {2, 5})

    def test_accepted_id_propagates_only_to_linked_fragment(self) -> None:
        identities = resolve_player_identities(
            [
                create_profile(2, 10, 20),
                create_profile(5, 25, 35),
                create_profile(3, 10, 35),
            ],
            [create_candidate(2, 5)],
        )
        accepted = {2: "P2"}

        resolved = propagate_local_player_ids(identities, accepted)

        self.assertEqual(resolved, {2: "P2", 5: "P2"})
        self.assertEqual(accepted, {2: "P2"})

    def test_conflicting_accepted_ids_are_not_propagated(self) -> None:
        identities = resolve_player_identities(
            [
                create_profile(2, 10, 20),
                create_profile(5, 25, 35),
                create_profile(8, 40, 50),
            ],
            [create_candidate(2, 5), create_candidate(5, 8)],
        )

        resolved = propagate_local_player_ids(identities, {2: "P2", 5: "P3"})

        self.assertEqual(resolved, {2: "P2", 5: "P3"})

    def test_unaccepted_local_group_remains_unresolved(self) -> None:
        identities = resolve_player_identities(
            [create_profile(2, 10, 20), create_profile(5, 25, 35)],
            [create_candidate(2, 5)],
        )

        self.assertEqual(propagate_local_player_ids(identities, {}), {})


if __name__ == "__main__":
    unittest.main()
