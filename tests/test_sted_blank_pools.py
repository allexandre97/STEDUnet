from fibras.sted_blank_pools import BLANK_POOL_ROLES, assign_blank_pools, validate_blank_pools


def rows():
    out = []
    for group_i in range(4):
        for series in range(2):
            out.append(
                {
                    "stable_image_id": f"blank_{group_i}_{series}",
                    "source_root_id": "sted_blank_data",
                    "relative_path": f"g{group_i}_{series}.tif",
                    "source_sha256": f"sha{group_i}_{series}",
                    "acquisition_group": f"group_{group_i}",
                    "blank_status": "expert_validated",
                    "validation_source": "human_expert_review",
                    "validator_role": "STED expert",
                    "validation_date": "not_recorded",
                    "notes": "",
                }
            )
    return out


def test_blank_pool_assignment_keeps_acquisition_groups_together():
    assigned = assign_blank_pools(rows())
    assert validate_blank_pools(assigned) == []
    role_by_group = {}
    for row in assigned:
        role_by_group.setdefault(row["acquisition_group"], row["blank_pool_role"])
        assert role_by_group[row["acquisition_group"]] == row["blank_pool_role"]
        assert row["human_approved"] == "false"
    assert set(role_by_group.values()) == set(BLANK_POOL_ROLES)
