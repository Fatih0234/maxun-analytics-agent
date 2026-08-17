"""Add durable resumable Maxun turn events and lifecycle state.

Revision ID: 007
Revises: 006
Create Date: 2026-08-17
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "007"
down_revision: str | None = "006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "maxun_turns",
        sa.Column("attempt", sa.Integer(), nullable=False, server_default="1"),
    )
    op.add_column(
        "maxun_turns",
        sa.Column("cancel_requested_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "maxun_turns",
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "maxun_turns",
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "maxun_turns",
        sa.Column("next_event_sequence", sa.Integer(), nullable=False, server_default="1"),
    )
    op.create_index(
        "idx_maxun_turns_processing_updated",
        "maxun_turns",
        ["status", "updated_at"],
    )
    op.create_table(
        "maxun_turn_events",
        sa.Column("id", sa.String(255), nullable=False),
        sa.Column("turn_record_id", sa.String(255), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("event_type", sa.String(64), nullable=False),
        sa.Column("payload", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["turn_record_id"], ["maxun_turns.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "turn_record_id",
            "sequence",
            name="uq_maxun_turn_event_sequence",
        ),
    )
    op.create_index(
        "idx_maxun_turn_events_replay",
        "maxun_turn_events",
        ["turn_record_id", "sequence"],
    )


def downgrade() -> None:
    op.drop_index("idx_maxun_turn_events_replay", table_name="maxun_turn_events")
    op.drop_table("maxun_turn_events")
    op.drop_index("idx_maxun_turns_processing_updated", table_name="maxun_turns")
    op.drop_column("maxun_turns", "next_event_sequence")
    op.drop_column("maxun_turns", "finished_at")
    op.drop_column("maxun_turns", "started_at")
    op.drop_column("maxun_turns", "cancel_requested_at")
    op.drop_column("maxun_turns", "attempt")
