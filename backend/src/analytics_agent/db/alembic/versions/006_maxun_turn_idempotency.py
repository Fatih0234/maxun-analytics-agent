"""Persist Maxun turn idempotency and replay results.

Revision ID: 006
Revises: 005
Create Date: 2026-08-17
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "006"
down_revision: str | None = "005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "maxun_turns",
        sa.Column("id", sa.String(255), nullable=False),
        sa.Column("conversation_id", sa.String(255), nullable=False),
        sa.Column("maxun_turn_id", sa.String(255), nullable=False),
        sa.Column("request_digest", sa.String(64), nullable=False),
        sa.Column("status", sa.String(32), nullable=False, server_default="processing"),
        sa.Column("result_json", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["conversation_id"], ["conversations.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "conversation_id",
            "maxun_turn_id",
            name="uq_maxun_turn_conversation_request",
        ),
    )
    op.create_index(
        "idx_maxun_turns_conversation_updated",
        "maxun_turns",
        ["conversation_id", "updated_at"],
    )


def downgrade() -> None:
    op.drop_index("idx_maxun_turns_conversation_updated", table_name="maxun_turns")
    op.drop_table("maxun_turns")
