"""Correlate finalized Maxun history messages with durable turns.

Revision ID: 008
Revises: 007
Create Date: 2026-08-21
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "008"
down_revision: str | None = "007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    columns = {column["name"] for column in inspector.get_columns("messages")}
    if "maxun_turn_record_id" not in columns:
        op.add_column(
            "messages",
            sa.Column("maxun_turn_record_id", sa.String(), nullable=True),
        )
    indexes = {index["name"] for index in inspector.get_indexes("messages")}
    if "idx_messages_maxun_turn_record" not in indexes:
        op.create_index(
            "idx_messages_maxun_turn_record",
            "messages",
            ["maxun_turn_record_id"],
        )


def downgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    indexes = {index["name"] for index in inspector.get_indexes("messages")}
    if "idx_messages_maxun_turn_record" in indexes:
        op.drop_index("idx_messages_maxun_turn_record", table_name="messages")
    columns = {column["name"] for column in inspector.get_columns("messages")}
    if "maxun_turn_record_id" in columns:
        op.drop_column("messages", "maxun_turn_record_id")
