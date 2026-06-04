"""Add global_cortex pgvector column + HNSW index to ada_thoughts.

Lets Ada's cognitive memory persist the global cortex as a first-class vector
(reconstructed on load without re-encoding) and recall via native pgvector
cosine at scale. PostgreSQL-only specifics (vector type + HNSW) are guarded by
dialect; on SQLite the column is plain text and this is a no-op-ish add.

Revision ID: 0006
Revises: 0005
"""

from alembic import op
import sqlalchemy as sa

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    is_pg = bind.dialect.name == "postgresql"

    # Add the column (plain text first so it works on every dialect).
    op.add_column("ada_thoughts", sa.Column("global_cortex", sa.Text(), nullable=True))

    if is_pg:
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")
        op.execute(
            "ALTER TABLE ada_thoughts ALTER COLUMN global_cortex "
            "TYPE vector(2048) USING global_cortex::vector(2048)"
        )
        op.execute(
            "CREATE INDEX IF NOT EXISTS idx_thought_cortex ON ada_thoughts "
            "USING hnsw (global_cortex vector_cosine_ops) "
            "WITH (m = 16, ef_construction = 64)"
        )


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("DROP INDEX IF EXISTS idx_thought_cortex")
    op.drop_column("ada_thoughts", "global_cortex")
