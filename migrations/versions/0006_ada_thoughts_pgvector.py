"""Create/upgrade ada_thoughts with a global_cortex pgvector column.

No prior migration creates `ada_thoughts` (on SQLite it is born via
metadata.create_all; alembic only runs on Postgres). So on a fresh Postgres
this migration must CREATE the table, then add the pgvector column + HNSW index
so Ada's cognitive memory persists and reconstructs without re-encoding.

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
    conn = op.get_bind()
    insp = sa.inspect(conn)
    is_pg = conn.dialect.name == "postgresql"

    if "ada_thoughts" not in insp.get_table_names():
        op.create_table(
            "ada_thoughts",
            sa.Column("thought_id", sa.String(36), primary_key=True),
            sa.Column("content", sa.Text(), nullable=False),
            sa.Column("speaker", sa.String(50), nullable=False, server_default="incoming"),
            sa.Column("glyph_data", sa.Text(), nullable=False),
            sa.Column("content_vector", sa.Text(), nullable=True),
            sa.Column("global_cortex", sa.Text(), nullable=True),
            sa.Column("strength", sa.Float(), nullable=False, server_default="1.0"),
            sa.Column("access_count", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("created_at", sa.Float(), nullable=False),
            sa.Column("last_accessed", sa.Float(), nullable=False),
            sa.Column("extra_data", sa.JSON(), nullable=True),
            sa.Column("archived", sa.Integer(), nullable=False, server_default="0"),
        )
        op.create_index("idx_thought_strength", "ada_thoughts", [sa.text("strength DESC")])
        op.create_index("idx_thought_accessed", "ada_thoughts", [sa.text("last_accessed DESC")])
        op.create_index("idx_thought_archived", "ada_thoughts", ["archived"])
    else:
        cols = {c["name"] for c in insp.get_columns("ada_thoughts")}
        if "global_cortex" not in cols:
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
    conn = op.get_bind()
    if conn.dialect.name == "postgresql":
        op.execute("DROP INDEX IF EXISTS idx_thought_cortex")
    op.drop_column("ada_thoughts", "global_cortex")
