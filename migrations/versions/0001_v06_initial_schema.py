"""Create v0.6 scoring, case, audit, and feedback tables.

Revision ID: 0001_v06
Revises: None
Create Date: 2026-08-01
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001_v06"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    json_type = postgresql.JSONB(astext_type=sa.Text())
    op.create_table(
        "transactions",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("external_id", sa.String(length=128), nullable=False),
        sa.Column("idempotency_key", sa.String(length=256), nullable=False),
        sa.Column("request_hash", sa.String(length=64), nullable=False),
        sa.Column("step", sa.Integer(), nullable=False),
        sa.Column("transaction_type", sa.String(length=16), nullable=False),
        sa.Column("amount", sa.Float(), nullable=False),
        sa.Column("origin_account_hash", sa.String(length=64), nullable=False),
        sa.Column("destination_account_hash", sa.String(length=64), nullable=False),
        sa.Column("feature_version", sa.String(length=32), nullable=False),
        sa.Column("feature_snapshot", json_type, nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "received_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("external_id", name="uq_transactions_external_id"),
        sa.UniqueConstraint(
            "idempotency_key", name="uq_transactions_idempotency_key"
        ),
    )
    op.create_index(
        "ix_transactions_received_at", "transactions", ["received_at"]
    )

    op.create_table(
        "predictions",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("transaction_id", sa.String(length=36), nullable=False),
        sa.Column("risk_probability", sa.Float(), nullable=False),
        sa.Column("risk_points", sa.Integer(), nullable=False),
        sa.Column("decision", sa.String(length=16), nullable=False),
        sa.Column("reason_codes", json_type, nullable=False),
        sa.Column("component_scores", json_type, nullable=False),
        sa.Column("model_version", sa.String(length=64), nullable=False),
        sa.Column("policy_version", sa.String(length=64), nullable=False),
        sa.Column("feature_version", sa.String(length=32), nullable=False),
        sa.Column("request_id", sa.String(length=64), nullable=False),
        sa.Column("latency_ms", sa.Float(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["transaction_id"], ["transactions.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("transaction_id", name="uq_predictions_transaction_id"),
    )
    op.create_index("ix_predictions_created_at", "predictions", ["created_at"])
    op.create_index(
        "ix_predictions_decision_created",
        "predictions",
        ["decision", "created_at"],
    )

    op.create_table(
        "cases",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("prediction_id", sa.String(length=36), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("priority", sa.String(length=16), nullable=False),
        sa.Column("assigned_to", sa.String(length=128), nullable=True),
        sa.Column("resolution", sa.String(length=32), nullable=True),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["prediction_id"], ["predictions.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("prediction_id", name="uq_cases_prediction_id"),
    )
    op.create_index(
        "ix_cases_queue", "cases", ["status", "priority", "created_at"]
    )

    op.create_table(
        "case_events",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("case_id", sa.String(length=36), nullable=False),
        sa.Column("event_type", sa.String(length=32), nullable=False),
        sa.Column("actor", sa.String(length=128), nullable=False),
        sa.Column("details", json_type, nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["case_id"], ["cases.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_case_events_case_created", "case_events", ["case_id", "created_at"]
    )

    op.create_table(
        "investigator_feedback",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("case_id", sa.String(length=36), nullable=False),
        sa.Column("disposition", sa.String(length=32), nullable=False),
        sa.Column("fraud_confirmed", sa.Boolean(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("investigator", sa.String(length=128), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["case_id"], ["cases.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_feedback_case_created",
        "investigator_feedback",
        ["case_id", "created_at"],
    )

    op.create_table(
        "model_versions",
        sa.Column("version", sa.String(length=64), nullable=False),
        sa.Column("feature_version", sa.String(length=32), nullable=False),
        sa.Column("policy_version", sa.String(length=64), nullable=False),
        sa.Column("artifact_path", sa.Text(), nullable=False),
        sa.Column("active", sa.Boolean(), nullable=False),
        sa.Column(
            "registered_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("version"),
    )
    op.execute(
        sa.text(
            "INSERT INTO model_versions "
            "(version, feature_version, policy_version, artifact_path, active) "
            "VALUES ('v0.5', 'v0.6.0', 'v0.5-policy', "
            "'models/v0.5/risk_bundle.joblib', TRUE)"
        )
    )
    op.create_foreign_key(
        "fk_predictions_model_version",
        "predictions",
        "model_versions",
        ["model_version"],
        ["version"],
    )


def downgrade() -> None:
    op.drop_table("model_versions")
    op.drop_index(
        "ix_feedback_case_created", table_name="investigator_feedback"
    )
    op.drop_table("investigator_feedback")
    op.drop_index("ix_case_events_case_created", table_name="case_events")
    op.drop_table("case_events")
    op.drop_index("ix_cases_queue", table_name="cases")
    op.drop_table("cases")
    op.drop_index("ix_predictions_decision_created", table_name="predictions")
    op.drop_index("ix_predictions_created_at", table_name="predictions")
    op.drop_table("predictions")
    op.drop_index("ix_transactions_received_at", table_name="transactions")
    op.drop_table("transactions")
