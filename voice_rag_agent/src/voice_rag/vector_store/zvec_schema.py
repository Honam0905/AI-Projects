"""Zvec collection schema helpers."""

from __future__ import annotations

from zvec import CollectionSchema, FieldSchema, VectorSchema
from zvec.typing import DataType


def build_kb_collection_schema(
    collection_name: str,
    embedding_dimension: int,
) -> CollectionSchema:
    """Build schema for KB chunk storage."""

    fields = [
        FieldSchema(name="doc_id", data_type=DataType.STRING),
        FieldSchema(name="source_name", data_type=DataType.STRING),
        FieldSchema(name="page", data_type=DataType.INT32),
        FieldSchema(name="chunk_index", data_type=DataType.INT32),
        FieldSchema(name="text", data_type=DataType.STRING),
        FieldSchema(name="created_at", data_type=DataType.INT64),
    ]
    vectors = [
        VectorSchema(
            name="embedding",
            data_type=DataType.VECTOR_FP32,
            dimension=embedding_dimension,
        )
    ]
    return CollectionSchema(name=collection_name, fields=fields, vectors=vectors)
