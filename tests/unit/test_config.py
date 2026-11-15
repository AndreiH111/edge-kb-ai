"""Offline checks for core.config.Settings: aliases, blank skipping, derived defaults."""

import os
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from core.config import BASE_DIR, Settings

# Blank means unset (env_ignore_empty), so this neutralises conftest's pins and
# whatever the developer's shell exports for the variables under test.
_UNSET = dict.fromkeys(
    (
        "LLM_MODE",
        "LOCAL_LLM_HOSTED",
        "CHUNKING_MODE",
        "EMBEDDING_MODE",
        "HOSTED_EMBEDDING_PROVIDER",
        "HOSTED_LLM_API_KEY",
        "GROQ_API_KEY",
        "OPENAI_API_KEY",
        "HOSTED_EMBEDDING_API_KEY",
        "AWS_S3_BUCKET",
        "S3_BUCKET_NAME",
        "AWS_S3_PREFIX",
        "S3_PREFIX",
        "DATA_DIR",
        "CHROMA_PORT",
    ),
    "",
)


def _settings(**env: str) -> Settings:
    with patch.dict(os.environ, {**_UNSET, **env}):
        return Settings()


class SettingsTests(unittest.TestCase):
    def test_alias_chain_resolves_legacy_names(self) -> None:
        cfg = _settings(GROQ_API_KEY="groq", S3_BUCKET_NAME="bucket", S3_PREFIX="/docs/")
        self.assertEqual(cfg.hosted_llm_api_key, "groq")
        self.assertEqual(cfg.aws_s3_bucket, "bucket")
        self.assertEqual(cfg.aws_s3_prefix, "docs")

    def test_blank_value_falls_through_alias_chain(self) -> None:
        cfg = _settings(HOSTED_LLM_API_KEY="", OPENAI_API_KEY="openai")
        self.assertEqual(cfg.hosted_llm_api_key, "openai")
        self.assertEqual(cfg.hosted_embedding_api_key, "openai")

    def test_legacy_local_llm_hosted_false_means_hosted(self) -> None:
        cfg = _settings(LOCAL_LLM_HOSTED="false")
        self.assertEqual(cfg.llm_mode, "hosted")
        self.assertFalse(cfg.use_local_llm)
        self.assertFalse(cfg.local_llm_hosted)
        self.assertEqual(_settings().llm_mode, "local")
        self.assertEqual(_settings(LOCAL_LLM_HOSTED="false", LLM_MODE="ollama").llm_mode, "local")

    def test_chunking_mode_defaults_to_llm_mode(self) -> None:
        cfg = _settings(LLM_MODE=" Remote ")
        self.assertEqual((cfg.llm_mode, cfg.chunking_mode), ("hosted", "hosted"))
        self.assertEqual(_settings(LLM_MODE="hosted", CHUNKING_MODE="local").chunking_mode, "local")

    def test_embedding_mode_inferred_from_provider(self) -> None:
        self.assertEqual(_settings().embedding_mode, "local")
        self.assertEqual(_settings().hosted_embedding_provider, "openai")
        cfg = _settings(HOSTED_EMBEDDING_PROVIDER="Hosted")
        self.assertEqual((cfg.embedding_mode, cfg.hosted_embedding_provider), ("hosted", "openai"))
        cfg = _settings(HOSTED_EMBEDDING_PROVIDER="hash")
        self.assertEqual((cfg.embedding_mode, cfg.hosted_embedding_provider), ("local", "hash"))
        self.assertTrue(_settings(EMBEDDING_MODE="hash").use_custom_hash_embeddings)

    def test_relative_data_dir_resolved_against_base_dir(self) -> None:
        self.assertEqual(_settings(DATA_DIR="uploads").data_dir, BASE_DIR / "uploads")
        absolute = BASE_DIR.parent / "uploads"
        self.assertEqual(_settings(DATA_DIR=str(absolute)).data_dir, absolute)
        self.assertEqual(_settings().data_dir, BASE_DIR / "data")

    def test_model_copy_update(self) -> None:
        cfg = _settings()
        copy = cfg.model_copy(update={"retriever_max_file_chars": 10})
        self.assertEqual(copy.retriever_max_file_chars, 10)
        self.assertEqual(cfg.retriever_max_file_chars, 12000)
        with self.assertRaises(ValidationError):
            cfg.retriever_k = 1  # frozen

    def test_invalid_values_raise(self) -> None:
        with self.assertRaisesRegex(ValidationError, "LLM_MODE must be either"):
            _settings(LLM_MODE="cloud")
        with self.assertRaisesRegex(ValidationError, "chroma_port"):
            _settings(CHROMA_PORT="notanint")


if __name__ == "__main__":
    unittest.main()
