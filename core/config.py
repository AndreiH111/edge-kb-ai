"""Application settings: one pydantic-settings field per environment variable.

To add a setting, add one field named as its env var lowercased, e.g.
``chroma_port: int = 8000`` reads ``CHROMA_PORT``. Legacy or alternative names go
in ``Field(validation_alias=AliasChoices("NEW_NAME", "OLD_NAME"))``; the first
non-blank one wins. Malformed values (a non-integer port, an unknown mode) raise
a ``ValidationError`` when this module is imported.
"""

from pathlib import Path
from typing import ClassVar, FrozenSet

from dotenv import load_dotenv
from pydantic import AliasChoices, Field, ValidationInfo, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent

# Keep local development predictable: values in .env should win.
load_dotenv(BASE_DIR / ".env", override=True)

_MODES = {
    "local": "local",
    "ollama": "local",
    "custom": "custom",
    "hash": "custom",
    "hosted": "hosted",
    "remote": "hosted",
}


class Settings(BaseSettings):
    """Frozen, env-backed settings. Blank env values count as unset."""

    model_config = SettingsConfigDict(frozen=True, extra="ignore", env_ignore_empty=True)

    base_dir: ClassVar[Path] = BASE_DIR
    db_location: ClassVar[Path] = BASE_DIR / "chrome_langchain_db"
    collection_name: ClassVar[str] = "uploaded_files"
    supported_excel_extensions: ClassVar[FrozenSet[str]] = frozenset({".xlsx", ".xls", ".xlsm"})
    supported_pdf_extensions: ClassVar[FrozenSet[str]] = frozenset({".pdf"})
    supported_powerpoint_extensions: ClassVar[FrozenSet[str]] = frozenset({".pptx", ".pptm"})

    data_dir: Path = BASE_DIR / "data"
    storage_backend: str = Field("local", validation_alias=AliasChoices("FILE_STORAGE_BACKEND"))
    storage_cache_dir: Path = BASE_DIR / ".storage_cache"
    aws_s3_bucket: str = Field("", validation_alias=AliasChoices("AWS_S3_BUCKET", "S3_BUCKET_NAME"))
    aws_s3_prefix: str = Field("", validation_alias=AliasChoices("AWS_S3_PREFIX", "S3_PREFIX"))
    aws_s3_region_name: str = Field(
        "",
        validation_alias=AliasChoices("AWS_REGION_NAME", "AWS_REGION", "AWS_DEFAULT_REGION"),
    )
    aws_s3_endpoint_url: str = ""

    # "" means unset; resolved in _resolve_defaults (legacy LOCAL_LLM_HOSTED fallback).
    llm_mode: str = ""
    # Legacy: historically meant "use the local Ollama model" when true.
    legacy_local_llm_hosted: bool = Field(True, validation_alias=AliasChoices("LOCAL_LLM_HOSTED"))
    ollama_llm_model: str = "gemma4"
    hosted_llm_provider: str = "openai"
    hosted_llm_model: str = "gpt-4o-mini"
    hosted_llm_api_key: str = Field(
        "",
        validation_alias=AliasChoices("HOSTED_LLM_API_KEY", "GROQ_API_KEY", "OPENAI_API_KEY"),
    )
    hosted_llm_base_url: str = ""
    hosted_llm_temperature: float = 1.0
    hosted_llm_top_p: float = 1.0
    hosted_llm_max_tokens: int = 2048
    groq_reasoning_effort: str = "medium"

    # "" means unset; defaults to llm_mode.
    chunking_mode: str = ""
    ollama_chunking_model: str = Field(
        "gemma4",
        validation_alias=AliasChoices("OLLAMA_CHUNKING_MODEL", "OLLAMA_LLM_MODEL"),
    )
    hosted_chunking_provider: str = Field(
        "openai",
        validation_alias=AliasChoices("HOSTED_CHUNKING_PROVIDER", "HOSTED_LLM_CHUNKING_PROVIDER"),
    )
    hosted_chunking_model: str = Field(
        "gpt-4o-mini",
        validation_alias=AliasChoices("HOSTED_CHUNKING_MODEL", "HOSTED_LLM_CHUNKING_MODEL"),
    )
    hosted_chunking_api_key: str = Field(
        "",
        validation_alias=AliasChoices(
            "HOSTED_CHUNKING_API_KEY", "HOSTED_LLM_CHUNKING_API_KEY", "OPENAI_API_KEY"
        ),
    )
    hosted_chunking_base_url: str = Field(
        "",
        validation_alias=AliasChoices("HOSTED_CHUNKING_BASE_URL", "HOSTED_LLM_CHUNKING_BASE_URL"),
    )
    hosted_chunking_temperature: float = Field(
        1.0,
        validation_alias=AliasChoices(
            "HOSTED_CHUNKING_TEMPERATURE", "HOSTED_LLM_CHUNKING_TEMPERATURE"
        ),
    )
    hosted_chunking_top_p: float = Field(
        1.0,
        validation_alias=AliasChoices("HOSTED_CHUNKING_TOP_P", "HOSTED_LLM_CHUNKING_TOP_P"),
    )
    hosted_chunking_max_tokens: int = Field(
        2048,
        validation_alias=AliasChoices(
            "HOSTED_CHUNKING_MAX_TOKENS", "HOSTED_LLM_CHUNKING_MAX_TOKENS"
        ),
    )

    # "" means unset; inferred from hosted_embedding_provider.
    embedding_mode: str = ""
    ollama_embedding_model: str = Field(
        "nomic-embed-text",
        validation_alias=AliasChoices("OLLAMA_EMBEDDING_MODEL", "EMBEDDING_MODEL"),
    )
    # "" means unset; normalised to "openai" after embedding_mode is inferred.
    hosted_embedding_provider: str = ""
    hosted_embedding_model: str = "text-embedding-3-small"
    hosted_embedding_api_key: str = Field(
        "",
        validation_alias=AliasChoices("HOSTED_EMBEDDING_API_KEY", "OPENAI_API_KEY"),
    )
    hosted_embedding_base_url: str = ""
    local_embedding_dimensions: int = 384

    vector_db_backend: str = "chroma"
    chroma_mode: str = "local"
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    chroma_ssl: bool = False
    chroma_headers: str = ""
    hana_db_address: str = ""
    hana_db_port: int = 443
    hana_db_user: str = ""
    hana_db_password: str = ""
    hana_db_table_name: str = "LANGCHAIN_VECTORS"

    retriever_k: int = 5
    retriever_k_per_source: int = 3
    retriever_max_file_chars: int = 12000
    max_chat_history_messages: int = 12
    max_chat_history_content_chars: int = 2000

    # ---- Agentic chat + Jira via MCP ----
    agent_enabled: bool = True
    jira_enabled: bool = False
    jira_url: str = ""
    jira_username: str = ""
    jira_api_token: str = ""
    jira_read_only: bool = True
    # Optional comma-separated allow-list passed to the MCP server as ENABLED_TOOLS.
    jira_enabled_tools: str = ""
    # Transport for the Jira MCP server: "stdio" (spawn a subprocess) or
    # "streamable_http" (connect to an already-running server at jira_mcp_url).
    jira_mcp_transport: str = "stdio"
    jira_mcp_command: str = "mcp-atlassian"
    jira_mcp_url: str = ""

    @field_validator(
        "storage_backend",
        "hosted_llm_provider",
        "hosted_chunking_provider",
        "hosted_embedding_provider",
        "vector_db_backend",
        "chroma_mode",
    )
    @classmethod
    def _normalise_name(cls, value: str) -> str:
        return value.strip().lower()

    @field_validator("llm_mode", "chunking_mode", "embedding_mode")
    @classmethod
    def _normalise_mode(cls, value: str, info: ValidationInfo) -> str:
        if not value:
            return value
        try:
            return _MODES[value.strip().lower()]
        except KeyError as exc:
            name = str(info.field_name).upper()
            raise ValueError(f"{name} must be either 'local', 'custom', or 'hosted'") from exc

    @model_validator(mode="after")
    def _resolve_defaults(self) -> "Settings":
        """Fill cross-field defaults; writes __dict__ because the model is frozen."""
        resolved = self.__dict__
        resolved["llm_mode"] = self.llm_mode or (
            "local" if self.legacy_local_llm_hosted else "hosted"
        )
        resolved["chunking_mode"] = self.chunking_mode or self.llm_mode
        provider = self.hosted_embedding_provider
        resolved["embedding_mode"] = self.embedding_mode or (
            "hosted" if provider not in {"", "local", "hash"} else "local"
        )
        resolved["hosted_embedding_provider"] = "openai" if provider in {"", "hosted"} else provider
        resolved["data_dir"] = BASE_DIR / self.data_dir.expanduser()
        resolved["storage_cache_dir"] = BASE_DIR / self.storage_cache_dir.expanduser()
        resolved["aws_s3_prefix"] = self.aws_s3_prefix.strip("/")
        return self

    @property
    def use_local_llm(self) -> bool:
        return self.llm_mode == "local"

    @property
    def local_llm_hosted(self) -> bool:
        return self.use_local_llm

    @property
    def use_local_chunking(self) -> bool:
        return self.chunking_mode == "local"

    @property
    def use_local_embeddings(self) -> bool:
        return self.embedding_mode == "local"

    @property
    def use_custom_hash_embeddings(self) -> bool:
        return self.embedding_mode == "custom"

    @property
    def supported_file_extensions(self) -> FrozenSet[str]:
        return (
            self.supported_excel_extensions
            | self.supported_pdf_extensions
            | self.supported_powerpoint_extensions
        )


settings = Settings()
