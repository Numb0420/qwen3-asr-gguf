from pydantic import BaseModel, Field
from typing import Optional


class ErrorResponse(BaseModel):
    code: str = Field(..., description="Machine-readable error identifier")
    message: str = Field(..., description="Human-readable error description")
    context: Optional[dict] = Field(None, description="Debug data")
    statusCode: int = Field(..., description="HTTP status code")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Status of the service", examples=["ok"])
    model_loaded: bool = Field(..., description="Whether the GGUF engine is loaded")
    backend: str = Field(..., description="Inference backend", examples=["gguf"])
    state: str = Field(..., description="Backend lifecycle state")
    model_id: Optional[str] = Field(None, description="Decoder filename")
    onnx_provider: Optional[str] = Field(None, description="ONNX EP", examples=["DML"])
    llm_gpu: Optional[str] = Field(None, description="Decoder accelerator", examples=["vulkan"])
    aligner: bool = Field(False, description="Whether ForcedAligner is loaded (offline JSON timestamps)")
    ws_final_align: bool = Field(
        False,
        description="Whether WebSocket finals run ForcedAligner and emit chars",
    )
    format_num: bool = Field(False, description="Whether Chinese number ITN is enabled")


class WordTimestamp(BaseModel):
    word: str = Field(..., description="Aligned token or character")
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")


class TranscribePathRequest(BaseModel):
    audio_path: str = Field(..., description="Local audio file path")
    language: str = Field("auto", description="zh / en / auto")


class TaskSubmitResponse(BaseModel):
    ok: bool
    task_id: str
    status: str
    progress_percent: float


class CharTimestamp(BaseModel):
    text: str
    start: float
    end: float


class TranscriptSegment(BaseModel):
    index: int
    start: float
    end: float
    text: str
    punctuation: str = ""
    speaker: Optional[str] = None
    chars: list[CharTimestamp] = Field(default_factory=list)


class TaskResultResponse(BaseModel):
    segments: list[TranscriptSegment]


class TaskStatusResponse(BaseModel):
    task_id: str
    filename: str
    status: str = Field(..., description="running / completed / failed / cancelled")
    message: str
    progress_percent: float
    completed_windows: int
    total_windows: int
    created_at: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    files: dict = Field(default_factory=dict)


API_TAGS = [
    {
        "name": "Transcription",
        "description": "Offline file transcription tasks.",
    },
    {
        "name": "Streaming",
        "description": "Single-connection WebSocket real-time transcription.",
    },
    {
        "name": "System",
        "description": "Health checks and model status.",
    },
]

API_DESCRIPTION = """\
Local speech-to-text API using Qwen3-ASR (ONNX Encoder + GGUF Decoder).

## Features
- Offline HTTP tasks `/offline/transcribe` and `/offline/transcribe-path`
- Single-connection WebSocket `/realtime/stream`
- Windows local runtime (DirectML + Vulkan), no Docker / PyTorch

## Audio Formats
Supported: WAV, FLAC, MP3, OGG, AIFF, CAF, AU, W64, RF64.
"""
