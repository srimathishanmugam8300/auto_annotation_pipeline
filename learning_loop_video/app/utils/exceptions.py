class PipelineError(Exception):
    """Base exception for pipeline failures."""


class VideoReadError(PipelineError):
    """Raised when the input video cannot be read."""


class ModelLoadingError(PipelineError):
    """Raised when a model cannot be initialized."""


class ExportError(PipelineError):
    """Raised when generated artifacts cannot be written."""

