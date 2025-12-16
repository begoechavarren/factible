from pathlib import Path
from typing import Dict, List, Optional
import logging
import yaml

from experiments.evaluator.models import VideoGroundTruth

_logger = logging.getLogger(__name__)


class GroundTruthManager:
    """Manages loading and caching of ground truth annotations"""

    def __init__(self, ground_truth_dir: Path):
        self.gt_dir = Path(ground_truth_dir)
        self.cache: Dict[str, VideoGroundTruth] = {}

        if not self.gt_dir.exists():
            raise FileNotFoundError(
                f"Ground truth directory does not exist: {self.gt_dir}\n"
                f"Expected ground truth YAML files in this location."
            )

    def load(self, video_id: str) -> VideoGroundTruth:
        """Load ground truth for a video"""
        if video_id in self.cache:
            return self.cache[video_id]

        path = self.gt_dir / f"{video_id}.yaml"
        if not path.exists():
            alternative_path = self._find_file_by_video_id(video_id)
            if alternative_path is None:
                raise FileNotFoundError(
                    f"Ground truth file not found: {path}\n"
                    f"Create it using the ground truth template as reference."
                )
            path = alternative_path

        with open(path) as f:
            data = yaml.safe_load(f)

        gt = VideoGroundTruth(**data)
        self.cache[video_id] = gt
        return gt

    def _find_file_by_video_id(self, video_id: str) -> Optional[Path]:
        """Search YAML files for a matching video_id field."""
        for path in self.gt_dir.glob("*.yaml"):
            if path.stem == "TEMPLATE_video":
                continue

            try:
                with open(path) as f:
                    data = yaml.safe_load(f) or {}
            except Exception:
                continue

            if data.get("video_id") == video_id:
                return path

        return None

    def list_available(self) -> List[str]:
        """List all available ground truth video IDs"""
        video_ids: List[str] = []

        for path in sorted(self.gt_dir.glob("*.yaml")):
            if path.stem == "TEMPLATE_video":
                continue

            video_id: Optional[str] = None
            try:
                with open(path) as f:
                    data = yaml.safe_load(f) or {}
                raw_video_id = data.get("video_id")
                if isinstance(raw_video_id, str) and raw_video_id.strip():
                    video_id = raw_video_id.strip()
            except Exception as exc:
                _logger.warning(
                    "Failed to parse video_id from %s: %s. Falling back to filename.",
                    path.name,
                    exc,
                )

            video_ids.append(video_id or path.stem)

        return video_ids
