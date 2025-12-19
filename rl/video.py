from __future__ import annotations

import os
from typing import Iterable

import numpy as np


def can_write_mp4() -> bool:
    """
    Check if MP4 writing is available (imageio + working ffmpeg).
    """
    try:
        import imageio.v2 as _imageio  # type: ignore
        import imageio_ffmpeg  # noqa: F401
        
        # Actually test if ffmpeg exe can be found (imageio-ffmpeg may be installed but broken)
        try:
            _ = imageio_ffmpeg.get_ffmpeg_exe()
            return True
        except RuntimeError:
            # imageio-ffmpeg is installed but can't find ffmpeg. Try system ffmpeg as fallback.
            import shutil
            if shutil.which("ffmpeg") is not None:
                # System ffmpeg available; set env var so imageio uses it
                os.environ.setdefault("IMAGEIO_FFMPEG_EXE", shutil.which("ffmpeg"))
                return True
            return False
    except Exception:
        return False


def write_mp4(path: str, frames: Iterable[np.ndarray], *, fps: int = 30) -> None:
    """
    Write frames (H,W,3) uint8 to an MP4 (H.264) video.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    try:
        import imageio.v2 as imageio  # type: ignore
        import imageio_ffmpeg
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "imageio is required for MP4 writing. Install with: pip install imageio imageio-ffmpeg"
        ) from e

    # Ensure ffmpeg is available. Prefer bundled, fallback to system only if bundled fails.
    # Clear any existing IMAGEIO_FFMPEG_EXE to force bundled version first.
    bundled_works = False
    if "IMAGEIO_FFMPEG_EXE" in os.environ:
        # Temporarily clear to test bundled
        old_exe = os.environ.pop("IMAGEIO_FFMPEG_EXE")
        try:
            _ = imageio_ffmpeg.get_ffmpeg_exe()
            bundled_works = True
        except RuntimeError:
            # Bundled failed, restore system path
            os.environ["IMAGEIO_FFMPEG_EXE"] = old_exe
    else:
        try:
            _ = imageio_ffmpeg.get_ffmpeg_exe()
            bundled_works = True
        except RuntimeError:
            pass
    
    if not bundled_works:
        # Try system ffmpeg as fallback
        import shutil
        system_ffmpeg = shutil.which("ffmpeg")
        if system_ffmpeg:
            os.environ.setdefault("IMAGEIO_FFMPEG_EXE", system_ffmpeg)
        else:
            raise RuntimeError(
                "No ffmpeg found. Install via: brew install ffmpeg (macOS) or apt install ffmpeg (Linux). "
                "Or ensure imageio-ffmpeg can find its bundled ffmpeg."
            )

    # Note: imageio uses ffmpeg (via imageio-ffmpeg) for mp4.
    # yuv420p maximizes player compatibility.
    # If system ffmpeg fails (broken pipe), try bundled version.
    try:
        with imageio.get_writer(
            path,
            fps=fps,
            codec="libx264",
            quality=8,
            pixelformat="yuv420p",
            macro_block_size=None,
        ) as w:
            for f in frames:
                w.append_data(f)
    except (OSError, BrokenPipeError) as e:
        # System ffmpeg may have broken dependencies. Try bundled version.
        if "IMAGEIO_FFMPEG_EXE" in os.environ:
            os.environ.pop("IMAGEIO_FFMPEG_EXE")
            try:
                # Force bundled version
                _ = imageio_ffmpeg.get_ffmpeg_exe()
                with imageio.get_writer(
                    path,
                    fps=fps,
                    codec="libx264",
                    quality=8,
                    pixelformat="yuv420p",
                    macro_block_size=None,
                ) as w:
                    for f in frames:
                        w.append_data(f)
            except Exception as e2:
                # Provide helpful error message
                if "libtheora" in str(e) or "Broken pipe" in str(e):
                    raise RuntimeError(
                        f"System ffmpeg has broken dependencies (missing libtheora). "
                        f"Fix with: brew reinstall ffmpeg theora (macOS). "
                        f"Bundled ffmpeg also unavailable. Error: {e2}"
                    ) from e2
                else:
                    raise RuntimeError(
                        f"Video writing failed. System error: {e}. Bundled error: {e2}. "
                        f"Try: brew reinstall ffmpeg (macOS) or ensure imageio-ffmpeg is properly installed."
                    ) from e2
        else:
            # If no system ffmpeg was set, the error is from bundled version
            if "Broken pipe" in str(e) or "libtheora" in str(e):
                raise RuntimeError(
                    f"ffmpeg failed with broken dependencies. Fix with: brew reinstall ffmpeg theora (macOS)"
                ) from e
            raise


