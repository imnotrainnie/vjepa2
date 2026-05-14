import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from tqdm import tqdm


FRAME_REGEX = re.compile(r"frame_(\d+)\.")


@dataclass
class PipelineConfig:
    csv_path: str = "/data/epic-kitchens-100-annotations/EPIC_100_validation.csv"
    visor_dir: str = "/data/ek"
    rgb_root: str = "/data/rgb_frames"
    output_root: str = "/nvme/vjepa_data_32f"
    manifest_csv: str = "/nvme/vjepa_data_32f/manifest_5verbs_40each.csv"
    frame_count: int = 32
    min_duration: int = 32
    verb_classes: Tuple[int, ...] = (0, 1, 2, 3, 4)
    per_class_quota: int = 40
    image_height: int = 1080
    image_width: int = 1920
    random_seed: int = 42
    downloader_script: str = "/data/epic-kitchens-download-scripts/epic_downloader.py"
    raw_tars_dir: str = "/data/raw_tars"
    skip_download: bool = False
    no_cleanup: bool = False
    use_nearest_visor_frame: bool = True
    max_visor_frame_gap: int = 128
    anticipation_time_sec: Optional[float] = None
    fps: Optional[float] = None


def _parse_frame_id_from_name(name: str) -> Optional[int]:
    match = FRAME_REGEX.search(name)
    if match is None:
        return None
    return int(match.group(1))


def _sample_32_indices(start_frame: int, stop_frame: int, frame_count: int) -> List[int]:
    sampled = np.linspace(start_frame, stop_frame, num=frame_count)
    sampled = np.round(sampled).astype(np.int64)
    sampled = np.clip(sampled, start_frame, stop_frame)
    return sampled.tolist()


def _sample_strict_anticipation_indices(
    action_start_frame: int,
    frame_count: int,
    anticipation_time_sec: float,
    fps: float,
) -> Tuple[List[int], int, int, int]:
    tau_frame_offset = int(round(float(anticipation_time_sec) * float(fps)))
    context_stop_frame = int(action_start_frame) - tau_frame_offset
    context_start_frame = context_stop_frame - int(frame_count) + 1
    sampled = _sample_32_indices(context_start_frame, context_stop_frame, frame_count)
    return sampled, context_start_frame, context_stop_frame, tau_frame_offset


def _normalize_polygon(poly: Sequence) -> Optional[np.ndarray]:
    if not isinstance(poly, Sequence) or len(poly) < 3:
        return None

    points = []
    for point in poly:
        if not isinstance(point, Sequence) or len(point) < 2:
            continue
        x = int(round(float(point[0])))
        y = int(round(float(point[1])))
        points.append([x, y])

    if len(points) < 3:
        return None

    return np.array(points, dtype=np.int32)


def _extract_polygons_from_annotation(annotation: dict) -> List[np.ndarray]:
    polygons: List[np.ndarray] = []
    segments = annotation.get("segments", [])

    if isinstance(segments, dict):
        candidate_segments = [segments]
    else:
        candidate_segments = segments

    for seg in candidate_segments:
        if isinstance(seg, dict):
            if "points" in seg:
                poly = _normalize_polygon(seg["points"])
                if poly is not None:
                    polygons.append(poly)
            elif "polygon" in seg:
                poly = _normalize_polygon(seg["polygon"])
                if poly is not None:
                    polygons.append(poly)
        else:
            poly = _normalize_polygon(seg)
            if poly is not None:
                polygons.append(poly)

    return polygons


def _build_frame_to_polygons_map(visor_json: dict) -> Dict[int, List[np.ndarray]]:
    frame_to_polygons: Dict[int, List[np.ndarray]] = {}
    video_annotations = visor_json.get("video_annotations", [])

    for frame_entry in video_annotations:
        image_info = frame_entry.get("image", {})
        image_name = image_info.get("name") or image_info.get("image_path", "")
        frame_id = _parse_frame_id_from_name(image_name)
        if frame_id is None:
            continue

        polygons: List[np.ndarray] = []
        for annotation in frame_entry.get("annotations", []):
            polygons.extend(_extract_polygons_from_annotation(annotation))

        frame_to_polygons[frame_id] = polygons

    return frame_to_polygons


def _get_polygons_for_frame(
    frame_to_polygons: Dict[int, List[np.ndarray]],
    source_frame_id: int,
    use_nearest_visor_frame: bool,
    max_visor_frame_gap: int,
) -> List[np.ndarray]:
    if source_frame_id in frame_to_polygons:
        return frame_to_polygons[source_frame_id]

    if not use_nearest_visor_frame or not frame_to_polygons:
        return []

    keys = np.fromiter(frame_to_polygons.keys(), dtype=np.int64)
    nearest_idx = int(np.argmin(np.abs(keys - int(source_frame_id))))
    nearest_frame_id = int(keys[nearest_idx])
    if abs(nearest_frame_id - int(source_frame_id)) <= int(max_visor_frame_gap):
        return frame_to_polygons.get(nearest_frame_id, [])
    return []


def _resolve_rgb_frame_path(video_rgb_dir: Path, frame_id: int) -> Optional[Path]:
    candidates = [
        video_rgb_dir / f"frame_{frame_id:010d}.jpg",
        video_rgb_dir / f"frame_{frame_id:010d}.png",
        video_rgb_dir / f"{frame_id:010d}.jpg",
        video_rgb_dir / f"{frame_id:010d}.png",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    glob_candidates = list(video_rgb_dir.glob(f"*{frame_id:010d}*.jpg"))
    if glob_candidates:
        return glob_candidates[0]

    glob_candidates = list(video_rgb_dir.glob(f"*{frame_id:010d}*.png"))
    if glob_candidates:
        return glob_candidates[0]

    # 兼容解压后出现多层目录结构的情况
    glob_candidates = list(video_rgb_dir.rglob(f"*{frame_id:010d}*.jpg"))
    if glob_candidates:
        return glob_candidates[0]

    glob_candidates = list(video_rgb_dir.rglob(f"*{frame_id:010d}*.png"))
    if glob_candidates:
        return glob_candidates[0]

    return None


def _video_dir_has_images(video_rgb_dir: Path) -> bool:
    if not video_rgb_dir.exists() or not video_rgb_dir.is_dir():
        return False

    for pattern in ("*.jpg", "*.png", "*.jpeg", "*.JPG", "*.PNG", "*.JPEG"):
        if any(video_rgb_dir.rglob(pattern)):
            return True
    return False


def _find_video_archives(video_id: str, raw_tars_dir: Path) -> List[Path]:
    patterns = [
        f"{video_id}.tar",
        f"{video_id}.tar.gz",
        f"{video_id}.tgz",
        f"{video_id}*.tar",
        f"{video_id}*.tar.gz",
        f"{video_id}*.tgz",
    ]
    archives: List[Path] = []
    for pattern in patterns:
        # 下载器通常会把文件放在多层目录中，如:
        # /data/raw_tars/EPIC-KITCHENS/P02/rgb_frames/P02_12.tar
        archives.extend(raw_tars_dir.rglob(pattern))

    # 去重并按修改时间降序（最新优先）
    unique_archives = sorted(
        {
            p.resolve()
            for p in archives
            if p.is_file() and p.suffix in {".tar", ".gz", ".tgz"}
        },
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    return unique_archives


def _download_video_archive(video_id: str, config: PipelineConfig) -> List[Path]:
    raw_tars_dir = Path(config.raw_tars_dir)
    raw_tars_dir.mkdir(parents=True, exist_ok=True)
    downloader_script_path = Path(config.downloader_script).resolve()
    downloader_workdir = downloader_script_path.parent

    if not downloader_script_path.exists():
        raise FileNotFoundError(f"下载脚本不存在: {downloader_script_path}")

    # epic_downloader.py 依赖相对路径 data/epic_55_splits.csv 与 data/epic_100_splits.csv
    epic55_split = downloader_workdir / "data" / "epic_55_splits.csv"
    epic100_split = downloader_workdir / "data" / "epic_100_splits.csv"
    if not epic55_split.exists():
        raise FileNotFoundError(f"缺少 split 文件: {epic55_split}")
    if not epic100_split.exists():
        raise FileNotFoundError(f"缺少 split 文件: {epic100_split}")

    command_candidates = [
        [
            sys.executable,
            str(downloader_script_path),
            "--output-path",
            str(raw_tars_dir),
            "--rgb-frames",
            "--specific-videos",
            video_id,
        ],
        [
            sys.executable,
            str(downloader_script_path),
            "--parts",
            "rgb",
            "--ids",
            video_id,
        ],
    ]

    command_errors: List[str] = []
    print(f"[下载] 正在下载 {video_id} ...")
    for idx, command in enumerate(command_candidates, start=1):
        print(f"[下载] 尝试命令 {idx}: {' '.join(command)}")
        result = subprocess.run(
            command,
            cwd=str(downloader_workdir),
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            archives = _find_video_archives(video_id, raw_tars_dir)
            if archives:
                return archives
            command_errors.append(
                f"命令{idx}执行成功但未找到压缩包。stdout:\n{result.stdout[-1000:]}\nstderr:\n{result.stderr[-1000:]}"
            )
            continue

        command_errors.append(
            f"命令{idx}失败(returncode={result.returncode})。stdout:\n{result.stdout[-1000:]}\nstderr:\n{result.stderr[-1000:]}"
        )

    raise RuntimeError(
        f"下载失败: {video_id}\n" + "\n\n".join(command_errors)
    )


def _extract_archive_to_video_dir(archive_path: Path, rgb_video_dir: Path, video_id: str) -> None:
    print(f"[解压] 正在解压 {video_id}: {archive_path.name}")
    rgb_video_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, mode="r:*") as tar_handle:
        tar_handle.extractall(path=rgb_video_dir)


def _cleanup_video_artifacts(
    video_id: str,
    rgb_root: Path,
    raw_tars_dir: Path,
    cleanup_rgb_dir: bool = True,
    cleanup_archives: bool = True,
) -> None:
    print(f"[清理] 正在清理 {video_id} 释放磁盘空间...")
    if cleanup_rgb_dir:
        rgb_video_dir = rgb_root / video_id
        shutil.rmtree(rgb_video_dir, ignore_errors=True)

    if cleanup_archives:
        archive_paths = _find_video_archives(video_id, raw_tars_dir)
        for archive_path in archive_paths:
            try:
                archive_path.unlink(missing_ok=True)
            except Exception:
                pass


def generate_manifest(config: PipelineConfig) -> pd.DataFrame:
    """
    生成满足条件的 200 个 clip 清单：
    - verb_class 属于 [0,1,2,3,4]
    - stop_frame - start_frame >= 32
    - 存在同名 VISOR JSON
    - 每类动作精确抽取 40 个
    - 若设置 anticipation_time_sec + fps，则按严格 anticipation 窗口重算 sampled_frames
    """
    df = pd.read_csv(config.csv_path)
    required_columns = {
        "narration_id",
        "video_id",
        "start_frame",
        "stop_frame",
        "verb_class",
    }
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"CSV 缺少必要字段: {missing_columns}")

    df = df.copy()
    total_rows = len(df)

    visor_dir = Path(config.visor_dir)
    if not visor_dir.exists():
        raise FileNotFoundError(f"VISOR 目录不存在: {visor_dir}")

    # 优先检查 VISOR 标注是否存在（按用户要求）
    available_visor_ids = {p.stem for p in visor_dir.glob("*.json")}
    df = df[df["video_id"].isin(available_visor_ids)].copy()
    after_visor_rows = len(df)

    # 对有 VISOR 标注的数据再进行其他条件过滤
    df["duration"] = df["stop_frame"] - df["start_frame"]
    df = df[df["verb_class"].isin(config.verb_classes)]
    after_verb_rows = len(df)
    df = df[df["duration"] >= config.min_duration]
    after_duration_rows = len(df)

    strict_anticipation_enabled = (
        config.anticipation_time_sec is not None and config.fps is not None
    )
    if strict_anticipation_enabled:
        tau_frame_offset = int(round(float(config.anticipation_time_sec) * float(config.fps)))
        df["anticipation_anchor_frame"] = df["start_frame"] - tau_frame_offset
        df["strict_context_start_frame"] = df["anticipation_anchor_frame"] - int(config.frame_count) + 1
        df = df[df["strict_context_start_frame"] >= 1].copy()
        after_strict_anticipation_rows = len(df)
    else:
        after_strict_anticipation_rows = after_duration_rows

    print(
        "[generate_manifest] 过滤统计: "
        f"total={total_rows}, "
        f"after_visor={after_visor_rows}, "
        f"after_verb={after_verb_rows}, "
        f"after_duration={after_duration_rows}, "
        f"after_strict_anticipation={after_strict_anticipation_rows}"
    )

    selected_rows: List[pd.DataFrame] = []
    rng_seed = config.random_seed
    for verb_class in config.verb_classes:
        sub = df[df["verb_class"] == verb_class].drop_duplicates(subset=["narration_id"]).copy()
        if len(sub) < config.per_class_quota:
            raise RuntimeError(
                f"verb_class={verb_class} 仅有 {len(sub)} 条，无法满足每类 {config.per_class_quota} 条需求"
            )
        sampled = sub.sample(n=config.per_class_quota, random_state=rng_seed).sort_values("narration_id")
        selected_rows.append(sampled)

    manifest = pd.concat(selected_rows, axis=0).reset_index(drop=True)
    manifest["clip_id"] = manifest["narration_id"].astype(str)

    if strict_anticipation_enabled:
        sampled_info = manifest.apply(
            lambda row: _sample_strict_anticipation_indices(
                action_start_frame=int(row["start_frame"]),
                frame_count=config.frame_count,
                anticipation_time_sec=float(config.anticipation_time_sec),
                fps=float(config.fps),
            ),
            axis=1,
        )
        manifest["sampled_frames"] = sampled_info.map(lambda x: json.dumps(x[0], ensure_ascii=False))
        manifest["context_start_frame"] = sampled_info.map(lambda x: int(x[1]))
        manifest["context_stop_frame"] = sampled_info.map(lambda x: int(x[2]))
        manifest["anticipation_frame_offset"] = sampled_info.map(lambda x: int(x[3]))
        manifest["anticipation_time_sec"] = float(config.anticipation_time_sec)
        manifest["fps"] = float(config.fps)
        manifest["sampling_mode"] = "strict_anticipation"
    else:
        manifest["sampled_frames"] = manifest.apply(
            lambda row: json.dumps(
                _sample_32_indices(int(row["start_frame"]), int(row["stop_frame"]), config.frame_count),
                ensure_ascii=False,
            ),
            axis=1,
        )
        manifest["context_start_frame"] = manifest["start_frame"].astype(int)
        manifest["context_stop_frame"] = manifest["stop_frame"].astype(int)
        manifest["anticipation_frame_offset"] = 0
        manifest["anticipation_time_sec"] = np.nan
        manifest["fps"] = np.nan
        manifest["sampling_mode"] = "segment_span"

    output_path = Path(config.manifest_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_path, index=False)
    return manifest


def execute_pipeline(
    manifest_df: pd.DataFrame,
    config: PipelineConfig,
) -> pd.DataFrame:
    """
    根据清单执行抽取：
    - 按 video_id 分组（微批处理）
    - 每个 clip 输出 32 张 RGB + 32 张 mask
    - 每个 video_id 完成后强制清理原始文件（阅后即焚）
    """
    output_root = Path(config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    image_backend = "pil"
    cv2 = None
    try:
        import cv2  # pylint: disable=import-outside-toplevel

        image_backend = "cv2"
        print("[后端] 使用 OpenCV 后端处理图像与掩码。")
    except (ModuleNotFoundError, ImportError) as error:
        print(
            "[后端] 无法导入 cv2，将回退到 PIL 后端继续执行。"
            f" 原始错误: {error}"
        )

    def _read_image(path: Path) -> Optional[np.ndarray]:
        if image_backend == "cv2" and cv2 is not None:
            return cv2.imread(str(path), cv2.IMREAD_COLOR)
        try:
            return np.array(Image.open(path).convert("RGB"))
        except Exception:
            return None

    def _write_frame(path: Path, image: np.ndarray) -> bool:
        if image_backend == "cv2" and cv2 is not None:
            return bool(cv2.imwrite(str(path), image))
        try:
            Image.fromarray(image.astype(np.uint8)).save(path)
            return True
        except Exception:
            return False

    def _write_mask(path: Path, mask: np.ndarray) -> bool:
        if image_backend == "cv2" and cv2 is not None:
            return bool(cv2.imwrite(str(path), mask))
        try:
            Image.fromarray(mask.astype(np.uint8), mode="L").save(path)
            return True
        except Exception:
            return False

    def _fill_polygons(mask: np.ndarray, polygons: List[np.ndarray]) -> np.ndarray:
        if not polygons:
            return mask
        if image_backend == "cv2" and cv2 is not None:
            cv2.fillPoly(mask, polygons, 255)
            return mask

        pil_mask = Image.fromarray(mask, mode="L")
        draw = ImageDraw.Draw(pil_mask)
        for poly in polygons:
            draw.polygon([(int(p[0]), int(p[1])) for p in poly.tolist()], fill=255)
        return np.array(pil_mask, dtype=np.uint8)

    if manifest_df.empty:
        raise ValueError("manifest_df 为空，无法执行流水线")

    required_columns = {"clip_id", "video_id", "sampled_frames"}
    missing_columns = required_columns - set(manifest_df.columns)
    if missing_columns:
        raise ValueError(f"manifest_df 缺少必要字段: {missing_columns}")

    summary_records = []
    grouped = manifest_df.groupby("video_id", sort=True)
    raw_tars_dir = Path(config.raw_tars_dir)
    raw_tars_dir.mkdir(parents=True, exist_ok=True)
    rgb_root = Path(config.rgb_root)
    rgb_root.mkdir(parents=True, exist_ok=True)
    rgb_root.mkdir(parents=True, exist_ok=True)

    for video_id, group in tqdm(grouped, total=grouped.ngroups, desc="按video_id微批处理"):
        video_id = str(video_id)
        rgb_video_dir = rgb_root / video_id
        visor_json_path = Path(config.visor_dir) / f"{video_id}.json"
        downloaded_archives: List[Path] = []
        had_local_rgb_before = _video_dir_has_images(rgb_video_dir)
        extracted_in_this_run = False

        try:
            if _video_dir_has_images(rgb_video_dir):
                print(f"[检查] {video_id} 已存在本地帧，跳过下载。")
            else:
                if config.skip_download:
                    for _, row in group.iterrows():
                        summary_records.append(
                            {
                                "clip_id": row["clip_id"],
                                "status": "missing_rgb_skip_download",
                                "video_id": video_id,
                                "error": "skip_download=True 且本地不存在 RGB 帧",
                            }
                        )
                    continue

                downloaded_archives = _download_video_archive(video_id, config)
                try:
                    # 使用最新压缩包进行解压
                    _extract_archive_to_video_dir(downloaded_archives[0], rgb_video_dir, video_id)
                    extracted_in_this_run = True
                except Exception as error:
                    for _, row in group.iterrows():
                        summary_records.append(
                            {
                                "clip_id": row["clip_id"],
                                "status": "extract_failed",
                                "video_id": video_id,
                                "error": str(error),
                            }
                        )
                    continue

            if not _video_dir_has_images(rgb_video_dir):
                for _, row in group.iterrows():
                    summary_records.append(
                        {
                            "clip_id": row["clip_id"],
                            "status": "missing_rgb_after_extract",
                            "video_id": video_id,
                        }
                    )
                continue

            if not visor_json_path.exists():
                for _, row in group.iterrows():
                    summary_records.append({"clip_id": row["clip_id"], "status": "missing_visor_json", "video_id": video_id})
                continue

            try:
                with open(visor_json_path, "r", encoding="utf-8") as file:
                    visor_json = json.load(file)
                frame_to_polygons = _build_frame_to_polygons_map(visor_json)
            except Exception:
                for _, row in group.iterrows():
                    summary_records.append({"clip_id": row["clip_id"], "status": "invalid_visor_json", "video_id": video_id})
                continue

            for _, row in group.iterrows():
                clip_id = str(row["clip_id"])
                clip_output_dir = output_root / clip_id
                frames_output_dir = clip_output_dir / "frames"
                masks_output_dir = clip_output_dir / "masks"
                frames_output_dir.mkdir(parents=True, exist_ok=True)
                masks_output_dir.mkdir(parents=True, exist_ok=True)

                sampled_frames = json.loads(row["sampled_frames"])
                clip_ok = True

                for frame_idx_in_clip, source_frame_id in enumerate(sampled_frames):
                    frame_path = _resolve_rgb_frame_path(rgb_video_dir, int(source_frame_id))
                    if frame_path is None:
                        clip_ok = False
                        break

                    image = _read_image(frame_path)
                    if image is None:
                        clip_ok = False
                        break

                    img_h, img_w = int(image.shape[0]), int(image.shape[1])
                    mask = np.zeros((img_h, img_w), dtype=np.uint8)
                    polygons = _get_polygons_for_frame(
                        frame_to_polygons=frame_to_polygons,
                        source_frame_id=int(source_frame_id),
                        use_nearest_visor_frame=config.use_nearest_visor_frame,
                        max_visor_frame_gap=config.max_visor_frame_gap,
                    )
                    if polygons:
                        clipped_polygons = []
                        for poly in polygons:
                            poly_copy = poly.copy()
                            poly_copy[:, 0] = np.clip(poly_copy[:, 0], 0, img_w - 1)
                            poly_copy[:, 1] = np.clip(poly_copy[:, 1], 0, img_h - 1)
                            clipped_polygons.append(poly_copy)
                        if clipped_polygons:
                            mask = _fill_polygons(mask, clipped_polygons)

                    dst_frame_path = frames_output_dir / f"{frame_idx_in_clip:02d}.jpg"
                    dst_mask_path = masks_output_dir / f"{frame_idx_in_clip:02d}.png"

                    ok_frame = _write_frame(dst_frame_path, image)
                    ok_mask = _write_mask(dst_mask_path, mask)
                    if not (ok_frame and ok_mask):
                        clip_ok = False
                        break

                if clip_ok:
                    summary_records.append({"clip_id": clip_id, "status": "ok", "video_id": video_id})
                else:
                    shutil.rmtree(clip_output_dir, ignore_errors=True)
                    summary_records.append({"clip_id": clip_id, "status": "failed_clip", "video_id": video_id})

        except Exception as error:
            # 防御式兜底：单个 video_id 失败不影响整体流水线
            for _, row in group.iterrows():
                summary_records.append(
                    {
                        "clip_id": row["clip_id"],
                        "status": "video_level_exception",
                        "video_id": video_id,
                        "error": str(error),
                    }
                )
        finally:
            if config.no_cleanup:
                print(f"[清理] no_cleanup=True，保留 {video_id} 的中间文件。")
            else:
                # 仅清理本次流水线产生的临时数据，避免误删已有本地帧
                _cleanup_video_artifacts(
                    video_id=video_id,
                    rgb_root=rgb_root,
                    raw_tars_dir=raw_tars_dir,
                    cleanup_rgb_dir=extracted_in_this_run and (not had_local_rgb_before),
                    cleanup_archives=bool(downloaded_archives),
                )

    summary_df = pd.DataFrame(summary_records)
    summary_path = Path(config.output_root) / "pipeline_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    return summary_df


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EK100 + VISOR 到 V-JEPA 2.1 训练目录结构构建脚本")
    parser.add_argument("--csv-path", default="/data/epic-kitchens-100-annotations/EPIC_100_validation.csv")
    parser.add_argument("--visor-dir", default="/data/ek")
    parser.add_argument("--rgb-root", default="/data/rgb_frames")
    parser.add_argument("--output-root", default="/nvme/vjepa_data_32f")
    parser.add_argument("--manifest-csv", default="/nvme/vjepa_data_32f/manifest_5verbs_40each.csv")
    parser.add_argument("--frame-count", type=int, default=32)
    parser.add_argument("--per-class-quota", type=int, default=40)
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--use-existing-manifest", action="store_true", help="直接读取 --manifest-csv 并执行，不重新生成清单")
    parser.add_argument("--downloader-script", default="/data/epic-kitchens-download-scripts/epic_downloader.py")
    parser.add_argument("--raw-tars-dir", default="/data/raw_tars")
    parser.add_argument("--skip-download", action="store_true", help="跳过下载压缩包；仅使用 --rgb-root 下已存在的本地帧")
    parser.add_argument("--no-cleanup", action="store_true", help="暂时不清理下载的压缩包和解压后的 RGB 帧，便于调试")
    parser.add_argument("--disable-nearest-visor-frame", action="store_true", help="禁用 VISOR 最近帧回退，仅使用精确 frame id 匹配")
    parser.add_argument("--max-visor-frame-gap", type=int, default=128, help="最近 VISOR 帧回退的最大允许帧差")
    parser.add_argument(
        "--anticipation-time-sec",
        type=float,
        default=None,
        help="严格 anticipation 模式：上下文终点设为 action_start_frame - anticipation_time_sec * fps",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="严格 anticipation 模式下使用的帧率；需与 --anticipation-time-sec 一起提供",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    strict_args = (args.anticipation_time_sec is not None, args.fps is not None)
    if strict_args.count(True) == 1:
        parser.error("--anticipation-time-sec 与 --fps 必须同时提供")
    if args.anticipation_time_sec is not None and args.anticipation_time_sec < 0:
        parser.error("--anticipation-time-sec 必须 >= 0")
    if args.fps is not None and args.fps <= 0:
        parser.error("--fps 必须 > 0")

    config = PipelineConfig(
        csv_path=args.csv_path,
        visor_dir=args.visor_dir,
        rgb_root=args.rgb_root,
        output_root=args.output_root,
        manifest_csv=args.manifest_csv,
        frame_count=args.frame_count,
        per_class_quota=args.per_class_quota,
        downloader_script=args.downloader_script,
        raw_tars_dir=args.raw_tars_dir,
        skip_download=args.skip_download,
        no_cleanup=args.no_cleanup,
        use_nearest_visor_frame=not args.disable_nearest_visor_frame,
        max_visor_frame_gap=args.max_visor_frame_gap,
        anticipation_time_sec=args.anticipation_time_sec,
        fps=args.fps,
    )

    if args.use_existing_manifest:
        manifest_path = Path(config.manifest_csv)
        if not manifest_path.exists():
            raise FileNotFoundError(f"指定的 manifest 不存在: {manifest_path}")
        manifest_df = pd.read_csv(manifest_path)
        print(f"读取已有 manifest: {manifest_path}，共 {len(manifest_df)} 条")
    else:
        manifest_df = generate_manifest(config)
        print(f"manifest 生成完成，共 {len(manifest_df)} 条，保存到: {config.manifest_csv}")

    if args.manifest_only:
        return

    summary_df = execute_pipeline(
        manifest_df=manifest_df,
        config=config,
    )
    print("流水线执行完成。状态统计:")
    print(summary_df["status"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
