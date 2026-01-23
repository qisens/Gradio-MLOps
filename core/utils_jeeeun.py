import os
from typing import Dict, List

def scan_run_dirs(root_dir: str) -> Dict[str, str]:
    """
    root_dir 하위의 디렉토리를 순회하면서
    target_filename 이 존재하는 경우만 수집한다.

    반환:
        { run_name: run_dir_path }
    """
    result = {}

    if not os.path.isdir(root_dir):
        return result

    for name in os.listdir(root_dir):
        dir_path = os.path.join(root_dir, name)
        if os.path.isdir(dir_path):
            result[name] = dir_path

    return result


def sort_map_by_mtime_desc(path_map: Dict[str, str]) -> List[str]:
    """
    path_map 을 파일 수정시간 기준 내림차순으로 정렬한 key 리스트 반환
    """
    return sorted(
        path_map.keys(),
        key=lambda k: os.path.getmtime(path_map[k]),
        reverse=True,
    )


def pick_default_key(sorted_keys: List[str]) -> str | None:
    """
    정렬된 key 목록에서 default 선택
    """
    if not sorted_keys:
        return None
    return sorted_keys[0]


def resolve_selected_path(
    selected_key: str,
    path_map: Dict[str, str],
) -> str:
    """
    Dropdown에서 선택된 key를 실제 경로로 변환
    """
    if not selected_key:
        return ""
    if isinstance(path_map, dict) and selected_key in path_map:
        return path_map[selected_key]
    return ""


def resolve_run_artifact(run_dir: str, artifact: str) -> str | None:
    """
    run_dir에서 특정 artifact 경로를 반환

    artifact 예:
    - "results_csv"
    - "weights_dir"
    """
    if not run_dir:
        return None

    if artifact == "results_csv":
        path = os.path.join(run_dir, "results.csv")
        return path if os.path.isfile(path) else None

    if artifact == "weights_dir":
        path = os.path.join(run_dir, "weights")
        return path if os.path.isdir(path) else None

    return None

def build_artifact_path_map(
    run_dir_map: Dict[str, str],
    artifact: str,
) -> Dict[str, str]:
    """
    run_dir_map에서 특정 artifact가 존재하는 run만 필터링

    반환:
        { run_name: artifact_path }
    """
    result = {}

    for run_name, run_dir in run_dir_map.items():
        path = resolve_run_artifact(run_dir, artifact)
        if path:
            result[run_name] = path

    return result