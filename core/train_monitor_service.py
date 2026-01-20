from __future__ import annotations

import math
import os
import re
import glob
import time
from dataclasses import dataclass
from typing import Tuple, Union, List, Optional

import pandas as pd
import gradio as gr
import matplotlib.pyplot as plt

from core.utils_csv import find_latest_results_csv, read_results_csv, ensure_epoch


# ============================================================
# 1) Paging utilities (작고 순수한 유틸은 클래스로 묶지 않아도 OK)
# ============================================================
@dataclass(frozen=True)
class Paging:
    """
    [페이지네이션 계산 유틸]
    여러 개의 그래프(컬럼)를 "page_size개씩" 끊어서 보여주기 위한 계산용 도우미.

    - total_pages(): 전체 페이지 수 계산
    - safe_page(): 범위를 벗어난 page를 1~total_pages로 보정
    - prev_page()/next_page(): 이전/다음 페이지 계산(입력 타입이 str/None이어도 안전)
    """
    page_size: int = 6

    def total_pages(self, n_cols: int) -> int:
        """
            [페이지네이션 계산 유틸]
            여러 개의 그래프(컬럼)를 "page_size개씩" 끊어서 보여주기 위한 계산용 도우미.

            - total_pages(): 전체 페이지 수 계산
            - safe_page(): 범위를 벗어난 page를 1~total_pages로 보정
            - prev_page()/next_page(): 이전/다음 페이지 계산(입력 타입이 str/None이어도 안전)
        """
        return max(1, math.ceil(max(0, int(n_cols)) / self.page_size))

    def safe_page(self, page: int, total_pages: int) -> int:
        """
            [페이지 번호 보정]
            page 값이 범위를 벗어나면 1~total_pages 범위로 clamp한다.

            Args:
                page: 현재 페이지(사용자 입력/상태값)
                total_pages: 전체 페이지 수

            Returns:
                보정된 페이지 번호
        """
        if total_pages <= 0:
            return 1
        return max(1, min(int(page), int(total_pages)))

    def prev_page(self, page: Union[int, str, None]) -> int:
        """
            [페이지 번호 보정]
            page 값이 범위를 벗어나면 1~total_pages 범위로 clamp한다.

            Args:
                page: 현재 페이지(사용자 입력/상태값)
                total_pages: 전체 페이지 수

            Returns:
                보정된 페이지 번호
        """
        try:
            return max(1, int(page) - 1)
        except Exception:
            return 1

    def next_page(self, page: Union[int, str, None], n_cols: int) -> int:
        """
            [이전 페이지]
            page가 str/None일 수 있으므로 안전하게 int 변환 후 -1 한다.

            Args:
                page: 현재 페이지

            Returns:
                이전 페이지(최소 1)
        """
        try:
            p = int(page)
        except Exception:
            p = 1
        total = self.total_pages(n_cols)
        return min(total, p + 1)


# ============================================================
# 2) Train Monitor: results.csv 6-plots + compare overlay
# ============================================================
class TrainResultsPlotter:
    """
        [다음 페이지]
        page가 str/None일 수 있으므로 안전하게 int 변환 후 +1 한다.
        단, total_pages를 넘지 않도록 clamp한다.

        Args:
            page: 현재 페이지
            n_cols: 전체 컬럼(그래프) 개수

        Returns:
            다음 페이지(최대 total_pages)
    """

    def __init__(
        self,
        runs_dir: str,
        metric_cols: List[str],
        loss_cols: List[str],
        paging: Optional[Paging] = None,
    ):
        """
            Args:
                runs_dir: Ultralytics runs 디렉토리 루트
                metric_cols: metrics 모드에서 보여줄 컬럼명 목록
                loss_cols: loss 모드에서 보여줄 컬럼명 목록
                paging: 페이지네이션 설정(없으면 기본 Paging(page_size=6))
        """
        self.runs_dir = runs_dir
        self.metric_cols = list(metric_cols or [])
        self.loss_cols = list(loss_cols or [])
        self.paging = paging or Paging(page_size=6)

    # -----------------------
    # small helpers
    # -----------------------
    @staticmethod
    def _to_float(v):
        """
            [유틸] 다양한 타입(v)을 float로 변환(불가하면 None).
            - None / "" 는 None 처리
        """
        try:
            if v is None or v == "":
                return None
            return float(v)
        except Exception:
            return None

    def _cols_for_mode(self, mode):
        if mode == "all":
            return self.metric_cols + self.loss_cols
        elif mode == "metrics":
            return self.metric_cols
        elif mode == "loss":
            return self.loss_cols
        else:
            return self.metric_cols + self.loss_cols

    def resolve_primary_csv(self, csv_path: str) -> str:
        """
            [primary csv 경로 확정]
            - csv_path가 주어지면 그대로 사용
            - 비어있으면 runs_dir에서 최신 results.csv를 자동 탐색하여 사용

            Returns:
                primary results.csv 경로(없으면 "")
        """
        path = (csv_path or "").strip()
        if path:
            return path
        return find_latest_results_csv(self.runs_dir) or ""

    def resolve_compare_csv(self, compare_csv_path: str, compare_enabled: bool) -> str:
        """
            [compare csv 경로 확정]
            - compare_enabled=False이면 비교 off("") 처리
            - True이면 compare_csv_path를 그대로 사용(빈값 가능)

            Returns:
                compare results.csv 경로(비교 off면 "")
        """
        if not compare_enabled:
            return ""
        return (compare_csv_path or "").strip()

    # -----------------------
    # plotting primitives
    # -----------------------
    def make_single_series_plot(self, rows: List[dict], col: str, title: str):
        """
            [단일 시계열 그래프 생성]
            rows(=results.csv 파싱 결과)에서 특정 col 컬럼을 epoch-x축으로 plot한다.

            Args:
                rows: ensure_epoch(read_results_csv(...)) 형태의 list[dict]
                col: 시각화할 컬럼명
                title: 그래프 타이틀

            Returns:
                matplotlib.figure.Figure
                - 데이터/컬럼이 없으면 안내 문구를 넣은 빈 figure 반환
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title(title)

        if not rows or (col not in rows[0]):
            ax.text(0.5, 0.5, f"'{col}' column not found", ha="center", va="center")
            ax.axis("off")
            plt.close(fig)
            return fig

        x = [r.get("epoch", i) for i, r in enumerate(rows)]
        y = [r.get(col, None) for r in rows]

        xx, yy = [], []
        for xi, yi in zip(x, y):
            try:
                if yi is None or yi == "":
                    continue
                xx.append(float(xi))
                yy.append(float(yi))
            except Exception:
                continue

        if not xx:
            ax.text(0.5, 0.5, f"No valid values for '{col}'", ha="center", va="center")
            ax.axis("off")
            plt.close(fig)
            return fig

        ax.plot(xx, yy)
        ax.set_xlabel("epoch")
        ax.grid(True)
        plt.close(fig)
        return fig

    def make_single_series_plot_compare(self, rows_primary, rows_compare, col, title):
        """
            [비교 오버레이 그래프 생성]
            한 plot에
              - primary(실선)
              - compare(점선)
            를 겹쳐 그린다.

            Args:
                rows_primary: primary results.csv rows
                rows_compare: compare results.csv rows
                col: 시각화할 컬럼명
                title: 그래프 타이틀

            Returns:
                matplotlib.figure.Figure
                - 둘 다 데이터가 없으면 안내 문구를 넣은 빈 figure 반환
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title(title)

        # primary
        if rows_primary and (col in rows_primary[0]):
            epochs_p = [int(float(r.get("epoch", i))) for i, r in enumerate(rows_primary)]
            ys_p = [self._to_float(r.get(col)) for r in rows_primary]
            if any(v is not None for v in ys_p):
                ax.plot(epochs_p, ys_p, label="primary")

        # compare
        if rows_compare and (col in rows_compare[0]):
            epochs_c = [int(float(r.get("epoch", i))) for i, r in enumerate(rows_compare)]
            ys_c = [self._to_float(r.get(col)) for r in rows_compare]
            if any(v is not None for v in ys_c):
                ax.plot(epochs_c, ys_c, linestyle="--", label="compare")

        if len(ax.lines) == 0:
            ax.text(0.5, 0.5, f"No data for '{col}'", ha="center", va="center")
            ax.axis("off")
        else:
            ax.set_xlabel("epoch")
            ax.grid(True)
            ax.legend(loc="best")

        plt.close(fig)
        return fig

    # -----------------------
    # main refresh API
    # -----------------------
    def refresh_6plots_compare_manual(
         self,
         csv_path: str,
         page_now: int,
         mode: str,
         compare_csv_path: str = "",
         compare_enabled: bool = False,
    ):
        """
            [UI timer.tick 메인 엔트리포인트]
            6개 plot + 상태 메시지 + timer update + page 값을 반환한다.
            (app.py에서 outputs를 6개 Plot + msg + timer + page로 매칭)

            동작 흐름:
              1) primary/compare csv 경로 확정
              2) mode에 따른 컬럼 목록 결정
              3) page 범위 보정
              4) primary csv가 없으면 (대기용) 빈 plot 6개 반환
              5) csv 로드 후 epoch 보정(ensure_epoch)
              6) 현재 페이지에 해당하는 6개 컬럼 plot 생성(비교면 오버레이)
              7) 상태 메시지와 함께 반환

            Returns:
                (*figs(6), msg, timer_update, page_now)
        """
        # timer_update = gr.update(value=float(refresh_s))

        # paths
        primary_path = self.resolve_primary_csv(csv_path)
        # comp_path = self.resolve_compare_csv(compare_csv_path, compare_enabled)
        comp_path = (compare_csv_path or "").strip()

        cols = self._cols_for_mode(mode)
        total_pages = self.paging.total_pages(len(cols))
        page_now = self.paging.safe_page(page_now, total_pages)

        # primary 없으면: 대기 figure 6개 생성
        if not primary_path or not os.path.exists(primary_path):
            figs = []
            for i in range(self.paging.page_size):
                idx = (page_now - 1) * self.paging.page_size + i
                col = cols[idx] if idx < len(cols) else "(empty)"
                figs.append(self.make_single_series_plot([], col, col))
            # return (*figs, "마지막 갱신: (대기 중) — results.csv 없음", timer_update, page_now)
            return *figs, page_now, "마지막 갱신: (대기 중) — results.csv 없음"

        # load data
        rows_primary = ensure_epoch(read_results_csv(primary_path))

        rows_compare = []
        if comp_path and os.path.exists(comp_path):
            rows_compare = ensure_epoch(read_results_csv(comp_path))

        # plot 6
        figs = []
        for i in range(self.paging.page_size):
            idx = (page_now - 1) * self.paging.page_size + i
            if idx < len(cols):
                col = cols[idx]
                if rows_compare:
                    figs.append(self.make_single_series_plot_compare(rows_primary, rows_compare, col, col))
                else:
                    figs.append(self.make_single_series_plot(rows_primary, col, col))
            else:
                figs.append(self.make_single_series_plot([], "(empty)", "(empty)"))

        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        comp_note = f"\ncompare: `{comp_path}`" if rows_compare else "\ncompare: (off)"
        msg = f"마지막 갱신: {ts}  \nprimary: `{primary_path}`{comp_note}  \n({mode} {page_now}/{total_pages})"

        # return (*figs, msg, timer_update, page_now)
        return *figs, page_now, msg


# ============================================================
# 3) Epoch Conf Monitor: scan epoch*_*.txt / best_*.txt / last_*.txt
# ============================================================
class EpochConfMonitor:
    """
        [Epoch Conf txt 모니터링 서비스]

        목적:
          - weights 디렉토리의 conf txt 파일명을 스캔해서
            epoch/best/last별 conf 추세를 plot으로 보여준다.
          - 스캔 결과를 요약 csv(epoch_conf_summary.csv)로 저장한다.

        txt 파일명 규칙(예):
          - epoch200_0.8670.txt
          - best_0.9001.txt
          - last_0.7555.txt
    """

    EPOCH_RE = re.compile(r"^epoch(?P<epoch>\d+)_+(?P<conf>\d+(?:\.\d+)?)\.txt$", re.IGNORECASE)
    BEST_RE  = re.compile(r"^best_+(?P<conf>\d+(?:\.\d+)?)\.txt$", re.IGNORECASE)
    LAST_RE  = re.compile(r"^last_+(?P<conf>\d+(?:\.\d+)?)\.txt$", re.IGNORECASE)

    @staticmethod
    def normalize_dir_value(v: Union[str, List[str], None]) -> str:
        """
            [입력 정규화]
            gradio 컴포넌트에서 값이 str로 오기도 하고 list[str]로 오기도 해서
            항상 단일 문자열로 변환한다.

            Args:
                v: gradio 입력값(문자열 또는 리스트)

            Returns:
                단일 경로 문자열(없으면 "")
        """
        if v is None:
            return ""
        if isinstance(v, list):
            return str(v[0]) if v else ""
        return str(v)

    def scan_epoch_conf_txts(self, weights_dir_value: Union[str, List[str], None]) -> Tuple[pd.DataFrame, str]:
        """
            [conf txt 스캔]
            weights_dir 하위의 *.txt 파일 중,
            epoch/best/last 규칙에 매칭되는 파일을 찾아 DataFrame으로 만든다.

            입력:
                weights_dir_value: weights 디렉토리 경로(str 또는 list[str])

            출력:
                df: 스캔 결과 DataFrame
                    columns:
                      - type: "epoch" | "best" | "last"
                      - epoch: int 또는 None(best/last)
                      - conf: float
                      - filename: 파일명
                      - filepath: 전체 경로
                      - key: 정렬용 키(epoch는 epoch, best/last는 큰 값)
                msg: 상태 메시지(성공/실패 사유)

            정렬:
                - epoch는 epoch 숫자 기준
                - best/last는 맨 뒤에 배치되도록 key를 크게 설정
        """
        weights_dir = self.normalize_dir_value(weights_dir_value).strip()
        cols = ["type", "epoch", "conf", "filename", "filepath", "key"]

        if not weights_dir or not os.path.isdir(weights_dir):
            return pd.DataFrame(columns=cols), f"경로가 유효하지 않음: {weights_dir}"

        txt_paths = sorted(glob.glob(os.path.join(weights_dir, "*.txt")))
        rows = []

        for p in txt_paths:
            fn = os.path.basename(p)

            m = self.EPOCH_RE.match(fn)
            if m:
                epoch = int(m.group("epoch"))
                conf = float(m.group("conf"))
                rows.append({
                    "type": "epoch",
                    "epoch": epoch,
                    "conf": conf,
                    "filename": fn,
                    "filepath": p,
                    "key": epoch,
                })
                continue

            m = self.BEST_RE.match(fn)
            if m:
                conf = float(m.group("conf"))
                rows.append({
                    "type": "best",
                    "epoch": None,
                    "conf": conf,
                    "filename": fn,
                    "filepath": p,
                    "key": 10**9,
                })
                continue

            m = self.LAST_RE.match(fn)
            if m:
                conf = float(m.group("conf"))
                rows.append({
                    "type": "last",
                    "epoch": None,
                    "conf": conf,
                    "filename": fn,
                    "filepath": p,
                    "key": 10**9 + 1,
                })
                continue

        df = pd.DataFrame(rows, columns=cols)
        if df.empty:
            return df, f"매칭되는 파일이 없음: {weights_dir} (예: epoch200_0.8670.txt / best_0.9001.txt)"

        df = df.sort_values(["key", "type"]).reset_index(drop=True)
        return df, f"스캔 완료: {len(df)}개 파일"

    def make_epoch_conf_plot(self, df: pd.DataFrame, title: str = "Weights vs Conf"):
        """
            [epoch/best/last conf 추세 plot 생성]
            scan 결과 DataFrame을 받아 x축 라벨을 epochN/best/last로 구성하고
            y축에 conf 값을 표시한다.

            입력:
                df: scan_epoch_conf_txts() 결과 DF
                title: 그래프 타이틀

            출력:
                matplotlib.figure.Figure
                - df가 비어있으면 "No Data" 안내 plot 반환
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)

        if df is None or df.empty:
            ax.text(0.5, 0.5, "No Data", ha="center", va="center")
            ax.axis("off")
            plt.close(fig)
            return fig

        x_labels = []
        y_vals = []

        for _, r in df.iterrows():
            t = r["type"]
            if t == "epoch":
                x_labels.append(f"epoch{int(r['epoch'])}")
            else:
                x_labels.append(str(t))
            y_vals.append(float(r["conf"]))

        ax.plot(range(len(x_labels)), y_vals, marker="o")
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
        ax.set_ylabel("mean conf")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plt.close(fig)
        return fig

    def update_epoch_conf_view(self, weights_dir_value, run_title: str):
        """
            [UI 버튼 클릭 핸들러]
            UI에서 "스캔 & 그래프 업데이트" 버튼을 누르면 호출된다.

            수행:
              1) weights_dir에서 txt 스캔
              2) plot 생성
              3) 요약 csv(epoch_conf_summary.csv) 저장
              4) UI 컴포넌트 출력(Plot/Dataframe/File/Status)을 위한 값 반환

            Returns:
                (fig, out_df, out_csv_path, status_msg)

                - 스캔 실패/데이터 없음:
                    (None, empty_df, None, msg)
        """
        df, msg = self.scan_epoch_conf_txts(weights_dir_value)

        if df is None or df.empty:
            return None, df, None, msg

        title = run_title.strip() if run_title and run_title.strip() else "Weights vs Conf"
        fig = self.make_epoch_conf_plot(df, title=title)

        weights_dir = self.normalize_dir_value(weights_dir_value).strip()
        out_csv = os.path.join(weights_dir, "epoch_conf_summary.csv")
        out_df = df[["type", "epoch", "conf", "filename"]].copy()
        out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

        return fig, out_df, out_csv, msg + f" | CSV 저장: {out_csv}"

    def build_ui(self, default_weights_dir: str = ""):
        """
            [UI 빌더]
            app.py에서 Tab 내부에 붙이기 위한 Gradio UI 컴포넌트 묶음을 생성한다.

            UI 구성:
              - weights_dir 입력(Textbox)
              - run_title 입력(Textbox)
              - refresh 버튼
              - plot(Plot)
              - table(Dataframe)
              - csv 다운로드(File)
              - status(Textbox)

            Returns:
                UI 컴포넌트들을 dict로 묶어 반환(필요시 app.py에서 추가 제어 가능)
        """
        gr.Markdown("### Weights별(conf txt) 추세 모니터링")
        with gr.Row():
            weights_dir = gr.Textbox(
                label="weights 폴더 경로(서버)",
                value=default_weights_dir,
                placeholder="예) /home/.../runs/segment/demo_exp22/weights",
            )
            run_title = gr.Textbox(
                label="그래프 타이틀(옵션)",
                value="",
                placeholder="예) demo_exp22 / segment 비교 등",
            )

        btn_refresh = gr.Button("스캔 & 그래프 업데이트", variant="primary")

        plot_img = gr.Plot(label="Weights vs Conf")
        table = gr.Dataframe(label="파일 목록", interactive=False)
        csv_file = gr.File(label="CSV 다운로드(요약)")
        status = gr.Textbox(label="상태", interactive=False)

        btn_refresh.click(
            fn=self.update_epoch_conf_view,
            inputs=[weights_dir, run_title],
            outputs=[plot_img, table, csv_file, status],
        )

        return {
            "weights_dir": weights_dir,
            "run_title": run_title,
            "btn_refresh": btn_refresh,
            "plot": plot_img,
            "table": table,
            "csv_file": csv_file,
            "status": status,
        }


# ============================================================
# 4) Backward-compatible function wrappers (기존 호출 유지용)
# ============================================================

# paging wrappers
_paging = Paging(page_size=6)

def prev_page(p):
    """
        [호환용 래퍼] 이전 페이지 계산
    """
    return _paging.prev_page(p)

def next_page(p, cols):
    """
        [호환용 래퍼] 다음 페이지 계산
        cols는 컬럼 리스트로 들어오므로 len(cols)로 총 개수 계산 후 next_page에 전달
    """
    return _paging.next_page(p, len(cols))

def safe_page(p: int, total_pages: int) -> int:
    """
        [호환용 래퍼] page 범위 보정
    """
    return _paging.safe_page(p, total_pages)


# plotter singleton (기존 코드에서 바로 쓰던 스타일 유지 가능)
_default_plotter: Optional[TrainResultsPlotter] = None

def get_plotter(runs_dir: str, metric_cols: List[str], loss_cols: List[str]) -> TrainResultsPlotter:
    """
        [plotter 싱글톤 getter]
        기존 코드가 '매번 plotter를 만들지 않고' 재사용하던 패턴을 유지하기 위한 함수.

        주의:
          - runs_dir / cols가 런타임에 바뀔 수 있다면, 캐시 무효화(재생성) 조건을 넣는 게 안전
          - 대부분 고정이라면 싱글톤으로 두어도 충분히 동작
    """
    global _default_plotter
    # runs_dir/cols가 바뀌면 새로 만드는 게 안전하지만,
    # 대부분 고정이라면 싱글톤으로 둬도 OK.
    if _default_plotter is None:
        _default_plotter = TrainResultsPlotter(runs_dir, metric_cols, loss_cols, paging=_paging)
    return _default_plotter


def refresh_6plots_compare_manual(
    csv_path: str,
    page_now: int,
    mode: str,
    runs_dir: str,
    metric_cols: list[str],
    loss_cols: list[str],
    # compare_csv_path: str = "",
    # compare_enabled: bool = True,
):
    """
        [호환용 래퍼]
        기존 app.py에서 호출하던 refresh_6plots_compare 시그니처를 유지한다.
        내부 구현은 TrainResultsPlotter.refresh_6plots_compare()에 위임한다.

        Returns:
            (*figs(6), msg, timer_update, page_now)
    """
    plotter = get_plotter(runs_dir, metric_cols, loss_cols)
    return plotter.refresh_6plots_compare_manual(
        csv_path=csv_path,
        page_now=page_now,
        mode=mode,
        # compare_csv_path=compare_csv_path,
        # compare_enabled=compare_enabled,
    )


# epoch conf monitor UI builder wrapper
_epoch_monitor = EpochConfMonitor()

def build_epoch_conf_monitor_ui(default_weights_dir: str = ""):
    return _epoch_monitor.build_ui(default_weights_dir=default_weights_dir)
