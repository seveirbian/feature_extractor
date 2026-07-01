#!/usr/bin/env python3
"""对比两个特征 HDF5 文件在公共帧上的差异(用于验证流式 vs 非流式提取是否一致)。

用法:
    uv run python scripts/compare_features.py A.h5 B.h5
    uv run python scripts/compare_features.py A.h5 B.h5 --branch depth --block-size 128

按 frame_indices 取两文件的公共帧对齐后逐帧比较:
  - depth: 归一化逆深度 [0,1] 上的 mean|diff| / max|diff| / 逐像素相关
  - dino : 每帧特征的余弦相似度 + 相对 L2
  - pose : 9 维位姿的逐分量绝对差
给了 --block-size 时,对 depth 高亮"段边界附近"的帧(分段拼接最容易出问题的地方)。
"""
from __future__ import annotations

import argparse

import h5py
import numpy as np


def _read_branch(path: str, branch: str):
    """返回 (frame_indices, values, attrs) 或 None(该分支不存在)。"""
    key = {"depth": "inv_depth", "dino": "features", "pose": "se3_trajectory"}[branch]
    with h5py.File(path, "r") as f:
        if branch not in f or key not in f[branch]:
            return None
        fi = f[f"{branch}/frame_indices"][:].astype(np.int64)
        vals = f[f"{branch}/{key}"][:]
        attrs = dict(f[branch].attrs)
    if branch == "depth":
        scale = float(attrs.get("scale", 65535.0))
        vals = vals.astype(np.float32) / scale  # -> [0,1]
    else:
        vals = vals.astype(np.float32)
    return fi, vals, attrs


def _common(fa: np.ndarray, fb: np.ndarray):
    ia = {int(v): k for k, v in enumerate(fa)}
    ib = {int(v): k for k, v in enumerate(fb)}
    common = sorted(set(ia) & set(ib))
    return common, ia, ib


def _fmt_meta(tag: str, fi: np.ndarray, attrs: dict) -> str:
    contiguous = np.array_equal(fi, np.arange(int(fi[0]), int(fi[0]) + len(fi)))
    return (f"  {tag:10s} n={len(fi):<6d} range=[{int(fi[0])}..{int(fi[-1])}] "
            f"contiguous={contiguous} complete={attrs.get('complete', '<none>')}")


def compare_depth(common, A, B, ia, ib, block_size: int | None):
    md, mx, cr = [], [], []
    for c in common:
        a, b = A[ia[c]], B[ib[c]]
        d = np.abs(a - b)
        md.append(float(d.mean())); mx.append(float(d.max()))
        cr.append(float(np.corrcoef(a.ravel(), b.ravel())[0, 1]))
    md, mx, cr = np.array(md), np.array(mx), np.array(cr)
    print(f"  frames compared : {len(common)}")
    print(f"  mean|diff|      : avg={md.mean():.4f}  worst={md.max():.4f}")
    print(f"  max |diff|      : avg={mx.mean():.4f}  worst={mx.max():.4f}")
    print(f"  pixel corr      : avg={cr.mean():.4f}  worst={cr.min():.4f}")
    order = np.argsort(-md)[:5]
    print("  worst 5 frames by mean|diff|:")
    for k in order:
        bnd = ""
        if block_size:
            near = min(int(common[k]) % block_size, block_size - int(common[k]) % block_size)
            bnd = "  <-near segment boundary" if near <= 2 else ""
        print(f"    frame {common[k]:7d}: mean={md[k]:.4f} max={mx[k]:.4f} corr={cr[k]:.4f}{bnd}")
    if block_size:
        bmask = np.array([min(int(c) % block_size, block_size - int(c) % block_size) <= 2
                          for c in common])
        if bmask.any():
            print(f"  boundary frames : mean|diff| avg={md[bmask].mean():.4f} "
                  f"(vs interior avg={md[~bmask].mean():.4f})" if (~bmask).any() else "")


def compare_dino(common, A, B, ia, ib):
    cos, rl2 = [], []
    for c in common:
        a, b = A[ia[c]].ravel(), B[ib[c]].ravel()
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
        cos.append(float(np.dot(a, b) / denom))
        rl2.append(float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-8)))
    cos, rl2 = np.array(cos), np.array(rl2)
    print(f"  frames compared : {len(common)}")
    print(f"  cosine sim      : avg={cos.mean():.5f}  worst={cos.min():.5f}")
    print(f"  relative L2     : avg={rl2.mean():.5f}  worst={rl2.max():.5f}")


def _rot6d_to_matrix(r6d: np.ndarray) -> np.ndarray:
    """Gram-Schmidt the 6D representation (two columns) back to a 3x3 rotation."""
    a1, a2 = r6d[:3].astype(np.float64), r6d[3:].astype(np.float64)
    b1 = a1 / (np.linalg.norm(a1) + 1e-12)
    a2 = a2 - np.dot(b1, a2) * b1
    b2 = a2 / (np.linalg.norm(a2) + 1e-12)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1)  # columns are the basis vectors


def _pose9_to_center_and_rot(p9: np.ndarray):
    """9D [t(3), rot6d(6)] (frame-0-relative world-to-camera) -> (center, R)."""
    t = p9[:3].astype(np.float64)
    R = _rot6d_to_matrix(p9[3:])         # world-to-camera rotation of the relative pose
    center = -R.T @ t                    # camera center in the relative world frame
    return center, R


def compare_pose(common, A, B, ia, ib):
    from feature_extractor.extractors.pose_streaming import umeyama_similarity

    print(f"  frames compared : {len(common)}")
    # raw diff (will look large if the two runs use a different global scale)
    raw = np.array([np.abs(A[ia[c]] - B[ib[c]]) for c in common])
    print(f"  raw |diff| 9D   : avg={raw.mean():.4f}  worst={raw.max():.4f}  "
          f"(受全局尺度影响,仅参考)")

    ca = np.array([_pose9_to_center_and_rot(A[ia[c]])[0] for c in common])
    cb = np.array([_pose9_to_center_and_rot(B[ib[c]])[0] for c in common])
    Ra = [_pose9_to_center_and_rot(A[ia[c]])[1] for c in common]
    Rb = [_pose9_to_center_and_rot(B[ib[c]])[1] for c in common]

    # rotation geodesic error (scale-invariant)
    ang = []
    for r_a, r_b in zip(Ra, Rb):
        cos = (np.trace(r_a.T @ r_b) - 1.0) / 2.0
        ang.append(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
    ang = np.array(ang)
    print(f"  rotation err    : mean={ang.mean():.3f}°  worst={ang.max():.3f}°")

    # translation ATE after sim(3) alignment of B's centers onto A's
    if len(common) >= 3:
        s, R, t = umeyama_similarity(cb, ca)
        aligned = s * (cb @ R.T) + t
        err = np.linalg.norm(ca - aligned, axis=1)
        extent = np.linalg.norm(ca - ca.mean(0), axis=1).mean() + 1e-9
        print(f"  global scale B→A: {s:.4f}   (两次归一化尺度之比)")
        print(f"  translation ATE : rmse={np.sqrt((err**2).mean()):.4f}  worst={err.max():.4f}  "
              f"(占轨迹尺度 {100*np.sqrt((err**2).mean())/extent:.2f}%)")
    else:
        print("  translation ATE : 需要 ≥3 个公共帧")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file_a")
    ap.add_argument("file_b")
    ap.add_argument("--branch", choices=["depth", "dino", "pose"], default=None,
                    help="只比某个分支(默认:两文件共有的全部分支)")
    ap.add_argument("--block-size", type=int, default=None,
                    help="给出后对 depth 高亮段边界附近的帧")
    args = ap.parse_args()

    branches = [args.branch] if args.branch else ["dino", "depth", "pose"]
    any_done = False
    for br in branches:
        ra = _read_branch(args.file_a, br)
        rb = _read_branch(args.file_b, br)
        if ra is None or rb is None:
            if args.branch:
                print(f"[{br}] 跳过:至少一个文件没有该分支")
            continue
        fa, A, aa = ra
        fb, B, ab = rb
        common, ia, ib = _common(fa, fb)
        print(f"\n===== branch: {br} =====")
        print(_fmt_meta("A", fa, aa))
        print(_fmt_meta("B", fb, ab))
        if not common:
            print("  公共帧为 0 —— 两次采样的 frame_indices 没有交集,无法逐帧对比。")
            continue
        any_done = True
        if br == "depth":
            compare_depth(common, A, B, ia, ib, args.block_size)
        elif br == "dino":
            compare_dino(common, A, B, ia, ib)
        else:
            compare_pose(common, A, B, ia, ib)

    if not any_done:
        print("\n没有可比较的公共分支/帧。")


if __name__ == "__main__":
    main()
