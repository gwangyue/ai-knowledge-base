"""GitHub API 工具函数（新版本）。"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import TypedDict


class GitHubRepoInfo(TypedDict):
    """GitHub 仓库基础信息。"""

    stars: int
    forks: int
    description: str | None


def get_repo_basic_info(
    owner: str,
    repo: str,
    token: str | None = None,
    timeout: int = 10,
) -> GitHubRepoInfo:
    """获取指定 GitHub 仓库的基础信息。

    Args:
        owner: 仓库所属用户或组织名。
        repo: 仓库名。
        token: 可选的 GitHub Token；未传入时会尝试读取环境变量 GITHUB_TOKEN。
        timeout: HTTP 请求超时时间，单位秒。

    Returns:
        包含 Star 数、Fork 数和描述的字典。

    Raises:
        ValueError: owner 或 repo 为空。
        RuntimeError: GitHub API 请求失败或返回异常结果。
    """

    if not owner or not owner.strip():
        raise ValueError("owner 不能为空")
    if not repo or not repo.strip():
        raise ValueError("repo 不能为空")

    owner = owner.strip()
    repo = repo.strip()
    auth_token = token or os.getenv("GITHUB_TOKEN", "")

    url = (
        "https://api.github.com/repos/"
        f"{urllib.parse.quote(owner)}/{urllib.parse.quote(repo)}"
    )
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "ai-knowledge-base",
    }
    if auth_token:
        headers["Authorization"] = f"token {auth_token}"

    request = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(
            f"GitHub API 请求失败: HTTP {exc.code} {exc.reason}. {detail}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"GitHub API 网络请求失败: {exc.reason}") from exc

    return {
        "stars": int(payload.get("stargazers_count", 0)),
        "forks": int(payload.get("forks_count", 0)),
        "description": payload.get("description"),
    }


if __name__ == "__main__":
    """命令行示例：获取 OpenAI 的 openai-python 仓库信息"""
    import sys

    try:
        info = get_repo_basic_info("openai", "openai-python")
        print(f"仓库: openai/openai-python")
        print(f"Stars: {info['stars']}")
        print(f"Forks: {info['forks']}")
        print(f"描述: {info['description'] or '(无描述)'}")
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)