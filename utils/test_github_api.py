"""GitHub API 工具函数的单元测试。"""

import json
import urllib.error
from unittest.mock import Mock, patch

import pytest

from github_api import GitHubRepoInfo, get_repo_basic_info


class TestGetRepoBasicInfo:
    """测试 get_repo_basic_info 函数。"""

    @pytest.fixture
    def mock_response(self):
        """模拟成功的 GitHub API 响应。"""
        return {
            "stargazers_count": 12345,
            "forks_count": 678,
            "description": "A test repository for unit testing.",
            "full_name": "testowner/testrepo",
            "html_url": "https://github.com/testowner/testrepo",
        }

    @patch("urllib.request.urlopen")
    def test_success_without_token(self, mock_urlopen, mock_response):
        """测试成功调用（无 token）。"""
        mock_resp = Mock()
        mock_resp.read.return_value = json.dumps(mock_response).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        info = get_repo_basic_info("testowner", "testrepo", token=None)

        assert isinstance(info, dict)
        assert info["stars"] == 12345
        assert info["forks"] == 678
        assert info["description"] == "A test repository for unit testing."

        call_args = mock_urlopen.call_args[0][0]
        assert call_args.full_url == "https://api.github.com/repos/testowner/testrepo"
        assert call_args.headers.get("Authorization") is None
        assert call_args.headers.get("Accept") == "application/vnd.github.v3+json"
        # Request 对象将头部键转换为 "User-agent"（首字母大写）
        assert call_args.headers.get("User-agent") == "ai-knowledge-base"

    @patch("urllib.request.urlopen")
    def test_success_with_token(self, mock_urlopen, mock_response):
        """测试成功调用（提供 token）。"""
        mock_resp = Mock()
        mock_resp.read.return_value = json.dumps(mock_response).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        info = get_repo_basic_info("testowner", "testrepo", token="ghp_testtoken")

        assert info["stars"] == 12345
        assert info["forks"] == 678

        call_args = mock_urlopen.call_args[0][0]
        assert call_args.headers["Authorization"] == "token ghp_testtoken"

    @patch("urllib.request.urlopen")
    def test_success_with_env_token(self, mock_urlopen, mock_response, monkeypatch):
        """测试成功调用（从环境变量读取 token）。"""
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_envtoken")
        mock_resp = Mock()
        mock_resp.read.return_value = json.dumps(mock_response).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        info = get_repo_basic_info("testowner", "testrepo", token=None)

        assert info["stars"] == 12345
        call_args = mock_urlopen.call_args[0][0]
        assert call_args.headers["Authorization"] == "token ghp_envtoken"

    @patch("urllib.request.urlopen")
    def test_description_is_none(self, mock_urlopen):
        """测试描述字段为 null 的情况。"""
        mock_response = {
            "stargazers_count": 100,
            "forks_count": 20,
            "description": None,
        }
        mock_resp = Mock()
        mock_resp.read.return_value = json.dumps(mock_response).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        info = get_repo_basic_info("some", "repo")
        assert info["description"] is None

    @patch("urllib.request.urlopen")
    def test_missing_fields_default_to_zero(self, mock_urlopen):
        """测试响应缺少某些字段时使用默认值。"""
        mock_response = {
            "stargazers_count": 999,
            "description": "No forks field",
        }
        mock_resp = Mock()
        mock_resp.read.return_value = json.dumps(mock_response).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        info = get_repo_basic_info("test", "repo")
        assert info["stars"] == 999
        assert info["forks"] == 0
        assert info["description"] == "No forks field"

    def test_empty_owner_raises_valueerror(self):
        """测试 owner 为空字符串时抛出 ValueError。"""
        with pytest.raises(ValueError, match="owner 不能为空"):
            get_repo_basic_info("", "repo")

        with pytest.raises(ValueError, match="owner 不能为空"):
            get_repo_basic_info("   ", "repo")

    def test_empty_repo_raises_valueerror(self):
        """测试 repo 为空字符串时抛出 ValueError。"""
        with pytest.raises(ValueError, match="repo 不能为空"):
            get_repo_basic_info("owner", "")

        with pytest.raises(ValueError, match="repo 不能为空"):
            get_repo_basic_info("owner", "   ")

    @patch("urllib.request.urlopen")
    def test_http_error_raises_runtimeerror(self, mock_urlopen):
        """测试 HTTP 错误（如 404）时抛出 RuntimeError。"""
        error = urllib.error.HTTPError(
            url="https://api.github.com/repos/nonexistent/repo",
            code=404,
            msg="Not Found",
            hdrs={},
            fp=None,
        )
        error.read = lambda: b'{"message": "Not Found"}'
        mock_urlopen.side_effect = error

        with pytest.raises(RuntimeError, match="GitHub API 请求失败: HTTP 404 Not Found"):
            get_repo_basic_info("nonexistent", "repo")

    @patch("urllib.request.urlopen")
    def test_url_error_raises_runtimeerror(self, mock_urlopen):
        """测试网络错误（如超时）时抛出 RuntimeError。"""
        mock_urlopen.side_effect = urllib.error.URLError(reason="timed out")

        with pytest.raises(RuntimeError, match="GitHub API 网络请求失败: timed out"):
            get_repo_basic_info("owner", "repo")

    def test_owner_repo_trimmed(self):
        """测试 owner 和 repo 参数两端的空格被去除。"""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = Mock()
            mock_resp.read.return_value = json.dumps({
                "stargazers_count": 1,
                "forks_count": 0,
                "description": "",
            }).encode("utf-8")
            mock_urlopen.return_value.__enter__.return_value = mock_resp

            get_repo_basic_info("  testowner  ", "  testrepo  ")

            call_args = mock_urlopen.call_args[0][0]
            assert call_args.full_url == "https://api.github.com/repos/testowner/testrepo"


if __name__ == "__main__":
    """允许直接运行测试（不依赖 pytest）。"""
    import sys
    sys.exit(pytest.main([__file__, "-v"]))