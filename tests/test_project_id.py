import subprocess


from graph_mem.project_id import get_project_id, normalize_git_url


# Graphiti requires group_ids to be alphanumeric, dashes, or underscores —
# so normalize_git_url replaces any other character (dots, slashes, colons)
# with '_'. Dashes inside repo names are preserved.


def test_normalize_https_url():
    assert normalize_git_url("https://github.com/user/repo.git") == "github_com_user_repo"


def test_normalize_ssh_url():
    assert normalize_git_url("git@github.com:user/repo.git") == "github_com_user_repo"


def test_normalize_https_no_git_suffix():
    assert normalize_git_url("https://github.com/user/repo") == "github_com_user_repo"


def test_normalize_ssh_no_git_suffix():
    assert normalize_git_url("git@gitlab.com:org/project") == "gitlab_com_org_project"


def test_get_project_id_with_remote(tmp_path):
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
    subprocess.run(
        ["git", "remote", "add", "origin", "https://github.com/user/my-project.git"],
        cwd=tmp_path,
        capture_output=True,
    )
    project_id = get_project_id(str(tmp_path))
    assert project_id == "project_github_com_user_my-project"


def test_get_project_id_no_remote(tmp_path):
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
    project_id = get_project_id(str(tmp_path))
    assert project_id == f"project_{tmp_path.name}"


def test_get_project_id_no_git(tmp_path):
    project_id = get_project_id(str(tmp_path))
    assert project_id == f"project_{tmp_path.name}"
