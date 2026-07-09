"""Tests for file tree listing, raw file access, and folder creation endpoints."""

from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(name="client")
def files_client_alias(files_test_client: TestClient):
    return files_test_client


def test_list_files_returns_tree_with_readme_entries(
    client: TestClient, tmp_path: Path
):
    user_root = tmp_path / "users" / "user_test_user" / "user_data"

    ado_folder = user_root / "sample_data" / "ADO"
    ado_folder.mkdir(parents=True, exist_ok=True)
    (ado_folder / "README.md").write_text(
        "# ADO Citation\n\nSample citation text.", encoding="utf-8"
    )
    (ado_folder / "documents.csv").write_text("text\nhello", encoding="utf-8")
    nested_folder = ado_folder / "nested"
    nested_folder.mkdir(parents=True, exist_ok=True)
    (nested_folder / "nested.csv").write_text("text\ninside", encoding="utf-8")

    no_readme_folder = user_root / "sample_data" / "NoReadme"
    no_readme_folder.mkdir(parents=True, exist_ok=True)
    (no_readme_folder / "items.csv").write_text("text\nworld", encoding="utf-8")

    (user_root / "user_file.csv").write_text("text\nprivate", encoding="utf-8")

    response = client.get("/api/files/")
    assert response.status_code == 200

    payload = response.json()
    assert isinstance(payload, list)

    sample_root = next(item for item in payload if item["name"] == "sample_data")
    assert sample_root["type"] == "directory"

    ado_node = next(item for item in sample_root["children"] if item["name"] == "ADO")
    assert ado_node == {
        "name": "ADO",
        "path": "sample_data/ADO",
        "type": "directory",
        "children": ado_node["children"],
    }

    readme_node = next(
        item for item in ado_node["children"] if item["name"] == "README.md"
    )
    assert readme_node == {
        "name": "README.md",
        "path": "sample_data/ADO/README.md",
        "type": "file",
        "size": (ado_folder / "README.md").stat().st_size,
    }

    documents_node = next(
        item for item in ado_node["children"] if item["name"] == "documents.csv"
    )
    assert documents_node["type"] == "file"
    assert documents_node["path"] == "sample_data/ADO/documents.csv"
    assert documents_node["size"] > 0

    nested_node = next(
        item for item in ado_node["children"] if item["name"] == "nested"
    )
    assert nested_node["type"] == "directory"
    assert nested_node["path"] == "sample_data/ADO/nested"
    assert nested_node["children"] == [
        {
            "name": "nested.csv",
            "path": "sample_data/ADO/nested/nested.csv",
            "type": "file",
            "size": (nested_folder / "nested.csv").stat().st_size,
        }
    ]

    no_readme_node = next(
        item for item in sample_root["children"] if item["name"] == "NoReadme"
    )
    assert no_readme_node["type"] == "directory"
    assert "readme_path" not in no_readme_node

    user_data = next(item for item in payload if item["name"] == "user_file.csv")
    assert user_data == {
        "name": "user_file.csv",
        "path": "user_file.csv",
        "type": "file",
        "size": (user_root / "user_file.csv").stat().st_size,
    }


def test_list_files_serializes_empty_directory_children_as_empty_list(
    client: TestClient,
):
    with patch(
        "ldaca_wordflow.api.files.crud._build_file_tree",
        return_value=[
            {
                "name": "Empty Folder",
                "path": "Empty Folder",
                "type": "directory",
            }
        ],
    ):
        response = client.get("/api/files/")

    assert response.status_code == 200
    assert response.json() == [
        {
            "name": "Empty Folder",
            "path": "Empty Folder",
            "type": "directory",
            "children": [],
        }
    ]


def test_raw_file_returns_markdown_content(client: TestClient, tmp_path: Path):
    user_root = tmp_path / "users" / "user_test_user" / "user_data"
    readme_path = user_root / "sample_data" / "ADO" / "README.md"
    readme_path.parent.mkdir(parents=True, exist_ok=True)
    readme_path.write_text("# ADO Citation\n\nSample citation text.", encoding="utf-8")

    response = client.get(
        "/api/files/raw", params={"path": "sample_data/ADO/README.md"}
    )

    assert response.status_code == 200
    assert response.text == "# ADO Citation\n\nSample citation text."
    assert response.headers["content-type"].startswith("text/markdown")


def test_raw_file_rejects_paths_outside_user_data(client: TestClient):
    response = client.get("/api/files/raw", params={"path": "../outside.md"})

    assert response.status_code == 400
    assert response.json()["message"] == "Invalid file path"


def test_create_folder_creates_root_and_nested_directories(
    client: TestClient, tmp_path: Path
):
    user_root = tmp_path / "users" / "user_test_user" / "user_data"
    (user_root / "sample_data").mkdir(parents=True, exist_ok=True)

    root_response = client.post(
        "/api/files/folders", json={"name": "Research Notes", "parent_path": ""}
    )
    assert root_response.status_code == 200
    assert root_response.json() == {
        "message": "Folder created",
        "path": "Research Notes",
    }
    assert (user_root / "Research Notes").is_dir()

    nested_response = client.post(
        "/api/files/folders",
        json={"name": "Drafts", "parent_path": "sample_data"},
    )
    assert nested_response.status_code == 200
    assert nested_response.json() == {
        "message": "Folder created",
        "path": "sample_data/Drafts",
    }
    assert (user_root / "sample_data" / "Drafts").is_dir()


def test_create_folder_rejects_invalid_names(client: TestClient):
    response = client.post(
        "/api/files/folders", json={"name": "../escape", "parent_path": ""}
    )

    assert response.status_code == 400
    assert response.json()["message"] == "Invalid folder name: name cannot contain '..'"


def test_move_file_moves_into_target_directory(client: TestClient, tmp_path: Path):
    user_root = tmp_path / "users" / "user_test_user" / "user_data"
    source_dir = user_root / "incoming"
    target_dir = user_root / "sample_data" / "Hansard"
    source_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    source_file = source_dir / "economy_agenda.csv"
    source_file.write_text("text\nhello", encoding="utf-8")

    response = client.post(
        "/api/files/move",
        json={
            "source_path": "incoming/economy_agenda.csv",
            "target_directory_path": "sample_data/Hansard",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "message": "File moved",
        "path": "sample_data/Hansard/economy_agenda.csv",
    }
    assert not source_file.exists()
    assert (target_dir / "economy_agenda.csv").is_file()
