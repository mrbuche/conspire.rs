"""Fetch and merge GitHub contributors across the conspire repos into one dedup'd grid."""

import json
import os
import urllib.request

REPOS = [
    "mrbuche/conspire",
    "mrbuche/conspire.rs",
    "mrbuche/conspire.py",
]

AVATAR_SIZE = 64


def fetch_contributors(repo):
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "conspire-book"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    contributors = []
    page = 1
    while True:
        url = (
            f"https://api.github.com/repos/{repo}/contributors?per_page=100&page={page}"
        )
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request) as response:
            batch = json.loads(response.read())
        if not batch:
            break
        contributors.extend(batch)
        page += 1
    return contributors


def main():
    merged = {}
    for repo in REPOS:
        try:
            contributors = fetch_contributors(repo)
        except urllib.error.HTTPError as error:
            print(f"<!-- failed to fetch contributors for {repo}: {error} -->")
            continue
        for contributor in contributors:
            if contributor.get("type") == "Bot":
                continue
            login = contributor["login"]
            entry = merged.setdefault(
                login,
                {
                    "login": login,
                    "avatar_url": contributor["avatar_url"],
                    "html_url": contributor["html_url"],
                    "contributions": 0,
                },
            )
            entry["contributions"] += contributor["contributions"]

    ordered = sorted(
        merged.values(), key=lambda entry: entry["contributions"], reverse=True
    )

    lines = ['<div style="display: flex; flex-wrap: wrap; gap: 4px;">']
    for entry in ordered:
        avatar = f"{entry['avatar_url']}&s={AVATAR_SIZE}"
        lines.append(
            f'<a href="{entry["html_url"]}" title="{entry["login"]} ({entry["contributions"]} contributions)">'
            f'<img src="{avatar}" width="{AVATAR_SIZE}" height="{AVATAR_SIZE}" style="border-radius: 50%;"/>'
            f"</a>"
        )
    lines.append("</div>")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
