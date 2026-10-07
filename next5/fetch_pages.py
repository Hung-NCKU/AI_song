"""Save the competition's overview pages (description / evaluation / rules) via the Kaggle API."""
from kaggle.api.kaggle_api_extended import KaggleApi
from common import WORK

api = KaggleApi()
api.authenticate()
out = WORK / "pages"
out.mkdir(exist_ok=True)
for p in api.competition_list_pages("datagame-2023"):
    name = getattr(p, "name", None) or "page"
    content = getattr(p, "content", "") or ""
    (out / f"{name}.md").write_text(content, encoding="utf-8")
    print(f"===== {name} ({len(content)} chars)\n{content}\n")
