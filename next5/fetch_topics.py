"""Dump discussion topics / writeups of the competition (incl. the 1st-place writeup) via the Kaggle API."""
from kaggle.api.kaggle_api_extended import KaggleApi
from common import WORK

api = KaggleApi()
api.authenticate()
out = WORK / "topics"
out.mkdir(exist_ok=True)
topics = api.competition_list_topics("datagame-2023").topics
for t in topics:
    resp = api.competition_list_topic_messages("datagame-2023", int(t.id), page_size=-1)
    msgs = api._flatten_topic_messages(resp.messages)
    text = f"# {t.title}\n{t.topic_url}\n\n"
    for m in msgs:
        d = m.to_dict() if hasattr(m, "to_dict") else {}
        body = d.get("rawMarkdown") or d.get("content") or d.get("message") or str(d)
        text += f"---\n**{d.get('authorName', '')}** {d.get('postDate', '')}\n\n{body}\n\n"
    (out / f"{t.id}.md").write_text(text, encoding="utf-8")
    print(t.id, t.title, len(msgs))
