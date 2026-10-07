"""List my submissions with scores and error messages."""
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
for s in api.competition_submissions("datagame-2023")[:12]:
    d = s.to_dict()
    print({k: d.get(k) for k in ("ref", "fileName", "date", "description", "status", "publicScore", "privateScore", "errorDescription")})
