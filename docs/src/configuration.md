# Configuration

> **Note:** The configuration format is not finalized. Examples below show the intended direction.

```yaml
conference:
  name: "KubeCon EU 2025"
  schedule: "schedule.yaml"
  recordings:
    youtube_playlist: "https://www.youtube.com/playlist?list=PLj6h78..."

focus:
  topics:
    - "platform engineering"
    - "observability"
    - "security"

output:
  directory: "reports/"
```

## Conference Source

| Field          | Description                                                              |
|----------------|--------------------------------------------------------------------------|
| `name`         | Display name used in report headers.                                     |
| `schedule`     | Path to schedule data file (titles, abstracts, speakers, tracks).        |
| `recordings`   | YouTube playlist URL or directory of pre-downloaded transcripts.         |

## Focus Areas

| Field    | Description                                      |
|----------|--------------------------------------------------|
| `topics` | List of topics or keywords to prioritize.        |

Topics influence cluster ranking and how insights are prioritized in the recording analysis. When omitted, the tool produces a neutral analysis.

## Output

| Field       | Description                           | Default    |
|-------------|---------------------------------------|------------|
| `directory` | Directory for generated report files. | `reports/` |
