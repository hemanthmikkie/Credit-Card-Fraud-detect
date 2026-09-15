"""Generate a starter Power BI PBIX template archive."""

import zipfile
import json
from pathlib import Path

pbix_path = Path(r"c:\10k\data science projects\Credit_Card_Fraud_Detection\dashboard\Fraud_Detection_Dashboard.pbix")

content_types_xml = (
    '<?xml version="1.0" encoding="utf-8"?>\n'
    '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">\n'
    '  <Default Extension="json" ContentType="application/json" />\n'
    '  <Default Extension="xml" ContentType="application/xml" />\n'
    '</Types>'
)

layout_data = {
    "id": 0,
    "resourcePackages": [],
    "sections": [
        {"id": 0, "name": "Section1", "displayName": "Executive Overview", "filters": "[]", "ordinal": 0},
        {"id": 1, "name": "Section2", "displayName": "Fraud Analysis", "filters": "[]", "ordinal": 1},
        {"id": 2, "name": "Section3", "displayName": "Model Performance", "filters": "[]", "ordinal": 2},
        {"id": 3, "name": "Section4", "displayName": "Transaction Monitoring", "filters": "[]", "ordinal": 3}
    ],
    "config": '{"version":"5.55"}'
}

with zipfile.ZipFile(pbix_path, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.writestr("[Content_Types].xml", content_types_xml)
    zf.writestr("Version", "1.28")
    zf.writestr("Report/Layout", json.dumps(layout_data))

print(f"Created template PBIX archive: {pbix_path}")
