# Dataset manifests

각 manifest는 출처, 라이선스, 수집일, 파일 해시, 스키마 버전, 데이터 분류(public/approved_restricted/synthetic/derived), 분할 정보를 기록해야 합니다.

`*.json` manifest에는 아래 필드가 필수입니다.

- `schema_version`, `dataset_id`, `dataset_version`, `created_at`
- `classification`: `public`, `approved_restricted`, `synthetic`, `derived` 중 하나
- `source`: `name`, `url`, `license`, `version`, ISO-8601 `collected_at`
- `split`: `name`, `strategy`, `unit`, `parameters`, `seed`
- `artifacts`: 각 항목에 `path`와 64자리 `sha256`
- `derived` 데이터: `preprocessing.implementation`, `version`, `parameters`
- `synthetic` 데이터: `generation.implementation`, `version`, `parameters`, `seed`
- `integrity_report`: 버전·생성시각과 스키마 검증, 필수필드 누락, 중복, source drift 검사 결과
- `integrity_report.split_integrity.entities`: `speaker`, `source`, `event`별 `passed`와 겹침 0건 또는 `not_applicable`과 사유

게시된 manifest 경로는 커밋 단위 append-only입니다. 새 데이터 버전은 같은 PR의 중간 커밋에서도 기존 JSON을 고치거나 지우지 말고 새 경로로 추가해야 합니다.

현재 CI는 위 메타데이터와 integrity report의 구조·통과 상태를 확인하지만 외부 원본을 다시 내려받아 통계를 재계산하지는 않습니다. 라이선스의 법적 유효성, URL의 진위, 데이터 내용의 품질은 이 검사만으로 보증되지 않습니다.

## 작은 fixture

Git에 둘 수 있는 작은 fixture는 `fixtures/` 아래에 한정합니다. 각 파일 옆에 `<원본파일명>.fixture.json`을 두고 분류(`synthetic` 또는 `public_redistributable`), 개인정보 없음, 라이선스, 출처 버전·수집시각, 파일 SHA-256을 기록해야 합니다. 합성 fixture에는 생성 구현·버전·파라미터·seed도 필요합니다. 이 조건을 통과한 작은 오디오 fixture는 허용하지만 실제 신고·무전 원본은 금지합니다. 모든 추적 파일에서 이메일·한국 전화번호·주민등록번호 형태를 보조 휴리스틱으로 차단하지만, 이 검사는 개인정보 부재를 완전히 보증하지 않으므로 사람의 검토가 함께 필요합니다.

JSON은 dataset manifest, fixture 동반 메타데이터, `schemas/` 아래 스키마만 허용합니다. 일반 코퍼스 JSON은 외부 데이터 저장소에 둡니다.
