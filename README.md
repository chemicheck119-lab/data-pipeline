# 케미체크119 Data Pipeline

AIHub·KOSHA·ICIS·PRTR 등 공개·승인 데이터를 재현 가능하게 수집하고, 출처와 품질을 검증해 버전이 고정된 manifest를 만드는 저장소입니다.

## 책임

- 데이터 다운로드·수집 스크립트
- 원본 출처·라이선스·기준일 기록
- 정규화·중복 제거·품질검증
- train/dev/test 분할 manifest와 파일 해시
- `speech-service`와 `analysis-engine`이 소비할 데이터 계약

## 저장하지 않는 것

- AIHub 원본 음성 및 개인정보
- KOSHA 등 원문 전체 덤프
- 모델 가중치와 전처리 산출물
- 현재 재고로 오해할 수 있는 미검증 시설 데이터

대용량 원본은 승인된 외부 저장소에 보관하고 Git에는 수집 방법, manifest, 해시, 스키마만 남깁니다. 작은 테스트 fixture는 라이선스·출처·개인정보 없음·SHA-256을 적은 동반 메타데이터를 통과한 경우에만 허용합니다.

Manifest는 용도(`training`, `development`, `evaluation`, `fixture`, `reference`)를 명시합니다. 평가용 manifest는 울산 Resolver 419건과 전국 Parser 442건의 식별자·건수를 서로 다르게 검증하며, 평가 split을 튜닝에 사용했다고 표시하면 CI가 거부합니다.

## AIHub 119 화재 음성 하위셋

전체 87GB를 내려받지 않고 광주 화재 파티션만 독립 기준선으로 사용합니다. 원본은 승인된 비공개 저장소에만 두고, 아래 명령은 음성·라벨 쌍, 중복, 필수 필드, 오디오 형식, split 누수를 검사한 뒤 통계와 SHA-256만 manifest에 기록합니다.

```bash
PYTHONPATH=src python -m chemicheck119_data.aihub119 \
  --training-audio /secure/TS_광주_화재.zip \
  --training-labels /secure/TL_광주_화재.zip \
  --validation-audio /secure/VS_광주_화재.zip \
  --validation-labels /secure/VL_광주_화재.zip \
  --artifact-prefix gs://PRIVATE_BUCKET/raw/aihub/71768/gwangju-fire \
  --collected-at 2026-09-05T07:16:31Z \
  --output-dir data/manifests
```

Manifest는 append-only이므로 동일한 출력 경로가 있으면 명령이 실패합니다. 새 원본 snapshot은 새 버전의 고유 경로로 기록해야 합니다. 첫 snapshot에는 비교 기준이 없으므로 source drift를 `passed`가 아닌 `not_applicable`로 기록합니다.

후속 snapshot에서는 이전 manifest 두 개가 있는 디렉터리를 `--baseline-dir`로 지정합니다. 아카이브 SHA-256이 하나라도 바뀌면 새 manifest를 발행하지 않고 실패하며, 모두 같을 때만 source drift를 `passed`로 기록합니다. 고정 snapshot은 Training 659건과 Validation 77건을 모두 충족해야 합니다.

이 데이터는 [AI 허브 개방 데이터 이용정책](https://www.aihub.or.kr/intrcn/guid/usagepolicy.do)의 2026-09-05 조회본을 적용했습니다. AI 모델 학습 목적, NIA 사업결과 표기, 승인 없는 제3자 열람·재배포 금지, 국외 반출 시 별도 합의 조건을 manifest에 기록했습니다. 게시된 정책 자체에는 버전 식별자가 없어 조회일도 함께 남겼습니다.

이번 AIHub ZIP에서는 라벨의 `audioPath`가 실제 압축 파일 안의 WAV 이름과 일치하지 않습니다. Training 659건과 Validation 77건 모두 라벨 JSON과 WAV의 **압축 멤버 파일명 stem**이 1:1로 일치해 그 규칙으로 연결했으며, provider path 불일치 건수를 manifest에 남깁니다. 이는 제공 원본의 한계이며 별도의 공식 키가 확인되기 전까지 “완전 검증된 연결”로 표현하지 않습니다.

이 평가는 `speech_aihub119_gwangju_fire_validation_77`이며 Resolver 울산 419건, Parser 전국 442건 평가와 별개입니다. 신고접수 음성이므로 실제 소방 무전 성능을 증명하지 않습니다.

## 현재 상태

| 항목 | 상태 |
|---|---|
| 저장소·정책 CI 골격 | 구현 완료 |
| AIHub 신고음성 manifest 생성기 | 구현 완료 |
| 광주 화재 Training 659건·Validation 77건 검사 | 로컬 실행 완료 |
| STT train/dev/test 분할 | AIHub 제공 Training/Validation 사용; 별도 test 미구성 |
| 실제 현장 무전 데이터 | 확보 불가·검증되지 않음 |

## 기본 검증

```bash
python -m pip install --requirement requirements-dev.txt
python scripts/check_repository_policy.py
```
