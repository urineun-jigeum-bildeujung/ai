# Issue 운영 가이드

## 작업 흐름

```text
Issue 생성
→ Issue Type·Label·Assignee 설정
→ develop 기준 작업 브랜치 생성
→ Issue의 Development에 브랜치 연결
→ 구현·검증
→ develop 대상 PR 생성 및 Closes #번호 작성
→ 완료 항목 체크
→ PR 병합 후 Issue 종료 확인
```

Issue에는 **무엇을 왜 하는지와 완료 조건**을 기록하고, PR에는 **실제 변경 내용과 검증 결과**를 기록합니다.

## Form 선택

| 작업 상황 | 사용할 Form | Issue Type | type Label |
| --- | --- | --- | --- |
| 새로운 사용자 기능 구현 | 일반 작업 | `Feature` | `type:feature` |
| 리팩터링·문서·CI·환경 설정 | 일반 작업 | `Task` | `type:maintenance` |
| 재현 가능한 오류 수정 | 버그 신고 | `Bug` | `type:bug` |
| 모델·데이터 실험 | 모델·데이터 실험 | `Task` | `type:experiment` |

Form으로 표현하기 어려운 작업은 빈 Issue를 사용할 수 있습니다. 다만 작업 목적과 완료 조건은 반드시 작성합니다.

## Issue 생성 후 설정

다음 항목을 Issue 오른쪽 사이드바에서 설정합니다.

1. **Issue Type 1개**
   - `Feature`: 새로운 사용자 기능
   - `Bug`: 정상 동작을 벗어난 오류
   - `Task`: 실험·리팩터링·문서·CI·환경 설정
2. **type Label 1개**
   - `type:feature`, `type:bug`, `type:experiment`, `type:maintenance` 중 하나
3. **area Label 1개 이상**
   - `area:repurchase`, `area:recommendation`, `area:nutrition`, `area:platform` 중 작업 영향 범위에 맞게 선택
   - 여러 영역을 함께 변경하면 관련 `area:*` Label을 모두 지정
4. **Assignee**
   - 실제 구현을 담당하는 팀원을 지정
   - 공동 작업이면 주 담당자를 Assignee로 두고 나머지 담당자는 본문에 기록

상태를 나타내는 별도 Label은 사용하지 않습니다. 진행 상태가 필요해지면 GitHub Project에서 관리합니다.

## 브랜치 생성과 Development 연결

모든 일반 작업 브랜치는 최신 `develop`에서 생성합니다.

```bash
git switch develop
git pull --ff-only origin develop
git switch -c <prefix>/<short-description>
```

| 작업 종류 | Prefix | 예시 |
| --- | --- | --- |
| 새로운 기능 | `feature/` | `feature/repurchase-reminder` |
| 오류 수정 | `fix/` | `fix/invalid-purchase-date` |
| 모델·데이터 실험 | `experiment/` | `experiment/survival-baseline` |
| 리팩터링·문서·CI·환경 설정 | `chore/` | `chore/issue-guidelines` |

- 영문 소문자와 하이픈을 사용합니다.
- 작업 내용을 알아볼 수 있는 짧은 이름을 사용합니다.
- `main`이나 `develop`에 직접 커밋하지 않습니다.
- 브랜치를 만든 뒤 Issue의 **Development → Link a branch**에서 연결합니다.

GitHub CLI를 사용하면 Issue와 연결된 원격 브랜치를 바로 만들 수 있습니다.

```bash
gh issue develop <issue-number> \
  --base develop \
  --name <prefix>/<short-description> \
  --checkout
```

## PR 연결과 Issue 종료

일반 작업 PR의 대상 브랜치는 `develop`입니다. `develop → main` PR은 검증된 변경을 릴리스 브랜치에 반영할 때만 생성합니다.

PR 본문의 관련 이슈 항목에 다음과 같이 작성합니다.

```text
Closes #14
```

- `Closes #번호`를 사용하면 PR과 Issue가 Development에 연결됩니다.
- 기본 브랜치에 PR이 병합되면 연결된 Issue가 자동 종료됩니다.
- 단순 참고 관계만 남길 때는 `Related to #번호`를 사용합니다.
- PR을 병합하기 전에 Issue의 작업 범위와 완료 기준을 `[x]`로 갱신합니다.
- 자동 종료 후에도 체크리스트는 자동으로 바뀌지 않으므로 반드시 직접 확인합니다.

## Issue를 생략할 수 있는 경우

다음 조건을 **모두** 만족하는 작은 변경은 Issue를 생략할 수 있습니다.

- 오탈자·링크·주석처럼 동작을 바꾸지 않는 수정
- 한 파일 안에서 끝나는 짧은 변경
- DB·API·모델·데이터·환경 설정에 영향이 없음
- 별도의 검토할 의사결정이나 완료 조건이 없음

Issue를 생략해도 보호 브랜치에는 직접 커밋하지 않고 PR을 생성합니다. PR의 관련 이슈 항목에는 `없음`이라고 작성합니다.

다음 작업은 크기가 작아도 Issue를 생성합니다.

- 사용자 기능이나 API 동작 변경
- 모델·피처·학습 데이터·평가 방식 변경
- 의존성·CI·배포·환경 설정 변경
- 장애·보안·데이터 품질 문제 수정
- 두 명 이상이 협업하거나 후속 추적이 필요한 작업

## 완료 전 확인

- [ ] Issue Type이 작업 성격과 일치하는가?
- [ ] `type:*` 1개와 `area:*` 1개 이상이 설정됐는가?
- [ ] 실제 구현 담당자가 Assignee로 지정됐는가?
- [ ] 작업 브랜치가 `develop`에서 생성되고 Development에 연결됐는가?
- [ ] 완료 조건을 실제 검증했고 `[x]`로 갱신했는가?
- [ ] PR 본문에 `Closes #번호`와 검증 결과를 작성했는가?
