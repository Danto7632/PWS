# 포트폴리오 페이지

CS 업무 시뮬레이터 프로젝트 포트폴리오 페이지입니다.

## 파일 구조

```
portfolio/
├── index.html          # 포트폴리오 메인 페이지
├── architecture.puml   # 시스템 아키텍처 다이어그램 (PlantUML)
├── db-schema.puml      # DB 스키마 다이어그램 (PlantUML)
├── architecture.png    # 변환된 아키텍처 이미지 (생성 필요)
└── db-schema.png       # 변환된 DB 스키마 이미지 (생성 필요)
```

## PlantUML 다이어그램 이미지 생성

### 방법 1: 온라인 서버 사용

PlantUML 온라인 서버에서 직접 변환:
- [PlantUML Web Server](http://www.plantuml.com/plantuml/uml/)
- `.puml` 파일 내용을 붙여넣고 PNG로 다운로드

### 방법 2: VS Code 확장 사용

1. VS Code에서 "PlantUML" 확장 설치
2. `.puml` 파일 열기
3. `Alt + D` 또는 명령 팔레트에서 "PlantUML: Export Current Diagram"

### 방법 3: CLI 사용

```bash
# PlantUML JAR 다운로드 필요
java -jar plantuml.jar architecture.puml
java -jar plantuml.jar db-schema.puml
```

### 방법 4: Docker 사용

```bash
docker run --rm -v $(pwd):/data plantuml/plantuml architecture.puml
docker run --rm -v $(pwd):/data plantuml/plantuml db-schema.puml
```

## 로컬에서 미리보기

```bash
# Python 간단 서버
cd portfolio
python -m http.server 8080

# 또는 Node.js
npx serve .
```

브라우저에서 `http://localhost:8080` 접속

## 커스터마이징

### index.html 수정 필요 항목

1. **GitHub 저장소 링크**: `your-username`을 실제 GitHub 사용자명으로 변경
2. **시연 영상**: YouTube 영상 ID를 `YOUR_VIDEO_ID`에서 실제 값으로 변경
3. **이메일/LinkedIn**: 하단 푸터에서 연락처 정보 수정
4. **배포 링크**: Live Demo 링크 업데이트

### 다이어그램 이미지

- `architecture.png`와 `db-schema.png` 파일이 없으면 fallback으로 텍스트 다이어그램이 표시됩니다.
- 더 나은 시각적 효과를 위해 PNG 이미지를 생성하여 추가하세요.

## GitHub Pages 배포

GitHub Actions workflow가 설정되어 있어서:

1. `main` 또는 `master` 브랜치에 push
2. `frontend/` 또는 `portfolio/` 폴더 변경 시 자동 배포
3. GitHub Pages에서 접속 가능

배포 URL: `https://<username>.github.io/<repo-name>/`
