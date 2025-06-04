#!/usr/bin/env bash
#
# supersonic_M1p5_download.sh
#
# NASA TMR “Supersonic Isothermal-Wall Channel Flow” 페이지에서
# 1) M=1.5 케이스용 .dist 파일(예: M1p5.dist)을 자동으로 내려받고,
# 2) 그 안에 나열된 실제 .bin 파일 목록을 읽어 한꺼번에 모두 다운로드합니다.
#
# 사용법:
#   1) 이 스크립트에 실행 권한을 부여합니다:
#        chmod +x supersonic_M1p5_download.sh
#   2) 스크립트를 실행합니다:
#        ./supersonic_M1p5_download.sh
#
# 실행 결과:
#   - “supersonic_M1p5.dist” 파일을 받은 뒤, 그 안에 나열된 “supersonic_M1p5_Re3000_tXXX.bin”들을
#     한 번에 “supersonic_M1p5_data/” 폴더로 다운로드합니다.


### 1) 기본 설정
BASE_PAGE_URL="https://turbmodels.larc.nasa.gov/Other_DNS_Data/supersonic-channel.html"
# .dist 파일을 통해 실제 .bin 파일 목록을 확인할 수 있는 주소(상대경로 또는 절대경로)
# “M1p5.dist” 로 끝나는 링크를 찾아 내려받습니다.
# (예: supersonic_M1p5.dist 또는 M1p5.dist 등, 실제 페이지에서 이름 확인 필요)

OUTPUT_DIR="supersonic_M1p5_data"
mkdir -p "${OUTPUT_DIR}"

TMP_DIST="supersonic_M1p5.dist"   # 내려받을 .dist 파일 이름. 페이지에 따라 실제 파일명이 다를 수 있으니 확인 후 수정.
EXT="bin"                         # 내릴 실제 데이터 파일 확장자. .bin 대신 .h5라면 "h5"로 변경.


### 2) “M=1.5 케이스(.dist 파일)” 링크 추출 및 다운로드
echo ">> 1) Supersonic 페이지에서 M=1.5(.dist) 링크를 찾아 내려받습니다."

# ① 페이지 HTML을 내려받아 “.dist” 링크를 grep으로 검색
dist_rel=$(curl -s "${BASE_PAGE_URL}" \
    | grep -Eo 'href="[^"]+M1p5[^"]+\.dist"' \
    | sed -e 's/^href="//' -e 's/"$//' \
    | head -n1)

if [[ -z "${dist_rel}" ]]; then
  echo "Error: M1p5 케이스용 .dist 링크를 찾지 못했습니다."
  exit 1
fi

# ② 만약 dist_rel이 절대 URL(http://...)이라면 그대로 쓰고, 아니면 BASE_DIR을 붙임
if [[ "${dist_rel}" =~ ^https?:// ]]; then
  dist_url="${dist_rel}"
else
  page_dir=$(dirname "${BASE_PAGE_URL}")
  dist_url="${page_dir}/${dist_rel}"
fi

echo "    → Found .dist file: ${dist_url}"
echo "    → Downloading to ${TMP_DIST} ..."
wget -q --show-progress -O "${TMP_DIST}" "${dist_url}"

if [[ ! -s "${TMP_DIST}" ]]; then
  echo "Error: .dist 파일을 정상적으로 내려받지 못했습니다."
  rm -f "${TMP_DIST}"
  exit 1
fi
echo "    → .dist 파일 다운로드 완료: ${TMP_DIST}"
echo ""


### 3) .dist 파일 내 실제 .bin 파일 목록 추출
# dist_list.txt 에 “supersonic_M1p5_Re3000_tXXX.bin” 같은 이름만 줄 단위로 뽑아둡니다.
echo ">> 2) .dist 파일에서 실제 .${EXT} 파일 목록을 추출합니다."
grep -E "supersonic_M1p5_Re3000_t[0-9]{3}\.${EXT}" "${TMP_DIST}" \
  | awk '{print $1}' \
  | sed -e 's/[\r\n]//g' > dist_list.txt

if [[ ! -s dist_list.txt ]]; then
  echo "Error: .dist 파일 안에 .${EXT} 파일 목록을 찾을 수 없습니다."
  rm -f dist_list.txt
  exit 1
fi

echo "    → dist_list.txt 에 `wc -l < dist_list.txt` 개의 파일명이 저장되었습니다."
echo ""


### 4) 목록에 있는 각 파일을 다운로드
echo ">> 3) 실제 .${EXT} 파일들을 한꺼번에 다운로드합니다."
# dist_list.txt 에 있는 파일명(상대경로일 수 있음)을 순회
while read -r rel_path; do
  # 공백이나 빈 줄이 섞여있으면 건너뜀
  [[ -z "${rel_path// }" ]] && continue

  # ① 링크가 절대 URL인지 확인
  if [[ "${rel_path}" =~ ^https?:// ]]; then
    file_url="${rel_path}"
    filename=$(basename "${rel_path}")
  else
    # 아니면 supersonic-channel.html 위치 디렉터리 기준 상대경로
    page_dir=$(dirname "${BASE_PAGE_URL}")
    file_url="${page_dir}/${rel_path}"
    filename=$(basename "${rel_path}")
  fi

  # ② 이미 같은 이름 파일이 있으면 스킵
  if [[ -f "${OUTPUT_DIR}/${filename}" ]]; then
    echo "[SKIP] ${filename} (already exists)"
    continue
  fi

  # ③ HTTP HEAD 요청으로 파일 존재 여부 확인
  http_code=$(curl -s -o /dev/null -w "%{http_code}" -I "${file_url}")
  if [[ "${http_code}" == "200" ]]; then
    echo "[DL]   ${filename}"
    wget -q --show-progress -O "${OUTPUT_DIR}/${filename}" "${file_url}"
  else
    echo "[MISS] ${filename} (HTTP ${http_code})"
  fi

done < dist_list.txt

echo ""
echo "=== 다운로드 완료 ==="
echo "총 다운로드된 파일 수: $(ls -1 ${OUTPUT_DIR}/*.${EXT} 2>/dev/null | wc -l)"
echo "다운로드된 파일 목록:"
ls -1 "${OUTPUT_DIR}"/*.${EXT} 2>/dev/null


### 5) 정리
# 임시 파일(dist_list.txt 등) 삭제
# (원한다면 이 부분을 주석 처리하여 dist_list.txt를 남겨놓을 수 있습니다)
rm -f "${TMP_DIST}" dist_list.txt

exit 0
