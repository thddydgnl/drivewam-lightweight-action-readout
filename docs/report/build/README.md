# 문서 재생성 방법

- `make_figs.py`: 그림 9장을 `fig/`에 생성한다 (matplotlib, Apple SD Gothic Neo). 같은 폴더의 `fonts/`에 Apple SD Gothic Neo의 Regular/Medium/SemiBold/Bold/Light ttf가 필요하다. 없으면 아래로 추출한다.
  ```bash
  python3 -c "from fontTools.ttLib import TTCollection; c=TTCollection('/System/Library/Fonts/AppleSDGothicNeo.ttc'); [c.fonts[i].save(f'fonts/AppleSDGothicNeo-{n}.ttf') for i,n in {0:'Regular',2:'Medium',4:'SemiBold',6:'Bold',8:'Light'}.items()]"
  ```
- `build_doc.py <출력 폴더>`: `fig/`의 그림을 넣어 docx를 만든다 (python-docx).
- PDF는 Microsoft Word로 열어 "PDF로 저장"하면 된다 (글꼴: Apple SD Gothic Neo).
