# NaverShoppingCrawler-ImageSimilarityComparison
Python과 Flask를 이용한 네이버 쇼핑 크롤러 & 이미지 유사도 비교 / 졸업작품

이미지 유사도 비교 코드는 
https://medium.com/daangn/%EC%9D%B4%EB%AF%B8%EC%A7%80-%ED%83%90%EC%A7%80%EA%B8%B0-%EC%89%BD%EA%B2%8C-%EA%B5%AC%ED%98%84%ED%95%98%EA%B8%B0-abd967638c8e
에서 참조하였습니다.

구동방법

1. Anaconda를 설치하여 Anaconda Navigator를 킨다. <br>
<img src="https://user-images.githubusercontent.com/56918200/95821735-7cb49700-0d65-11eb-81ac-53e11c2b194a.png" width="50%"></img>
<br>
2. Environments 탭을 클릭해서 create를 눌러 새로운 가상환경을 생성<br>
3. Installed를 Not Installed로 전환<br>
4. Search Packages에 flask를 검색, "flask" 이름 옆에 있는 공란을 체크<br>
<img src="https://user-images.githubusercontent.com/56918200/95822225-72df6380-0d66-11eb-8401-7cd8157d8dc4.png" width="50%"></img>
<br>

5. 다시 Search Packages에 Tensorflow를 검색하여 Tensorflow와 Tensorflow hub를 체크한다. <br>
6. 최하단에 Apply를 눌러서 패키지들을 설치 <br>
<img src="https://user-images.githubusercontent.com/56918200/95822615-395b2800-0d67-11eb-86cf-0bd533c20c52.png" width="50%"></img>
<br>

7. 다시 Home 탭으로 돌아가 Applications on 옆에 있는 가상환경을 새로 만든 가상환경으로 전환<br>
8. vscode를 설치하여서 launch 시킨 후 터미널을 열거나 cmd prompt를 설치하여 launch한다.<br>
<img src="https://user-images.githubusercontent.com/56918200/95822716-690a3000-0d67-11eb-93cd-f47f94eff042.png" width="50%"></img>
<br>

9. app.py가 있는 프로젝트 폴더의 경로에서 "flask run"을 입력하여 구동<br>
<img src="https://user-images.githubusercontent.com/56918200/95822925-bdadab00-0d67-11eb-9989-5311b09cc44e.png" width="80%"></img>
<br>

********** 시연 동영상 링크 **********<br>
<a href="https://drive.google.com/file/d/1pn5qyAX7TyKlQ7KW5qffFQCOaeN-ld78/view?usp=sharing" target="_blank"><img src="https://user-images.githubusercontent.com/56918200/95824914-103c9680-0d6b-11eb-98aa-54a2184b8c6f.jpg" 
alt="IMAGE ALT TEXT HERE" width="720" height="540" border="10" /></a>


※ 네이버 API를 가져올 때 Client Id와 Client Secret이 필요합니다. 
  네이버 개발자 포럼( https://developers.naver.com/main/ )에 문의. 
  
