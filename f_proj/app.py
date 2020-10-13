from flask import Flask
from flask import render_template, redirect, url_for
from flask import request
import time

# json 파싱
import os
import sys
import json
import urllib.request

# 디렉토리 생성
def mkdirfolder(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

# HTML 태그 제거
import re

# 이미지 다운로드 https://blog.naver.com/ahnsk3939/222004278490
def imgsave(file_path, file_name, img_url):
    urllib.request.urlretrieve(img_url,file_path+file_name)

app = Flask(__name__)

list_item = []              # 상품 데이터 [상품명, 이미지 URL, 상품 페이지 링크, 최저가(가격)]
similar = []                # similarity가 0.8이상인 리스트 홀수 index는 target, 짝수 index는 target 이미지와 similartiy가 0.8이상인 상품들의 index 리스트

# 검색화면
@app.route('/')
def main_page():
    return render_template('main.html')

# 결과화면
@app.route('/search',methods = ['POST', 'GET'])
def search():
    if request.method == 'POST':    # POST
        kw = request.form['keyword'] # 검색어
        find_num = "10" # 검색 출력 건수 => 10으로 고정

        # api 사용 - client id와 secret이 필요
        client_id = ""
        client_secret = ""
        url_base = "https://openapi.naver.com/v1/search/shop.json?query="

        key = kw # 검색어
        keyword = urllib.parse.quote(key) # 한글을 URL 인코딩시킨다

        # 1페이지부터 검색
        url_start = "$&start="
        keyword_start = "1"

        display_number = find_num # 검색할 건수

        url = url_base + keyword + url_start + keyword_start + "?display=" + display_number

        req = urllib.request.Request(url)
        req.add_header("X-Naver-Client-Id", client_id)
        req.add_header("X-Naver-Client-Secret", client_secret)
        req.add_header("Content-Type", "application/json")

        res = urllib.request.urlopen(req)
        res_code = res.getcode()

        if(res_code == 200):
            t_start = time.time()

            list_item.clear()
            similar.clear()

            res_body = res.read()
            json_items = json.loads(res_body)

            for item in json_items['items']:
                # title 값에서 HTML 태그 제거 https://blog.naver.com/wideeyed/221347960543 
                pattern = '<[^>]*>'
                repl = ''
                t = re.sub(pattern=pattern, repl=repl, string=item['title'])

                # 가격 3자리씩 끊어서 표시
                price = format(int(item['lprice']), ",")

                list_item.append([t, item['image'], item['link'], price]) # list_item = [상품명, 이미지 URL, 상품 페이지 링크, 가격(최저가)]
                print(t)
                print(item['image'])
                print(item['link'])
                print(price)


            # 당근 이미지 처리기 STEP 5
            import tensorflow.compat.v1 as tf
            tf.disable_v2_behavior()
            import graph 
            

            ####### 이미지 비교 ####### => 최대 10번 반복
            # 이미지들을 가져오기

            target_img_path = 'target_img.jpg'
            input_img = []                  # 10개
            input_img_paths = []            # 9개
            local_img_name = []             # 실제로 로컬에 저장되는 이미지 파일 이름들
            
            included = []                   # target image index를 거치면서 similar 리스트에 포함되어있는 확인하는 리스트

            non_similar = []                # similarity가 0.8미만인 리스트 => 1 ~ display_number-1 의 정수로 초기 구성.
            for i in range(0, 10):
                non_similar.append(i)

            target_idx = 0                  # 타겟이미지를 저장할 이미지의 인덱스(0 부터 시작)

            for img in list_item:
                input_img.append(img[1])    # img[i] : list_item 에서 2번째 원소인 item['image']를 나타냄.

            # 비교할 대상 이미지들 다운로드
            for i, url in enumerate(input_img):
                if len(url) > 0:
                    img_name = 'input_img%d.jpg' % i
                    imgsave('', img_name,url)
                    local_img_name.append(img_name)    
           
            ############# 반복구간 ################

            while target_idx < 10 :    # 최대 display 값만큼 반복
                print("target_idx : ", target_idx)
                # target 이미지 인덱스가 similarity가 0.8이상인 이중 리스트에 들어가는 경우 제외 => 다음 인덱스로 넘어간다.
                incl = 0            # 아무 원소도 같지 않음
                for i in included:
                    if target_idx == i:
                        incl = 1    # 원소가 하나라도 포함됨
                        break
                
                print("incl : ", incl)
                if incl != 0:       # target index가 similar 리스트에 포함되어 있는지 확인
                    target_idx = target_idx + 1 
                    continue
                else :
                    included.append(target_idx)
                    non_similar.pop(0)
                    similar.append(target_idx)

                    if (target_idx == 9) or (len(non_similar) == 0):     # 맨 끝 이미지가 타겟 이미지일 경우 또는 남은 non_similar 리스트 원소 개수가 없을 경우? => while문 종료
                        empty_arr = []
                        similar.append(empty_arr)
                        break

                    target_img_path = 'input_img%d.jpg' % target_idx    # 타겟이미지 지정

                    for i in non_similar:    # 비교 대상 이미지 지정
                        input_img_paths.append(local_img_name[i])

                    # Load bytes of image files
                    image_bytes = [tf.gfile.GFile(name, 'rb').read()
                    for name in input_img_paths]

                    hub_module_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_96/feature_vector/1" #@param {type:"string"}

                    with tf.Graph().as_default():
                        input_byte, similarity_op = graph.build_graph(hub_module_url, target_img_path)
  
                        with tf.Session() as sess:
                            sess.run(tf.global_variables_initializer())
                            t0 = time.time() # for time check
    
                            # Inference similarities
                            similarities = sess.run(similarity_op, feed_dict={input_byte: image_bytes})
    
                            print("%d images inference time: %.2f s" % (len(similarities), time.time() - t0))

                        # 이미지 유사도 추출
                        idx = target_idx + 1           # 타겟이미지 인덱스부터 시작
                        idx_del = 0                    # for non_similar.pop
                        temp = []               # 유사한 이미지들 집합
                        print("# Input images")
                        for similarity, input_img_path in zip(similarities, input_img_paths):
                            # 이미지 유사도 저장(target image는 제외, 1로 고정)
                            s = "%.2f" % similarity
                            if similarity >= 0.8 :      # similarity가 0.8이 넘을 경우
                                del non_similar[idx_del]
                                idx_del = idx_del - 1
                                included.append(idx)
                                temp.append(idx)

                            # 터미널에 출력
                            print(input_img_path)
                            print("- similarity: %.2f" % similarity)
                            idx = idx + 1
                            idx_del = idx_del + 1
                    
                    print("temp : ", temp)
                    print("similar : ", similar)
                    print("non_similar : ", non_similar)
                    print("included : ", included)
                    similar.append(temp)
                    input_img_paths.clear()
                    target_idx = target_idx + 1 

        else:
            print("Error Code:" + res_code)
        

    # 이미지 파일들 모두 삭제 https://stackoverflow.com/questions/1995373/deleting-all-files-in-a-directory-with-python/1995397
    rm_file = [ f for f in os.listdir('./') if f.endswith(".jpg")]
    for f in rm_file:
        os.remove(os.path.join('./', f))

    print("검색 소요 시간 : %.2f 초" % (time.time() - t_start))
    print("최종 similar : ", similar)
    print("최종 included : ", included)
    print("최종 non_similar : ", non_similar)
    return render_template("view_page.html", keyword = kw, 
    result_num = int(len(similar)/2), number = find_num, list_item = list_item, similar_set = similar)

@app.route('/content/<idx>')
def content(idx):
    return render_template("view_similar.html", list_item=list_item, idx = int(idx), similar=similar[int(idx)+1], target = similar[int(idx)])

if __name__ == "__main__":
    app.run()