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
        client_id = "직접 입력"
        client_secret = "직접 입력"
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

            res_body = res.read()
            json_items = json.loads(res_body)

            list_item = []

            for item in json_items['items']:
                # title 값에서 HTML 태그 제거 https://blog.naver.com/wideeyed/221347960543 
                pattern = '<[^>]*>'
                repl = ''
                t = re.sub(pattern=pattern, repl=repl, string=item['title'])

                list_item.append([t, item['image'], item['link'], 1]) # list_item = [상품명, 이미지 URL, 상품 페이지 링크, 유사도]
                print(t)
                print(item['image'])
                print(item['link'])


            # 당근 이미지 처리기 STEP 5
            import tensorflow.compat.v1 as tf
            tf.disable_v2_behavior()
            import graph

            ####### 이미지 비교 #######
            # 이미지들을 가져오기

            target_img_path = 'target_img.jpg'
            input_img = []
            input_img_paths = []
            
            i_n = 1

            for img in list_item:
                if i_n == 1:
                    target_img_path = img[1]
                    # 비교할 기준 이미지 다운로드(target image => 제일 첫번째 원소)
                    t_path = 'target_img.jpg'
                    imgsave('',t_path,target_img_path)
                    target_img_path = t_path
                    i_n = i_n + 1
                else:
                    input_img.append(img[1])
                    i_n = i_n + 1
            
            # 비교할 대상 이미지들 다운로드
            for i, url in enumerate(input_img):
                if len(url) > 0:
                    path = 'input_img%d.jpg' % i
                    imgsave('',path,url)
                    input_img_paths.append(path)
           
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
                idx = 1;
                print("# Input images")
                for similarity, input_img_path in zip(similarities, input_img_paths):
                    # 이미지 유사도 저장(target image는 제외, 1로 고정)
                    s = "%.2f" % similarity
                    list_item[idx].insert(3, s)

                    # 터미널에 출력
                    print(input_img_path)
                    print("- similarity: %.2f" % similarity)
                    idx = idx + 1

        else:
            print("Error Code:" + res_code)
        

    # 이미지 파일들 모두 삭제 https://stackoverflow.com/questions/1995373/deleting-all-files-in-a-directory-with-python/1995397
    rm_file = [ f for f in os.listdir('./') if f.endswith(".jpg")]
    for f in rm_file:
        os.remove(os.path.join('./', f))

    print("검색 소요 시간 : %.2f 초" % (time.time() - t_start))

    return render_template("view_page.html", keyword = kw, 
    result_num = len(list_item), number = find_num, list_item = list_item)

if __name__ == "__main__":
    app.run()