from flask import Flask, render_template, request
from flask_app.routes import feeling_analysis
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import re
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from hanspell import spell_checker
from numpy import dot
from numpy.linalg import norm
import urllib.request
from sentence_transformers import SentenceTransformer
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Embedding, Conv1D
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint



# , MultiHeadAttention, TransformerBlock, TokenAndPositionEmbedding
# https://getbootstrap.com/docs/5.0/forms/layout/
# https://getbootstrap.com/docs/5.0/content/images/

# https://wikidocs.net/81048
# 사용자가 ENTER를 눌렀을 때 데이터가 submit 되도록 한다. https://88240.tistory.com/52
# $ FLASK_APP=flask_app flask run
# $ pip freeze > requirements.txt
# https://fitness6302.herokuapp.com/ | https://git.heroku.com/fitness6302.git

# 데이터 불러오기 및 간단한 전처리
stop_word = pd.read_excel('./stop_words.xlsx',header=None) 
stop_words=set(stop_word.iloc[:,0])

feel_bic_dic = {'기쁨': 0, '당황': 2, '분노': 4, '불안': 1, '상처': 5, '슬픔': 3}
feel_bic_dic_reverse = {0: '기쁨', 1: '불안', 2: '당황', 3: '슬픔', 4: '분노', 5: '상처'}


# 모델 불러오기
# 다층퍼셉트론 커스템모델 불러오기(https://www.tensorflow.org/tutorials/keras/save_and_load?hl=ko)
# https://vision-ai.tistory.com/entry/%ED%85%90%EC%84%9C%ED%94%8C%EB%A1%9C%EC%9A%B0%EC%9D%98-%EB%AA%A8%EB%8D%B8-weight%EA%B0%80%EC%A4%91%EC%B9%98-%EC%A0%80%EC%9E%A5%ED%95%98%EA%B3%A0-%EB%B6%88%EB%9F%AC%EC%98%A4%EA%B8%B0
model_custom = load_model('./feel_analysis_DD.h5')


# ko-bert불러오기.
model_bert = SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')
df = pd.read_pickle("./chat_embedding_train.pkl") # 임베딩까지 완료된 데이터

def cos_sim(A, B):
    return dot(A, B)/(norm(A)*norm(B))

def return_answer(question):
    embedding = model_bert.encode(question)
    df['score'] = df.apply(lambda x: cos_sim(x['embedding'], embedding), axis=1)
    return df.loc[df['score'].idxmax()]['시스템응답1']


# 예측해보기 : 감정분류
okt=Okt() 
tokenizer = Tokenizer()
# 예측해보기
def sentiment_predict(new_sentence):
    new_sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣 ]','', new_sentence)
    new_sentence = okt.morphs(new_sentence, stem=True) # 토큰화
    new_sentence = [word for word in new_sentence if not word in stop_words] # 불용어 제거
    encoded = tokenizer.texts_to_sequences([new_sentence]) # 정수 인코딩
    pad_new = pad_sequences(encoded, maxlen = 25) # 패딩
    score = model_custom.predict(pad_new) # 예측
    print(score)
    print(score[0, score.argmax()])
    print(feel_bic_dic_reverse[score.argmax()])  #load_model을 했을 때 첫번째 말한 임베딩에 고정되어있다. 왜지?  -> ahp
    return feel_bic_dic_reverse[score.argmax()]


def create_app():
    app = Flask(__name__)
    # app.register_blueprint(feeling_analysis.bp)

    @app.route('/', methods=['POST', 'GET'])
    def index():
        if request.method == 'GET':   #GET 방식의 경우 모든 파라미터를 url로 보내 요청
            answer = '안녕하세요 세연님'
            KANNA = 'image/KANNA.png'
            return render_template('index.html', answer=answer, image_file=KANNA)
        if request.method == 'POST':

            # 챗봇대답
            usertext = request.form.get('formtext')
            print(usertext)
            answer = return_answer(usertext)
            print(answer)

            # 감정분류
            usertext = request.form.get('formtext')
            feel = sentiment_predict(usertext)
            print(feel)


            

# {0: '기쁨', 1: '불안', 2: '당황', 3: '슬픔', 4: '분노', 5: '상처'}
            if feel == '기쁨':
                KANNA = 'image/기쁨.png'
            elif feel == '불안':
                KANNA = 'image/불안.png'
            elif feel == '당황':
                KANNA = 'image/당황.png'
            elif feel == '슬픔':
                KANNA = 'image/슬픔.png'
            elif feel == '분노':
                KANNA = 'image/분노.png'
            elif feel == '상처':
                KANNA = 'image/상처.png'

            return render_template('index.html', feel=feel , answer= answer, image_file=KANNA)



    return app

if __name__ == "__main__":
    app = create_app(debug=False)
    app.run() 