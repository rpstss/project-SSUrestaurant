import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import numpy as np
import requests
from io import BytesIO
from matplotlib import font_manager



# 첫 번째 폰트 파일 다운로드.
font_url1 = "https://github.com/rpstss/project-SSUrestaurant/raw/main/NanumGothic.ttf" # 나눔고딕 ttf입니다.
response1 = requests.get(font_url1)
with open("NanumGothic.ttf", "wb") as f1:
    f1.write(response1.content)

# 두 번째 폰트 파일 다운로드
font_url2 = "https://github.com/rpstss/project-SSUrestaurant/raw/main/Arial%20Unicode%20MS.TTF" # airal unicode ms ttf입니다.
response2 = requests.get(font_url2)
with open("Arial Unicode MS.ttf", "wb") as f2:
    f2.write(response2.content)

# 폰트 파일 로드
font_manager.fontManager.addfont("NanumGothic.ttf")
font_manager.fontManager.addfont("Arial Unicode MS.ttf")

# 한글 변수 이름이 깨짐을 방지하기 위해 font 바꾸기
plt.rcParams['font.family'] = 'NanumGothic'

raw_df=pd.read_csv("https://raw.githubusercontent.com/rpstss/project-SSUrestaurant/main/%EC%88%AD%EC%8B%A4%EB%8C%80.csv",encoding="cp949") # 식당 데이터 csv으로 raw 데이터 프레임 생성
raw_copy=raw_df.copy() # 혹시 모를 데이터 정보 수정을 위해, raw_df의 복제본 생성


import streamlit as st
st.set_page_config(layout="wide") # streamlit의 좁은 레이아웃을 넓게 설정
st.title("숭실대 주변 500m 이내 식당 정보 :fork_and_knife:")
st.write("\n")
st.write("\n")
st.markdown("### **목차** ###")
st.markdown("- 카테고리별 식당 수 및 비율")
st.markdown("- 평점 분포 분석")
st.markdown("- 상관관계 분석")
st.markdown("- 가격대 분포")
st.markdown("- 리뷰 분석")
st.markdown("- 식당 추천")

st.markdown("---")

# 카테고리별 식당 수 데이터프레임 num_of_category에 할당
num_of_category=raw_copy.groupby("category",as_index=False)\
                        .agg(n=("category","count"))\
                        .sort_values("n",ascending=False) 
# 비율 분석을 위해  ratio 변수 생성
num_of_category=\
    num_of_category\
    .assign(ratio=lambda x: x["n"] / sum(x["n"])*100)\
    .round(1) 

st.subheader(":blue[카테고리별 식당 수 및 비율]")

col1,col2=st.columns([3,1])

with col1:
    col1_1,col1_2= st.columns(2)
    
    with col1_1:
        st.markdown("#### **식당 수 막대 그래프** ####")
        st.write("\n")
        fig_1 = plt.figure(figsize=(6, 6))
        
        # 카테고리별 식당 수 막대그래프 생성
        barplot_num_of_category = sns.barplot(
            data=num_of_category,
            x="category",
            y="n",
            order=num_of_category["category"])
        
        st.pyplot(fig_1)
        
    with col1_2:
        st.markdown("#### **식당 비율 파이차트** ####")
        fig_2=plt.figure()
        
        # 카테고리별 식당 비율 파이차트 생성
        piechart_ratio_of_category=\
        plt.pie(num_of_category["n"]
        ,labels=num_of_category["category"]
        ,autopct='%1.1f%%' # 비율 표시
        ,shadow=True # 그림자 효과 추가
        ,explode=[0.1,0.1,0.1,0,0,0,0]) # 파이 차트가 분리되게 지정
        
        st.pyplot(fig_2)

    
with col2:
    st.markdown("#### **데이터 프레임** ####")
    st.write("\n")
    st.dataframe(num_of_category)
    
st.markdown("---")


# 평점 분포
raw_copy=raw_copy.rename(columns={"star":"평점"}) # 변수명 교체

st.subheader(":blue[평점 분포 분석]")
 
col3, col4, col5 = st.columns([2,4,4])

with col3:
    agree = st.radio("type select",("전체","한식","그 외","양식","일식","카페 및 디저트","분식","중식")) # 카테고리를 선택하는 agrr 라디오 버튼 생성
    
def hist_star(x): # 카테고리별 평점 분포 히스토그램 작성 함수
    a= sns.histplot(data=raw_copy.query("category== '%s'"%x),x="평점",kde=True)
    return a

def box_star(x): # 카테고리별 평점 상자 그래프 작성 함수
    a=sns.boxplot(data=raw_copy.query("category== '%s'"%x),y="평점")
    return a

if agree=="전체": 
    with col4:
        st.markdown("#### **히스토그램 & 밀도 추정 곡선** ####")
        fig_3=plt.figure()
        histplot_grade_of_all=sns.histplot(data=raw_copy,x="평점",kde=True)
        st.pyplot(fig_3)
    with col5:
        st.markdown("#### **상자 그래프** ####")
        fig_4=plt.figure()
        histplot_grade_of_all=sns.boxplot(data=raw_copy,y="평점",x="category")
        st.pyplot(fig_4)

if agree=="한식":
    with col4:
        st.markdown("#### **히스토그램 & 밀도그래프** ####")
        fig_3=plt.figure()
        histplot_grade_of_category=hist_star("한식")
        st.pyplot(fig_3)
    with col5:
        st.markdown("#### **상자 그래프** ####")
        fig_4=plt.figure()
        histplot_grade_of_category=box_star("한식")
        st.pyplot(fig_4)
        
if agree=="그 외":
    with col4:
        st.markdown("#### **히스토그램 & 밀도그래프** ####")
        fig_3=plt.figure()
        histplot_grade_of_category=hist_star("그외")
        st.pyplot(fig_3)
    with col5:
        st.markdown("#### **상자 그래프** ####")
        fig_4=plt.figure()
        histplot_grade_of_category=box_star("그외")
        st.pyplot(fig_4)
        
if agree=="양식":
    with col4:
        st.markdown("#### **히스토그램 & 밀도그래프** ####")
        fig_3=plt.figure()
        histplot_grade_of_category=hist_star("양식")
        st.pyplot(fig_3)
    with col5:
        st.markdown("#### **상자 그래프** ####")
        fig_4=plt.figure()
        histplot_grade_of_category=box_star("양식")
        st.pyplot(fig_4)
        
if agree=="일식":
    with col4:
        st.markdown("#### **히스토그램 & 밀도그래프** ####")
        fig_3=plt.figure()
        histplot_grade_of_category=hist_star("일식")
        st.pyplot(fig_3)
    with col5:
        st.markdown("#### **상자 그래프** ####")
        fig_4=plt.figure()
        histplot_grade_of_category=box_star("일식")
        st.pyplot(fig_4)

if agree=="카페 및 디저트":
    with col4:
        st.markdown("#### **히스토그램 & 밀도그래프** ####")
        fig_3=plt.figure()
        histplot_grade_of_category=hist_star("카페,디저트")
        st.pyplot(fig_3)
    with col5:
        st.markdown("#### **상자 그래프** ####")
        fig_4=plt.figure()
        histplot_grade_of_category=box_star("카페,디저트")
        st.pyplot(fig_4)

if agree=="분식":
    with col4:
        st.markdown("#### **히스토그램 & 밀도그래프** ####")
        fig_3=plt.figure()
        histplot_grade_of_category=hist_star("분식")
        st.pyplot(fig_3)
    with col5:
        st.markdown("#### **상자 그래프** ####")
        fig_4=plt.figure()
        histplot_grade_of_category=box_star("분식")
        st.pyplot(fig_4)
    
if agree=="중식":
    with col4:
        st.markdown("#### **히스토그램 & 밀도그래프** ####")
        fig_3=plt.figure()
        histplot_grade_of_category=hist_star("중식")
        st.pyplot(fig_3)
    with col5:
        st.markdown("#### **상자 그래프** ####")
        fig_4=plt.figure()
        histplot_grade_of_category=box_star("중식")
        st.pyplot(fig_4)

st.markdown("---")

# 상관관계 분석

# '평균가격' 열 생성 및 초기화
raw_copy["평균 가격"]=0

# 메뉴의 평균가격 변수 생성
for i in range(len(raw_copy["name"])): # 식당의 개수만큼 반복문 실시
    
    # menu-price 변수를 "::"로 나눈후, 이를 통해 메뉴의 개수와 가격을 추출함
    split_1=str(raw_copy["menu-price"][i]).split("::") # menu-price를 "::"를 기준으로 나눈 문자열을 split_1 변수에 할당
    num_menu=len(split_1) # menu-price에 들어있는 메뉴의 개수를 num_menu 변수에 할당
    split_2=str(":".join(split_1)).split(":") # split_1을 ":"를 기준으로 나눈 리스트를 split_2 변수에 할당
    sum_menu=0 # menu의 가격 합 변수 생성 및 초기화
    for k in range(num_menu): # 메뉴의 개수 만큼 반복문 실행
        d=2*k+1
        sum_menu+=int(split_2[d]) # split_2 변수의 홀수 인덱스마다 가격 정보가 들어있으므로, 이를 sum_menu에 더함
    raw_copy.loc[i, "평균 가격"] = sum_menu / num_menu # '평균 가격' 변수에 메뉴들의 평균 가격을 할당함

raw_copy["평균 가격"]=raw_copy["평균 가격"].round(1)

plt.rcParams['font.family'] = 'Arial Unicode MS'


raw_copy=raw_copy.rename(columns={"review_num":"리뷰 개수"})  # review_num 변수의 이름을 '리뷰 개수'로 변경

st.subheader(":blue[상관관계 분석]")

col6, col7, col8=st.columns([3,3,2])

with col6:
    st.markdown("#### **전체 식당의 가격, 평점, 평균 가격 히트맵** ####")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    fig_5=plt.figure(figsize=(6,5))
    
    # 전체 식당의 가격, 평점, 리뷰 개수 사이의 상관관계 히트맵
    heatmap_all=sns.heatmap(data = raw_copy[["평균 가격","평점","리뷰 개수"]].corr(),
                        annot=True, # 각 셀에 숫자값 표시
                        fmt = '.1f', linewidths=.5, cmap='Blues')
    st.pyplot(fig_5)
        

# 전체 식당의 평점과 가격 관계 산점도 그래프
with col7:
    st.markdown("#### **전체 식당의 평점, 평균 가격 관계 그래프** ####")
    
    option = st.radio(
    'type select',
    ('scatter', 'hex')) #  산점도를 hex로 할지 그냥 단순한 산점도를 할지 선택하는 라디오 버튼 생성
    # hex는 육각형 빈도 그래프
    
    if option=="scatter":
        
        # jointplot을 그리는 시간이 너무 오래 걸려서 이미지로 대체함
        # 이미지 URL
        image_url = "https://github.com/rpstss/project-SSUrestaurant/raw/1466667d1e95cd6da3e50e4a91f47de3974f112e/scatter.jpg"
        # 이미지 다운로드
        response = requests.get(image_url) # url의 이미지 가져오기
        image = Image.open(BytesIO(response.content)) # resonse.content는 내용을 바이트 형식으로 변환, 
# resonse.content는 내용을 바이트 형식으로 변환, BytesIO() 는 바이트 형식의 이미지를 메모리 객체로 변환

        # 이미지 출력
        st.image(image)

    

    if option=="hex":
        
        # jointplot을 그리는 시간이 너무 오래 걸려서 이미지로 대체함
        image_url = "https://github.com/rpstss/project-SSUrestaurant/raw/b68ddaf10cc39b793493c770d03d2ef950faf398/hex.jpg"

        # 이미지 다운로드
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        st.image(image)
        

def corr_heatmap(x): # 카테고리별 상관계수 히트맵 작성 함수
    df=raw_copy.query('category=="%s"'%x)
    a=sns.heatmap(data = df[["평균 가격","평점"]].corr(), annot=True, fmt = '.1f', linewidths=.5, cmap='Blues')
    return a   
        
with col8:
    st.markdown("#### **카테고리별 평점, 평균 가격 히트맵** ####")
    selection = st.selectbox(
    'type select',
    ("한식", "그 외", "양식","일식","카페 및 디저트","분식","중식")) # 카테고리 선택박스 생성
    st.write("\n")
    st.write("\n")
    st.write("\n")
    
if selection=="한식":
    with col8:    
        fig_6=plt.figure()
        heatmap_of_category=corr_heatmap("한식")
        st.pyplot(fig_6)


if selection=="그 외":
    with col8:    
        fig_6=plt.figure()
        heatmap_of_category=corr_heatmap("그외")
        st.pyplot(fig_6)

if selection=="양식":
    with col8:    
        fig_6=plt.figure()
        heatmap_of_category=corr_heatmap("양식")
        st.pyplot(fig_6)
        
if selection=="일식":
    with col8:    
        fig_6=plt.figure()
        heatmap_of_category=corr_heatmap("일식")
        st.pyplot(fig_6)

if selection=="카페 및 디저트":
    with col8:    
        fig_6=plt.figure()
        heatmap_of_category=corr_heatmap("카페,디저트")
        st.pyplot(fig_6)

if selection=="분식":
    with col8:    
        fig_6=plt.figure()
        heatmap_of_category=corr_heatmap("분식")
        st.pyplot(fig_6)

if selection=="중식":
    with col8:    
        fig_6=plt.figure()
        heatmap_of_category=corr_heatmap("중식")
        st.pyplot(fig_6)

st.markdown("---")



# 가격대 분포
st.subheader(":blue[가격대 분포]")

col8, col9= st.columns(2)

with col8:
    st.markdown("#### **전체 식당 히스토그램** ####")
    fig_7=plt.figure()
    
    # 전체 식당의 메뉴 평균 가격 분포 히스토그램
    histplot_price_of_all=sns.histplot(data=raw_copy,x="평균 가격",kde=True)  # kde는 커널 밀도 추정 곡선
    st.pyplot(fig_7)


with col9:
    st.markdown("#### **카테고리별 상자 그래프** ####")
    fig_8=plt.figure()
    
    #카테고리별 메뉴 평균 가격 분포 상자그림
    histplot_price_of_category=sns.boxplot(data=raw_copy,y="평균 가격",x="category")
    st.pyplot(fig_8)

st.markdown("---")

# 리뷰 분석
st.subheader(":blue[리뷰 분석]")



col10, col11= st.columns(2)

with col10:
    st.markdown("#### **리뷰 워드 클라우드** ####")
    
    # 워드 클라우드 생성 시간이 너무 오래 걸려서 이미지로 대체
    image_url = "https://github.com/rpstss/project-SSUrestaurant/raw/1466667d1e95cd6da3e50e4a91f47de3974f112e/%EC%9B%8C%EB%93%9C%ED%81%B4%EB%9D%BC%EC%9A%B0%EB%93%9C.png"

    # 이미지 다운로드
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    # 이미지 크기 조정
    resized_image = image.resize((image.width * 2, image.height * 2))

    # Streamlit 앱에 이미지 표시
    st.image(resized_image)
    
with col11:
    st.markdown("#### **단어 개수 상위 15 리뷰 막대 그래프** ####")
    
    # 데이터 전처리 과정 시간이 너무 오래 걸려서 이미지로 대체
    image_url = "https://github.com/rpstss/project-SSUrestaurant/raw/1466667d1e95cd6da3e50e4a91f47de3974f112e/%EC%9B%8C%EB%93%9C%EB%B0%94%ED%94%8C%EB%9E%8F.png"
    # 이미지 다운로드
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))

    # Streamlit 앱에 이미지 표시
    st.image(image)
    
    
st.markdown("---")


# 식당 추천
st.subheader(":blue[식당 추천]")
st.write("식당을 추천해드리겠습니다. 평점과 평균 가격을 각각 몇 대 몇으로 중요하게 생각하는지 알려주세요.")
st.write("(ex. 평점과 가격을 4:6의 정도로 중요하게 생각한다면 각 칸에 4와 6을 입력해주세요.)")

W_score = st.number_input("평점에 대한 가중치", value=5, min_value=0, max_value=10,step=1)
W_price = st.number_input("평균 가격에 대한 가중치", value=5, min_value=0, max_value=10,step=1)

# 평점과 평균가격 변수를 Min-Max 정규화를 하기 위해 사이킷런 모듈 사용
from sklearn.preprocessing import MinMaxScaler 

# Min-Max 스케일링 객체 생성
scaler = MinMaxScaler()

# raw_copy의 복사본 생성
reco = raw_copy.copy()

# 사용자가 선택하는 카테고리만의 데이터 프레임을 만들기 위한 함수 생성
def make_reco(x):
    global reco

    reco = reco.query("category=='%s'"%x)

    reco = reco.drop(["리뷰 개수", "menu-price", "review"], axis=1)

    # MinMaxScaler은 2차원 배열을 기준으로 하므로, 열의 형태를 2차원으로 변경
    ratings = reco["평점"].values.reshape(-1, 1)
    prices = reco["평균 가격"].values.reshape(-1, 1)

    # 변경된 열을 MinMaxScaler에 적용
    reco["평점min_max"] = scaler.fit_transform(ratings)
    reco["가격min_max"] = scaler.fit_transform(prices)

    # 가중치 합 열 생성
    reco["Weight"]=reco["평점min_max"]*W_score + (1-reco["가격min_max"])*W_price
    reco=reco[["name","Weight","평점","평균 가격"]].sort_values("Weight",ascending=False).head()
    
    return reco

col12, col13, col14 = st.columns([2,4,4])

with col12:
     classfy= st.radio("type select",("한식","그 외","양식","일식","카페 및 디저트","분식","중식"),key="restaurant_type") #카테고리 선택 라디오 버튼 생성
    


if classfy=="한식":
    df= make_reco("한식")
    with col13:
        fig_9=plt.figure()
        sns.barplot(data=df,y="name",x="Weight")
        st.pyplot(fig_9)
    with col14:
        st.dataframe(df)  
        
if classfy=="그 외":
    df= make_reco("그외")
    with col13:
        fig_9=plt.figure()
        sns.barplot(data=df,y="name",x="Weight")
        st.pyplot(fig_9)
    with col14:
        st.dataframe(df)
    
        
if classfy=="양식":
    df= make_reco("양식")
    with col13:
        fig_9=plt.figure()
        sns.barplot(data=df,y="name",x="Weight")
        st.pyplot(fig_9)
    with col14:
        st.dataframe(df)
    
if classfy=="일식":
    df= make_reco("일식")
    with col13:
        fig_9=plt.figure()
        sns.barplot(data=df,y="name",x="Weight")
        st.pyplot(fig_9)
    with col14:
        st.dataframe(df)
        
if classfy=="카페 및 디저트":
    df= make_reco("카페,디저트")
    with col13:
        fig_9=plt.figure()
        sns.barplot(data=df,y="name",x="Weight")
        st.pyplot(fig_9)
    with col14:
        st.dataframe(df)
    
if classfy=="분식":
    df= make_reco("분식")
    with col13:
        fig_9=plt.figure()
        sns.barplot(data=df,y="name",x="Weight")
        st.pyplot(fig_9)
    with col14:
        st.dataframe(df)
        
if classfy=="중식":
    df= make_reco("중식")
    with col13:
        fig_9=plt.figure()
        sns.barplot(data=df,y="name",x="Weight")
        st.pyplot(fig_9)
    with col14:
        st.dataframe(df)
