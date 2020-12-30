# 패키지 설치
REPO_URL = "https://cran.seoul.go.kr/"
if (!require(dplyr))     install.packages("dplyr", repos=REPO_URL)
if (!require(tidyverse)) install.packages("tidyverse", repo=REPO_URL)
if (!require(stringr))   install.packages("stringr", repos=REPO_URL)
if (!require(rJava))     install.packages("rJava", repos=REPO_URL)
if (!require(memoise))   install.packages("memoise", repos=REPO_URL)
if (!require(KoNLP))     install.packages("KoNLP", repos=REPO_URL)
if (!require(wordcloud)) install.packages("wordcloud", repos=REPO_URL)
if (!require(extrafont)) install.packages("extrafont", repos=REPO_URL)

# 패키지 로드
library(dplyr)
library(readr)        # 파일 읽기 기능 제공 (tidyverse패키지에 포함됨)
library(stringr)      # 문자열 관련 기능 제공 패키지
library(rJava)        # KoNLP가 의존함 (Java기능 호출 패키지)
library(memoise)      # KoNLP가 의존함
library(KoNLP)        # 한글데이터 형태소 분석 패키지 (이름 대소문자 주의)
library(wordcloud)    # 워드클라우드 생성 패키지
library(RColorBrewer) # 색상 제어 패키지
library(extrafont)    # 폰트관리 패키지


# 시스템의 폰트 파일들중 이름에 D2라는 단어가 포함된 폰트를 R 디렉토리 안으로 복사한다.
# --> 오랜 시간이 걸리더라도 수행할지 여부를 묻는 "y/n" 확인이 필요하면 "y"를 입력 후 엔터
font_import(pattern="NanumGothic.ttf")
# 폰트 로드 --> 운영체제에 맞게 설정하세요.
loadfonts(device="win")       # Windows
#loadfonts()  # Mac

# 이 메시지가 출력될 때 까지 다음을 진행하지 마세요.
print("----------- 폰트스캔 완료 -----------");

# 폰트테이블 확인
fonts <- fonttable()
# 중복된 이름을 제거하고 출력
unique(fonts$FamilyName)

# 원격지 텍스트 파일 읽기(readr 패키지 사용)
txt <- read.table("new.txt", fill = TRUE , header = FALSE )
# 500글자만 추출하여 확인
useNIADic()

nouns <- extractNoun(txt)
nouns

wordcount <- table(unlist(nouns))
df_word <- as.data.frame(wordcount, stringsAsFactors = FALSE)
head(df_word, 20)
df_word <- rename(df_word, 단어=Var1, 빈도수=Freq)
df_word <- df_word[-1,]
head(df_word, 20)

# 두 글자 이상 단어 추출 > 역순정렬
result_df <- df_word %>% filter(nchar(단어) >= 2) %>% arrange(desc(빈도수))
result_df

options(repr.plot.width=13, repr.plot.height=13, warn=-1)
display.brewer.all()


pal <- brewer.pal(10,"Dark2")
pal

# 그래픽 사이즈 설정
options(repr.plot.width=8, repr.plot.height=8, warn=-1)

# 랜덤값 고정 -> 실행시마다 동일한 모양으로 생성되도록 함
set.seed(1234)

# 워드클라우드 생성
wordcloud(words = result_df$단어,    # 단어
          freq = result_df$빈도,     # 빈도
          min.freq = 2,             # 최소 단어 빈도
          max.words = 1000,         # 표현 단어 수
          random.order = FALSE,     # 고빈도 단어 중앙 배치
          random.color = FALSE,     # 색상으로 빈도 표현 여부
          scale = c(10, 1),         # 단어 크기 범위
          colors = pal,             # 색깔 목록
          family="NanumGothic")     # 사용할 폰트








###########################
setwd("C:/Users/Administrator/Desktop")
txt<-read.table("new.txt", fill = TRUE , header = FALSE )

#특수문자 제거
txt<-str_replace_all(txt,"\\W"," ")
nouns<-extractNoun(txt)

#추출한 명사 list를 문자열 벡터로 변환, 단어별 빈도표 생성
wordcount<-table(unlist(nouns))

#데이터 프레임으로 변환
df_word<-as.data.frame(wordcount,stringsAsFactors=F)

#변수명 수정
df_word<-rename(df_word, word=Var1,Freq=Freq)

#두 글자 이상 단어 추출
df_word<-filter(df_word,nchar(word)>=2)


REPO_URL = "https://cran.seoul.go.kr/"
if (!require(wordcloud2))     install.packages("wordcloud2", repos=REPO_URL)
library(wordcloud2)
wordcloud2(data = head(df_word, 1000), fontFamily = '나눔고딕')
wordcloud2(data = head(df_word,1000),    # 데이터프레임
           fontFamily = '나눔고딕',        # 사용할 글꼴
           fontWeight = 'normal',         # 글꼴의 굵기 (normal or bold)
           size = 1.3,                    # 글꼴크기(기본값=1)
           minSize= 0.3,                  # 글꼴의 최소 크기
           backgroundColor = "#ffffff",   # 배경색상
           widgetsize = c(800, 600),      # 위젯의 크기 (가로,세로) 픽셀 형식의 벡터
           color=brewer.pal(11, "RdYlGn") # 색상 팔래트 적용 (단일 값인 경우 단색으로 지정됨)
)
