##########################################################
###내장함수 dplyr 비교

#내장함수
exam$tot<-(exam$math+exam$english=exam$science)/3
aggregate(data=exam[exam$math>=50&exam$english>=80, tot~class,mean])

#dplyr
exam %>% 
  filter(math>=50&english>=80) %>% 
  mutate(tot=(math+english+science)/3) %>% 
  group_by(class) %>% 
  summarise(mean=mean(tot))

#내장함수
mpg$tot<-(mpg$cty+mpg$hwy)/2

df_comp<-mpg[mpg$class=="compact",]
df_suv<-mpg[mpg$class=="suv",]

mean(df_comp$tot)
mean(df_suv$tot)

#dplyr
mpg<-as.data.frame(ggplot::mpg)

mpg %>%
  mutate(tot=(cty+hwy)/2)
filter(class=="compact"|class=="suv") %>%
  group_by(class) %>% 
  summarise(mean_tot=mean(tot))


###########################################################
###인터랙티브 그래프

#파일 불러오기
setwd("C:/Users/동현중/Desktop")
install.packages("readxl")
library(readxl)
kospi<-read_excel("data.xls")
head(kospi)

#변수명 변경
library(dplyr)
kospi<-rename(kospi, num=현재지수, date=일자)
kospi<-data.frame(kospi)

#시계열 형태 전환
rownames(kospin)<-kospi$date
kospin<-matrix(kospi$num)
colnames(kospin)<-"num"
kospin<-data.frame(kospin)
head(kospin)
ggplot(kospin,aes(x=kospi$date, y=kospi$num))+geom_line()
?ggplot
#인터랙티브 시계열 그래프
dygraph(kospin)%>%dyRangeSelector()


install.packages("plotly")
library(plotly)
library(ggplot2)

#인터랙티브 그래프
p<-ggplot(data=mpg,aes(x=displ,y=hwy,col=drv))+geom_point()
ggplotly(p)

#인터랙티브 막대그래프
p<-ggplot(data=diamonds,aes(x=cut,fill=clarity))+geom_bar(position="dodge")
ggplotly(p)

#인터랙티브 시계열 그래프
#데이터 불러오기
install.packages("dygraphs")
library(dygraphs)
economics<-ggplot2::economics
head(economics)

#데이터 시간순 정렬
library(xts)
eco<-xts(economics$unemploy,order.by = economics$date)
head(eco)

#인터랙티브 시계열 그래프 작성
dygraph(eco)

#날짜 범위 선택 가능
dygraph(eco) %>% dyRangeSelector()

#여러값 표현
#저축률
eco_a<-xts(economics$psavert,order.by = economics$date)
#실업자수
eco_b<-xts(economics$unemploy/1000,order.by = economics$date)

eco2<-cbind(eco_a,eco_b)
colnames(eco2)<-c("psavert","unemploy")
head(eco2)
dygraph(eco2) %>% dyRangeSelector()

##################################################
###지도 단계 구분도

#패키지 준비
install.packages("ggiraphExtra")
library(ggiraphExtra)
#미국 주별 범죄 데이터 준비
str(USArrests)
head(USArrests)
library(tibble)
crime<-rownames_to_column(USArrests, var="state")
crime$state<-tolower(crime$state)
str(crime)

#미국 주 지도 데이터 준비
install.packages("ggplot2")
library(ggplot2)
install.packages("maps")
states_map<-map_data("state")
str(states_map)

#단계 구분도 만들기
install.packages("mapproj")
ggChoropleth(data=crime,   #지도에 표현할 데이터
             aes(fill=Murder,   #색깔로 표현할 변수
                 map_id=state),   #지역 기준 변수
             map=states_map)    #지도 데이터

#인터랙티브 단계 구분도
ggChoropleth(data=crime,   #지도에 표현할 데이터
             aes(fill=Murder,   #색깔로 표현할 변수
                 map_id=state),   #지역 기준 변수
             map=states_map,    #지도 데이터
             interactive=T)   #인터랙티브-마우스 움직임에 반응



##############################
#대한민국 시도별 인구 단계 구분도
install.packages("stringi")
install.packages("devtools")
devtools::install_github("cardiomoon/kormaps2014")
library(kormaps2014)

#대한민국 시도별 인구 데이터 
str(changeCode(korpop1))
library(dplyr)
korpop1<-rename(korpop1,
                pop=총인구_명,
                name=행정구역별_읍면동)

#단계 구분도 만들기
install.packages("mapproj")
ggChoropleth(data=korpop1,   #지도에 표현할 데이터
             aes(fill=pop,   #색깔로 표현할 변수
                 map_id=code,   #지역 기준 변수
                 tooltip=name),   #지도 위에 표시할 지역명
             map=kormap1,    #지도 데이터
             interactive=T)   #인터랙티브-마우스 움직임에 반응

#결핵환자 단계구분도
str(changeCode(tbc))

ggChoropleth(data=tbc,      #지도에 표현할 데이터
             aes(fill=NewPts,   #색깔로 표현할 변수
                 map_id=code,   #지역 기준 변수
                 tooltip=name),   #지도 위에 표시할 지역명
             map=kormap1,    #지도 데이터
             interactive=T)   #인터랙티브-마우스 움직임에 반응
