
# 주제
### COVID-19 전파 위험도 추정: 물체 및 장면 감지를 사용한 실시간 화면 분석
### Estimation of COVID-19 Transmission Risk: Real-time Screen Analysis using Object and Scene Detection
---
# Table of Contents
* [About the Project](#about-the-project)
* [Building](#Building)
* [Run Screen](#Run-Screen)
* [Member](#Member)

# About The Project
COVID-19의 확산과 지속으로 외출에 대한 불안은 증가하고 있다. 이에 국내 마스크 착용 의무화 지침이 진행되고 있지만, 장소별 위험 정도는 알 수 없다. 이를 해결하기 위해 본 논문은 질병 관리청에서 제시한 방역 수칙을 기준으로 특정 장소의 COVID-19 전파 위험도 측정을 진행한다. 전파 위험도를 측정하기 위한 세 가지 기준은 공간 개폐 여부, 군중 밀집도, 마스크 착용 상태이다. 세 가지 기준에 대한 데이터로 YOLO 모델과 이미지 분류 모델을 학습시키고 Deep Learning 알고리즘을 적용하여 값을 산출한다. 결과적으로 장소의 정보를 수집하여 COVID-19 전파 위험도 평가를 수행하고 CCTV와 같은 Live Feed 영상에 적용해 실시간으로 알려 예방할 수 있도록 기대한다.

# Building
1. installing
```
git clone http://khuhub.khu.ac.kr/2020-2-capstone-design1/BSH_project.git
```
2. Run executable
```
python3 Social_Distance_avg.py
```

# Run Screen
![1604146113797_re](https://user-images.githubusercontent.com/57438644/100606149-57e3a580-334c-11eb-855e-cc0d8dd30dee.png)

# Member
- 2015104147 공재호 asebn121@gmail.com
- 2015104198 이민호 lkjjr0424@khu.ac.kr
- 2015104201 이승민 madcat@khu.ac.kr
- 2015104228 지호진 1004n1996@gmail.com
- 2013104122 최성원 scott.choi1993@khu.ac.kr

