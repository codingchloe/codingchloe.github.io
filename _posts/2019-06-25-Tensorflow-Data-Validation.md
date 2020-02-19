---
layout: post
title: "Tensorflow가 만들어낸 편리한 Data validation"
categories: [Data, Tensorflow]
---
## Tensorflow Data Validation
Tensorflow가 여러모로 데이터쟁이들에게는 편리한 툴들을 보여주고 있다.
그 중 Data Validation([링크](https://www.tensorflow.org/tfx/guide/tfdv)는 다음과 같다.)이라는 라이브러리를 개발했는데
정말 편리한 View다.

직접 진행한 활용 예시는 추후 업데이트할 예정이지만

### 이게 왜 좋으냐면
보통 데이터 분석이나 모델링을 위한 분석 fullset을 만들기 전에 우리가 가지고 있는 pk(Primary Key)에 대한 정보(Varaibles 또는 column정보라 할 수 있다.)가 제대로 들어왔는지, 그리고 어떤 분포를 보이고 있는지 필수적으로 확인해야 한다.
보통 Excel VBA를 통해서 확인해보았는데 이제는 파이썬의 코드 몇 줄이면 이 결과를 볼 수 있다.

### _작성중_  
