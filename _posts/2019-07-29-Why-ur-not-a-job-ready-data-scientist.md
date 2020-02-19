---
layout: post
title: "당신이 (아직은)준비된 데이터사이언티스트가 아닌 이유"
categories: [Miscellaneous]
---
미디엄에 데이터 사이언티스트 관련 좋은 글이 올라왔다. Worth to translate. 중요 부분만 번역/의역을 진행하였다.<br>
원문: [Why you’re not a job-ready data scientist (yet)](https://towardsdatascience.com/why-youre-not-a-job-ready-data-scientist-yet-1a0d73f15012)
![img](../images/post-img-07-24.jpg)

# 이유 1: 데이터 사이언스에 필요한 Python 스킬
대부분의 데이터 사이언스는 Python 기반이므로, 저는 여기에 중점을 두겠습니다.
<br>
당신을 데이터 과학 이론과 구현의 능력치를 향상시키고 싶다면, 아래의 내용들을 활용해보세요. 아직 시도하지 않았다면 말이죠.  
<br>
## 1. Data exploration
`pandas`의 `.corr()`, `scatter_matrix()`, `.hist()`, `.bar()`와 같은 기능은 실행과 동시에 바로 활용할 줄 알아야 합니다. PCA나 t-SNE(`sklearn`의 `PCA`와 `TSNE`) 등을 이용해서 당신의 데이터를 시각화할 기회를 항상 엿봐야 합니다.
<br>
## 2. Feature selection
90%의 경우, 당신의 데이터셋은 당신이 필요한 것 그 이상으로 많은 feature를 보유할 것입니다. (feature가 많으면 불필요한 학습 시간을 초래하고, 과적합 리스크를 더 높이죠). 기본 필터 방법론(scikit-learn의 `VarianceThreshold`와  `SelectKBest`)과 조금 더 정교한 버전의 모델 기반 feature selection 방법론(`SelectFromModel`을 찾아보세요)에 익숙해지세요.
<br>
## 3. Hyperparameter search for model optimization
`GridSearchCV`가 무엇이고 어떻게 동작하는지 당연히 알아야 합니다. RandomSearchCV도 마찬가지죠. 더 눈에 띄고 싶다면 `skopt`'s `BayesSearchCV`를 실험해보세요. Hyperparameter search에 어떻게 베이지안 최적화를 적용할 수 있을지 알게끔 말이에요.
<br>
## 4. 파이프라인
전처리, feature selection, 모델링 단계를 함께 묶기 위해서 `sklearn`의 라이브러리 `pipeline`을 활용해보세요.  파이프라인에 불편함을 느낀다면 그 데이터 사이언티스트는 모델링 툴킷에 더 익숙해져야 함을 뜻합니다.

<br>
<br>
# 이유 2: 확률과 통계 지식
확률과 통계는 실무에서 노골적으로 등장하지는 않지만, **데이터 사이언스의 모든 과정에서의 기본**이 됩니다. 그러니, 만일 아래의 글을 읽지 않으면 면접에서 참담하게 폭격당하기 쉽습니다.

## 1. 베이즈 이론(Bayes’s theorem)
베이즈 이론은 확률 이론의 근본이며, 인터뷰에 항상 등장합니다. 화이트보드에 직접 써가면서 연습해 보는 것을 권장하고, 이 [유명한 책](http://www.med.mcgill.ca/epidemiology/hanley/bios601/GaussianModel/JaynesProbabilityTheory.pdf?source=post_page---------------------------)의 첫 번째 장을 읽어보면서 룰의 의미와 시초에 대해서 아주 잘 이해하는 것이 좋습니다(읽어보기 재밌는 자료에요!).

## 2. 기초 확률
[이런 질문](https://github.com/kojino/120-Data-Science-Interview-Questions/blob/master/probability.md?source=post_page---------------------------)에 대답할 수 있어야 합니다.
## 3. 모델 평가
예를 들어 분류 문제에 있어서, 정확도를 가장 기본 평가 지표로 삼는 것은 아주 안 좋은 선택입니다. (참고: [Why is accuracy not the best measure for assessing classfication models?](https://stats.stackexchange.com/questions/312780/why-is-accuracy-not-the-best-measure-for-assessing-classification-models?source=post_page---------------------------)
`sklearn`의 `precision_score`, `recall_score`, `f1_score`과 `roc_auc_score` 기능과 그 뒤의 이론에 대해 익숙해져야 합니다. 회귀 문제의 경우, 왜 `mean_absolute_error`보다 `mean_squared_error`를 써야 하는지(혹은 그 반대) 이해하는 일도 매우 중요합니다. `sklearn`의 공식 문서에 있는 모델 평가 지표에 대해 모두 알아보는 일도 권장합니다.

<br>
<br>
# 이유 3: 소프트웨어 엔지니어링 노하우
최근 들어, 데이터 사이언티스트에게 소프트웨어 엔지니어링 작업까지 요구하는 일이 많아지고 있다고 있습니다. 많은 고용주는 지원자에게 clean code와 code 관리에 대한 이해를 요구합니다. 특히나,

## 1. 버전 관리(Version control)
`git`을 어떻게 쓰는지 알아야 하고 커맨드라인을 이용해서 GitHub 레포를 사용할 줄 알아야 합니다. 만약 그렇지 않다면, [이 튜토리얼](https://product.hubspot.com/blog/git-and-github-tutorial-for-beginners?source=post_page---------------------------)을 읽어보길 권고합니다.
## 2. 웹 개발
어떤 회사는 자신의 데이터 사이언티스트가 web app이나 API를 통해 편하게 데이터에 접근함을 선호합니다. 웹 개발의 기본에 대해 편하게 생각하는 일은 매우 중요하고, 이렇게 되기 위해서는 [약간의 Flask를 배우는 것 ](https://www.freecodecamp.org/news/how-to-build-a-web-application-using-flask-and-deploy-it-to-the-cloud-3551c985e492/?source=post_page---------------------------)을 추천합니다.
## 3. 웹 스크래핑
웹 개발과 유사한 이야기긴 한데, 실제 웹사이트에서 스크래핑으로 데이터 수집 자동화를 할 줄 알아야 합니다. `BeautifulSoup`과 `scrapy` 같은 툴을 이용하면서 말이죠.
## 4. 클린 코드(Clean code)
docstrings를 어떻게 쓰는지 배워야 합니다. inline 코멘트를 너무 많이 쓰진 마세요. 당신의 함수를 더 작은 함수로 나눠보세요. 함수 코드가 10줄을 넘어가면 안됩니다. 함수 이름은 이해가 잘 가게 만드세요(`function_1`은 좋은 이름이 아닙니다). python의 관례를 따르고 변수는 underscore를 이용하세요. `like_this` and not `LikeThis`, `likeThis` . Python 모듈( .py files)을 400줄이 넘게 쓰지 마세요. 각 모듈은 목적이 뚜렷해야 합니다(예. `data_processing.py`, `predict.py`). `if name == '__main__':`이 무엇을 뜻하는지 알아야 합니다.  code block이 어떤 역할을 하고 [왜 중요한지](https://stackoverflow.com/questions/419163/what-does-if-name-main-do?source=post_page---------------------------) 알아야 합니다. list comprehension을 쓰세요. `for` loop을 [과하게 쓰지 마세요](https://medium.com/python-pandemonium/never-write-for-loops-again-91a5a4c84baf). 당신의 프로젝트에 `README` 파일을 추가하세요.

<br>
<br>
# 이유 4: 비즈니스 관점의 본능
놀라울 정도로 많은 사람들이 고용 된다는 사실이 당신이 기술적으로 가장 유능한 지원자라는 것을 보여준다고 생각합니다. **사실은 그렇지 않습니다.** 실제로 기업들은 더 많은 돈, 그리고 더 빨리 그 돈을 벌 수있는 사람들을 고용하려 합니다.
일반적으로 우리가 생각하는 tech 기술을 뛰어 넘어, 또 다른 기술 building을 뜻합니다.
## 1. 사람들이 원하는 무언가를 만들라
“data science learning mode”라고 할 때, 다음의 스텝을 따릅니다(import data, explore data, clean data, visualize data, model data, evaluate model). 이런 일은 당신이 새로운 라이브러리나 기술을 배우려 하는 이유라면 괜찮습니다. 하지만 비즈니스 관점에서는 이런 일이 그저 회사 시간(즉 돈)을 쓰는 일일 뿐입니다. 따라서 들이는 시간 대비 당신이 어떤 아웃풋을 낼지 생각하는 습관을 길러야 합니다.
## 2. 알맞은 질문을 해야 한다
모든 회사는 빅픽쳐를 그리고 그에 맞게 자신의 모델을 조정할 줄 아는 사람을 고용하고 싶어합니다. 따라서 데이터 사이언티스는 항상 이런 질문을 해야 합니다. **"이 모델이 정말 우리 팀과 회사에 도움이 되는 일일까? 아니면 단순히 내가 좋아하는 알고리즘을 활용해보고 싶은 것일까?"**, 그리고 **"내가 어떤 비즈니스를 최적화 싶어하는가? 그리고 그렇게 하기 위해 더 나은 방법이 있을까?"**
## 3. 당신의 결과를 설명하라
경영진은 제품 판매 실적이나 사용자가 경쟁 업체를 떠나는 이유 등이 궁금하지, precison이나 recall curve가 무엇인지도 모르고 당신이 모델 과적합을 피하기 위해서 얼마나 노력했는지도 관심조차 없습니다. 따라서 중요한 스킬은 **당신의 분석 결과를 비전문가에게 알기 쉽게 설명하는 일**입니다. 고등학교 때부터 수학을 듣지 않은 친구에게 프로젝트를 소개한다 생각하시면 됩니다.(힌트: 설명에 알고리즘 이름이나 hyperparameter 튜닝 내용이 포함되어서는 안됩니다. **Simple words are better words.**)
