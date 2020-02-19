---
layout: post
title: "Ad Tech에 쓰이는 머신러닝(1)"
categories: [MachineLearning, Data, AdTech]
---
# Ad Tech 산업에는 어떤 머신러닝 기법이 활용되고 있을까?

데이터로 할 수 있는 일이 정말 많은 산업이다. 그러므로 위의 질문에 대한 답을 하려면 연재 편으로 가야 할 만큼 답이 길어질 수 있다. 해당 산업을 훑어보는 견해로써 우선은 Real-Time Bidding과 관련하여 여러 자료를 찾아보았다.

## Real-Time Bidding이란?

우리는 컴퓨터나 스마트폰을 보는 거의 매 순간 display advertisement에 노출된다. 웹사이트나 앱 화면 내에 보이는 광고를 말한다. 화면 영역 내 한 자리에 대해서는 물론 자릿값이 존재한다. 이 자릿값은 bidding 형태로 이루어진다. 유저가 사이트에 입장하면, 유저의 쿠키 데이터가 어딘가로 전송이 된다. 광고주들은 이 데이터와 자신의 보유 데이터를 이용하여 적당한 가격을 제시하게 된다. 여러 광고주가 참여한 옥션이 시작되고 옥션에서의 승자가 결국 광고를 띄울 수 있게 된다. 이 과정은 우리가 화면을 클릭 후 페이지 전환이 되는 순간 이뤄진다. 그래서 RTB(Real-Time Bidding)이라 불린다.

## RTB에 대해 보기 전, Ad Tech 내 데이터를 이용하는 사례
1. **User targeting**
<br>
 유저 타겟팅 방법론은 여러 가지가 있다. site category targeting, domain/app-specific targeting, contextual targeting based on page content, first-party targeting, time/external condition-based targeting, demographic, gender, age, device targeting 등등. 이름은 다양하게 있지만, 목적은 타겟하고 싶은 유저군을 찾는 것이다. 무엇을 타겟하고 싶냐에 따라, 어떤 유형의 유저를 찾느냐에 따라 방법론은 다 다를 것이다. data dependency가 매우 높은 작업이기 때문에 어떤 방법론이 보편적이고 특이점은 이러하다고 써내려가기가 어렵다. real world에서 실제 데이터를 보면서 어떤 문제를 어떻게 정의해 나가느냐가 가장 중요한 작업으로 보인다.
<br>
2. **RTB**
<br>
3. **User data enhancement**
<br>
불필요한 데이터 찾기, 필요한 데이터 끌어내기 등등 수행해야 할 업무에 대한 data quality 개선 작업과도 같다.
<br>
이 밖에도  캠페인 관련 데이터 작업, A/B Test 등 매우 중요하면서도 수시로 진행해야 하는 작업도 있고, Fraud detection과 같이 내가 아직은 조사해보지 않은 부분도 있다.

<br>
## 그럼 RTB 작업에 대해서 좀 더 살펴보자

RTB의 목적은 어떤 유저가 광고를 한 번 더 클릭함으로써 구매를 진행하게 되는 것, 즉 광고주가 <u>정해진 예산 내 최대의 순이익 효과</u>를 보는 것이다.
Display Advertising의 시나리오를 본다면 과정마다 어떤 작업이 필요한지 예상할 수 있다.

### Display Advertising의 시나리오
0. The user visits one publisher’s web/app page
1. a bid request for the corresponding ad display opportunity, along with its information about the underlying user, the domain context and the auction information
2. broadcasts to all the possible advertisers for bid via ad exchange
3. through DSP(Demand-side platforms), advertisers estimates the potential campaign effectiveness and possible cost(=bid price) for the received bid request
4. each advertiser makes final decision of the bid price and send it to the ad exchange
5. the ad exchange determine the winner(who proposed the highest bid price), and the winner pay the second highest price(=market price)
6. the winner sends the ad to the user
7. user responses to the ad that he/she sees

<br>
### 목적을 달성하기 위해 진행해야 하는 3가지 Task
1. **CTR(click-through rate) or CVR(conversion rate) estimation**<br>
  User response prediction으로 캠페인 효과를 확보하는 일이다.

2. **Cost estimation**<br>
  market price 혹은 winning price를 예측

3. **Bidding price optimization**<br>
  1번과 2번의 결과를 캠페인 집행 예산(campaign budget)이나 경쟁 정도(auction volume)와 함께 반영하여 최적의 bidding price 결정한다. (재밌는게 이 3번의 이야기가 더해지면 경제학 관점의 사례가 굉장히 많이 나오겠다.)

### 각 Task별 활용 model 예시

1. **CTR(click-through rate) or CVR(conversion rate) estimation** <br>
  유저가 클릭을 할지/안할지, 전환을 할지/안할지를 에측하는 모델로 target value가 binary 값이니, linear model로는 <u>logistic regression</u>이 있겠고, non-linear model로는 <u>tree-based model</u>, <u>factorization machines</u>이 있다. 이 외에도 Bayesian profit regression, FTRL-trained factorization machines 등이 있다.
  <br>
  성능 지표로는 ROC curve, relative information gain를 활용한다.
<br>

2. **Cost estimation**
<br>
  Market price 분포를 예측하는 일이 목적이다. 조사한 바로는 non-linear regression(i.e. gradient boosting decision trees)을 이용하여 각 샘플에 해당하는 bid를 예측한다. log-normal distributes된  bid price를 campaign-level로 formalize 한다. (분포 변형에 대한 부분이 잘 와닿지 않는다.)
<br>

3. **Bidding price optimization**
<br>
	linear-bidding strategy(the bid price is calculated via the predicted CTR/CVR multiplied by a constant parameter tuned according to the campaign budget and performance)
