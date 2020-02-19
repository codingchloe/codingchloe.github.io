---
layout: post
title: "Overfitting이란?"
categories: [DeepLearning]
---
# Overfitting이란?
다음은 설립년도에 따른 매출액 변화를 볼 수 있는 데이터(점으로 표현한 부분)에 대한 예측 모델입니다.
첫 번째 그림은 2차 다항식 회귀모형입니다.
![img1](https://t1.daumcdn.net/thumb/R1280x0/?fname=http://t1.daumcdn.net/brunch/service/user/19LF/image/rZb1CfwJnamH_YrizSq1n3n-U6g.png)

이차 방정식이 단순 선형 방정식보다 실제 데이터에 더 잘 맞는 것을 알 수 있습니다. 그렇다면 이 경우에는 2차 회귀 모형의 R-Sqaure값이 단순 선형 회귀의 것보다 크겠죠? 네, 맞습니다. 왜냐하면 2차 회귀 모형이 실제 데이터와 유사한 모습을 보이니까요. 일반적으로 2차 및 3차 다항식이 많이 활용되지만, 더 높은 차수의 다항식을 대입해 볼 수도 있습니다.

아래는 같은 데이터에 대한 6차 다항식 회귀모형입니다.
![img2](https://t1.daumcdn.net/thumb/R1280x0/?fname=http://t1.daumcdn.net/brunch/service/user/19LF/image/cQ2Bp8sZnk-nAPbw7Y0N9lCrf_E.png)

이 그래프만 본다면 2차 다항식 회귀모형보다는 고차원 방정식이 실제 데이터 값을 더 잘 예측하는 것처럼 보입니다. 그렇다면 위 예제와 같은 경우에는 항상 고차원 방정식을 쓰는 것이 맞다고 생각하시나요? 아쉽게도 그렇지 않습니다. 만일 위의 그림과 같은 model을 산출하였다면 그것은 train set에 잘 맞는 것이지, train set이 아닌 다른 데이터 셋(test set)에서는 그 관계 설명에 있어서 실패할 수 있기 때문입니다. 위의 6차 방정식을 test set에 적용한다면 성능은 train set에서 얻어진 것 대비 현저히 떨어질 수 있습니다. 이러한 문제를 **과적합(over-fitting)** 이라고 합니다. 다른 말로 표현한다면 모델이 **high variance와 low bias를 가지고 있다** 고 말할 수 있습니다.

# Bias and Variance
bias와 variance는 정확히 무슨 뜻일까요? 양궁 과녁의 예를 보면서 이해해봅시다.
![img3](https://t1.daumcdn.net/thumb/R1280x0/?fname=http://t1.daumcdn.net/brunch/service/user/19LF/image/zmr0ZWD59JFcr7y15b723MMyPO8.png)

만일 우리가 error값이 작은 모형을 완성했다고 가정해봅시다. 이는 low bias, low variance를 가지고 있는 왼쪽 상단의 그림을 뜻하는 거겠지요. 보시다시피 모든 데이터 점들은 다 빨간색 과녁에 위치하고 있습니다. 여기서 variance가 증가하게 된다면, 데이터 점들의 분산은 예측력을 좀 더 떨어뜨리게 될 것입니다. 그리고 bias가 커지게 되면 실제값과 예측값의 오차는 커지게 되구요.

그렇다면 완벽한 model을 만들기 위해서는 bias와 variance가 어떻게 균형을 이뤄야 할까요?
![img4](https://t1.daumcdn.net/thumb/R1280x0/?fname=http://t1.daumcdn.net/brunch/service/user/19LF/image/Lw0rKacYWMPipX3Wc-xCQIBHbMk.png)

우리가 모형에 더 많은 변수를 넣게되면, 복잡성은 증가하게 됩니다. 복잡성이 증가하면, variance는 늘어나고 bias는 줄게 됩니다. 따라서 우리는 bias의 감소가 variance의 증가와 같아지는 최적의 point를 찾아야 합니다. 그것이 모델의 Overfitting을 줄일 수 있는 방법입니다.

Overfitting을 해결할 수 있는 방법론에 대한 설명은 다음 글들을 참고하시면 됩니다.
>
### 1. [L1 vs. L2 Regularization](https://codingchloe.github.io/2018-09-26/L1-vs-L2-Regularization)
