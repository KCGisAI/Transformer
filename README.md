https://www.notion.so/1-1439d23dd141809abb5eccc56214c8b5 (notion Attention is all you need Review)

# Abstract

논문의 주제가 나오는 가장 중요한 부분이라고 생각한다.

글에서 New simple network achitecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirly 라고 소개 한다.

즉, Recurrence 와 convolutions을 버리고 오로지 Attention mechanism만 사용했다는 것이다. 그래서 논문 주제가 Attention is all you need인 것이다. 주제를 너무 잘 만들었다고 생각 한다..

# Model Architecture

입, 출력이 모두 Sequence인 경우 Encoder와 Decoder가 있는 것이 가장 성능이 좋다고 한다. 

Encoder를 통해 Input Sequence(x) →  Continuous representation(z) 이후 Decoder가 Output(y)을 만든다.

하나씩 곱씹어 보면, 우리가 사용하는 단어들은 컴퓨터가 이해하기 어렵기 때문에 변환이 필요하다. 

이러한 변환이 곧 Encoder가 해주며 변환 작업(Word Embedding)의 결과가 Continuous representation(z)이다. 

이후 Decoder에서 다시 사람이 이해 할 수 있는 언어로 변환하여 y가 출력이 된다.

아래 Figure1이 논문에서 설명되어져있는 Model Architecture 이며 

좌측이 Encoder 우측이 Decoder이다.

![논문 Attention is all you need Figure 1 Model architecture 발췌](https://prod-files-secure.s3.us-west-2.amazonaws.com/2918753c-fd69-49a2-8ade-cc7cb2740b19/bec270a6-5076-4186-afe5-727fc199cd30/image.png)

논문 Attention is all you need Figure 1 Model architecture 발췌

Model Architecture을 이제부터 하나씩 이해해보자. 

비록 영어로 적혀있고 단어들이 복잡해 보이지만, 가장 중요한 것은 흐름을 이해하고 구현하는 것이라고 생각한다.

Inputs ~ Outputs 까지 단계를  차근차근 이해해보자.

## Input Embedding

단순하게 생각하면 인간이 사용하는 자연어를 컴퓨터가 이해할 수 있도록 벡터화 시켜주는 행위가 바로 Input Embedding이다. (pytorch를 이용해서 간단히 구현 가능하다.)

## Positional Encoding

Abstract에서 언급한 것처럼 Recurrence를 버렸다고 했다. 하지만, 우리가 사용하는 언어는 순서가 있다. 예를 들어, ‘나는 강아지보다 고양이가 더 좋다’ 의 문장이 있는데 강아지와 고양이의 위치가 바뀌면 정보가 정 반대로 된다.

위와 같은 문제를 방지하기 위해 Input Embedding 해준 값들에 위치 정보를 추가해 주는 것이다.

합쳐지기 위해선 ‘당연히’ Input Embedding 된 값과 동일 차원이다.

논문에선 하기 공식을 이용한다.

![논문 Attention is all you need 3.5 Positional Encoding 부분에서 발췌](https://prod-files-secure.s3.us-west-2.amazonaws.com/2918753c-fd69-49a2-8ade-cc7cb2740b19/5c452733-a1a0-4916-b52c-ed012879bed5/image.png)

논문 Attention is all you need 3.5 Positional Encoding 부분에서 발췌

Input Embedding과 Positional Encoding을 이어서 생각해보자.

예시문은 I love cat, embedding vector의 size는 1x3(model dimension=3)이라고 가정하자

| I | v1 | v2 | v3 |
| --- | --- | --- | --- |
| love | v4 | v5 | v6 |
| cat | v7 | v8 | v9 |

그러면 위처럼 embedding 값이 생성되고 Positional Encoding을 위해 아래와 같이 가정하고 position 정보를 더해준다

| I | p1 | p2 | p3 |
| --- | --- | --- | --- |
| love | p4 | p5 | p6 |
| cat | p7 | p8 | p9 |

수식의 pos는 embedding vector의 위치(행), i는 차원 인덱스(열) 이라고 생각하면 된다. 즉, 짝수 차원 인덱스은 Sin함수 홀수 차원 인덱스는 Cos 함수로 positional encoding을 해주는것이다.

## Scaled Dot-product Attention

Multi head attention을 설명하기 전 거쳐가는 부분이다.

![scaled dot product attention 식 논문 Attention is all you need 3.2.1-(1) 발췌](https://prod-files-secure.s3.us-west-2.amazonaws.com/2918753c-fd69-49a2-8ade-cc7cb2740b19/6eea9012-d2f6-4146-90fc-bce21aa512c3/image.png)

scaled dot product attention 식 논문 Attention is all you need 3.2.1-(1) 발췌

![논문 Attention is all you need Figure 2 발췌](https://prod-files-secure.s3.us-west-2.amazonaws.com/2918753c-fd69-49a2-8ade-cc7cb2740b19/79b8db61-844d-4cd7-a439-aaf377c7ac83/image.png)

논문 Attention is all you need Figure 2 발췌

위의 그림이 Scaled Dot Product Attention이다. 단어가 어려워 보이지만 쪼개서 생각하면 그렇지는 않다.

내 생각대로 표현한다면 아래와 같다

Scaled : 비례적으로 변화된

Dot Product : 내적한(상관도 분석)

Attention : 어텐션 알고리즘(어느 부분에 더 가중치를 줘야할까?)

더 자세히 설명하자면, 논문에서 model dimension의 sqrt만큼 나누어주는데 그 이유는 매우 큰 model dimension을 가지면 내적의 값이 커져 Softmax 함수의 기울기가 매우 작은 영역에 들어가게 된다(=Gradient Vanishing) 그래서! “Scaled”를 해주는 것이다.

그렇다면 왜 Dot Product일까? 기하학적인 성질을 확인해보자

![출처 [https://flaria.wordpress.com/2010/07/18/스칼라곱-dot-product/](https://flaria.wordpress.com/2010/07/18/%EC%8A%A4%EC%B9%BC%EB%9D%BC%EA%B3%B1-dot-product/)](https://prod-files-secure.s3.us-west-2.amazonaws.com/2918753c-fd69-49a2-8ade-cc7cb2740b19/25869043-fea0-4e45-8f45-d2cf4c5202a6/image.png)

출처 [https://flaria.wordpress.com/2010/07/18/스칼라곱-dot-product/](https://flaria.wordpress.com/2010/07/18/%EC%8A%A4%EC%B9%BC%EB%9D%BC%EA%B3%B1-dot-product/)

A라는 벡터를 B에 대해 정사영 해서 비교한다고 생각 할 수 있다.

theta라는 각도에 따라서 정사영 값이 바뀔 것이다 예를 들어 각도가 0이면 최대값이되고 직각을 이룬다면 0이 될 것이다.

따라서, A와 B의 벡터의 유사성이 높다면 theta는 작은값을 가지는 것처럼 내적을 통해 유사성을 확인 할 수 있는 것이다.

위와 같은 이유로 단어 선정에는 다 이유가 있다.

다시 나만의 단어로 재정리 하면, Gradient Vanishing 문제를 방지하기 위해 Scaled를 해주고 벡터 내적을 통해 유사성을 확인한 뒤, 어떤 단어에 더 가중치를 줘야 판단하는 기능인 것이다.

## Multi Head Attention

위에서 설명한 Scaled dot product attention은 Single head Attention이라고도 한다. 

그렇다면 단순히 Multi Head Attention의 차이는 head가 1개 or 1개 이상의 차이이다.

왜 Multi Head가 Single Head보다 좋을까?

직관적으로 말하면 Head(사람의 실제 머리) 라고 생각하면 쉽다.

Single Head(1명) 이서 생각하는 것보다는 Multi Head(여러명)이 같이 생각하는 것이 서로 다른 관점에서 해석 가능하기 때문이다.

위의 예시를 조금 더 학술적으로 얘기한다면,  병렬의 Multi Head를 사용함으로 여러 부분에 동시에 Attention을 가할 수 있다,

1개의 Head는 명사에 집중 하고 다른 Head는 관계에 집중하는 Attention 이렇게 다른 관점에서 Attention을 할 수 있어 입력 토큰 간의 더 복잡한 관계를 다룰 수 있다.

![논문 Attention is all you need Figure2](https://prod-files-secure.s3.us-west-2.amazonaws.com/2918753c-fd69-49a2-8ade-cc7cb2740b19/2e80f70c-beea-4f8f-8966-1c343e6cf9f5/image.png)

논문 Attention is all you need Figure2

위는 Multi Head Attention의 구조이다. 수식은 사실 Single Attention과 동일하지만 Head가 추가된다.

예를 들면, 3x3의 Input, head의 개수는 3이라고 가정 하자

위 그림과 같이 Head의 개수만큼 나누기 위해서는 Input을 3등분한다.

즉, 3x3의 Q,K,V가 3x1의 Q,K,V가 3개 등장하는 것이다. 그럼 Attention Value의 크기는 똑같이 3x1이며 3개의 Attention Value를 Cocat하면

3x3의 동일한 차원을 가지게 된다.
