# Inception-V4

- paper - https://arxiv.org/pdf/1602.07261

- Inception-V4, Inception-ResNet 제안
    - Inception-V4 : Inception 신경밍을 좀 더 효과적으로 넓고 깊게 만들기 위해 고안. V3보다 단순하고 획일화된 구조와 더 많은 Inception module을 사용
    - Inception-ResNet : Inception V4 + residual connection → 학습 속도가 빨라짐.

### Inception-V4

---

- Inception 모둘이 6개의 종류
- 입력에서 출력으로 갈수록 더 많은 개수의 branch 사용
- 텐서의 크기가 줄어드는 Reduction을 별도로 사용
- V : 패딩을 적용하지 않은 Conv (V표기가 없다면 zero-padding 적용)
    - Inception-V4 전체 구조
        
        ![스크린샷 2025-03-09 오후 3.16.28.png](doc/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-03-09_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_3.16.28.png)
        
    - Stem
        
        ![image.png](doc/image.png)
        
    - Inception-A
        
        ![image.png](doc/image%201.png)
        
    - Inception-B
        
        ![image.png](doc/image%202.png)
        
    - Inception-C
        
        ![image.png](doc/image%203.png)
        
    - Reduction-A
        
        ![image.png](doc/image%204.png)
        
    - Reduction-B
        
        ![image.png](doc/image%205.png)
        

### Inception-ResNet

---

- V1, V2 두 가지 버전
    - Inception-ResNet 구조
        
        ![image.png](doc/image%206.png)
        
    - Stem
        
        
        ![image.png](doc/image%207.png)
        
        ![image.png](doc/image.png)
        
    - Inception-ResNet-A
        
        
        ![image.png](doc/image%208.png)
        
        ![image.png](doc/image%209.png)
        
    - Inception-ResNet-B
        
        
        ![image.png](doc/image%2010.png)
        
        ![image.png](doc/image%2011.png)
        
    - Inception-ResNet-C
        
        
        ![image.png](doc/image%2012.png)
        
        ![image.png](doc/image%2013.png)
        
    - Reduction-A
        
        ![image.png](doc/image%204.png)
        
    - Reduction-B
        
        
        ![image.png](doc/image%2014.png)
        
        ![image.png](doc/image%2015.png)
        

- Filter의 수가 1000개를 초과할 때 Residual Variants가 불안정해져 학습 도중에 네트워크가 죽어버리는 문제 발생 (수만번의 Iteration이 지나면 Average Pooling 이전에 0 값만을 반환)
- 활성화 함수를 적용하기 이전에 잔차를 Scaling Down하여 학습 과정을 완화
    
    ![image.png](doc/image%2016.png)
    
- Inception과 Residual connection 결합의 의의