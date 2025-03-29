# DenseNet

- DenseNet (2016.08) - https://arxiv.org/pdf/1608.06993

### Advantage

---

- alleviate the vanishing gradient problem
- strengthen feature propagation
- encourage feature reuse
- substantially reduce the number of parameters

to ensure maximum information flow between layers in the network, we
connect all layers (with matching feature-map sizes) directly with each other.

### DenseNet

---

- ResNet
    - Skip connection을 통해 Vanishing Gradient 문제를 해결
        
        $$
        x_l = H_l(x_{l-1}) + x_{l-1}
        $$
        
    - 이전 레이어의 입력과 레이어의 해당 함수의 출력 값을 summation(합산)으로 결합하기에 네트워크의 information flow를 방해할 수 있다.
- Dense connectivity
    - 레이어들 간 information flow를 더욱 개선하기 위해 모든 레이어에서 모든 후속 레이어로 직접 연결하는 connectivity pattern을 제안
        
        ![image.png](Doc/DenseNet-fig1.png)
        
        $$
        x_l = H_l([x_0, x_1,..., x_{l-1}])
        $$
        
    - 이전 레이어들의 feature map을 summation(합산)으로 결합이 아니라 concatenation으로 연결
    - 구현의 용이성을 위해 함수의 입력을 하나의 텐서로 연결
- Composite function
    - 해당 논문에서는 $H_l()$ 을 BN + ReLU + 3x3 Conv로 정의
- Pooling layers
    - concatenation 연산은 feature map의 크기가 변할 때 사용할 수 없음
    - CNN은 feature map size를 변경하는 down-sampling layer가 필수적
    - Dense Block 사이에 Transition layer를 사용 (1x1 conv, 2x2 avg pooling)
    - 1x1 Conv : 채널 사이즈 절반으로 감소
    - 2x2 avg pooling(stride 2) : 사이즈 절반으로 감소
        
        ![image.png](Doc/DenseNet-fig2.png)
        
- Growth rate
    - feature map size를 K개로 고정
    - hyperparameter k (Growth rate)를 32로 제안
    - 새로운 정보의 양을 조절
- BottleNeck layer
    - 각 레이어는 k개의 feature map을 출력하지만 일반적으로 더 많은 입력이 있다
    - 3x3 conv앞에 1x1 conv를 사용하면 입력 feature map 수를 줄이고 계산 효율성 향상
    - DenseNet-B ⇒ BN-ReLU-Conv(1x1)-BN-ReLU-Conv(3x3)
    - 1x1 conv에서는 4k의 feature map을 생성
		![image.png](Doc/DenseNet-fig3.png)
- Compression
    - model compactness을 더욱 향상시키기 위해 transition layer에서 feature map 수를 줄일 수 있다
    - DenseNet-C
    - DenseNet-B와 결합하여 DenseNet-BC
- Details
    - ImageNet 제외
    - 3개의 Dense block 사용 (동일한 레이어의 수)
        - feature map size : 32×32, 16×16, 8×8
        - {L = 40, k = 12}, {L = 100, k = 12} 및 {L = 100, k = 24}  구성
        - DenseNet-BC :  {L = 100, k = 12}, {L = 250, k = 24} 및 {L = 190, k = 40} 구성
    - 첫 번째 Dense Block에 들어가기 전에 입력 이미지에 대해 16개의 출력 채널로 conv 진행
    - 3x3 conv 경우 feature map size 고정을 위해 zero padding 사용
    - 마지막 dense block 끝에서 GAP(Global Average Pooling) 수행 후 softmax classifier
    - ImageNet
    - 224x224 입력 이미지의 4개의 Dense block 사용
    - 첫 conv는 stride=2가 있는 7x7의 2k 사용
        
        ![image.png](Doc/DenseNet-table1.png)
        

- DenseNet의 변형 → U-Net이라 재미있게 읽은 논문 U-Net 이해에 도움이 됨