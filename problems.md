# 1. torch.Normalize의 잘못된 사용
Normalize를 넣으면 loss가 nan으로 가는 것 때문에 발견한 문제.

torch.Normalize 를 Standardzation(일명 z-score Normalization)으로 알고 있었다.

하지만, torch.Normalize 는 주어진 값에서 평균을 빼고, 표준편차로 나눠주는 단순한 변형이다.

즉 평균과 표준편차를 알아서 계산해주거나 하는게 아니다.

굳이 Standardzation으로 사용하려면 각 각의 색상마다 평균과 표준편차를 구해서 넣어줘야 한다.
# 2. predictor 의 layer 가 너무 깊었음
# 3. g(f(x)) 뒤에 predictor를 붙이는 것이 아니라, f(x) 뒤에 붙여야 함
위 두 문제는 feature을 상당부분 잃어버릴 수 있었음
이는 논문에도 나와 있었음
# 4. Optimizer의 잘못된 적용
Adam optimizer는 batch size가 클 때 수렴하지 못한다고 논문에 적혀 있었음.
따라서 LARS Optimizer를 사용
# 5. predictor는 레이어가 작고 일반적인 학습
따라서 batch size를 32 혹은 64로 적용


# 결론

논문을 좀 더 꼼꼼히 읽고 적용해야겠다.