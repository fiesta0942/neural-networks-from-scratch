Q1 Features?
X = 30 x 455 matrix -> 455명의 환자와 각 환자당 30개의 특성(feature)
특성이라는건 weight와 곱해져서 얼마나 암인지 (0 or 1)를 판단하게 해주는 역할

Q2. loss, cost, cross - entropy loss function의 차이?
loss function: sample 1개의 대한 오차를 계산하는 함수 (L). 환자 1명에 대해 얼마나 정답과 가까운지?
cost function: evaluation of "all" samples. J(w,b) = 1/m * sigma (np.sum) L_i. 455명을 얼마나 잘 판별하는지?
cross - entropy loss: specific type of loss function. L = -[y*log(a) + (1-y)*log(1-a)]. 

Q3. 간단한 Git
git status: 어떤 파일이 바뀌었는지? (아직 commit되지 않은 것)
git add: 다음 commit에 포함된 파일은? (git add README.md -> readme 파일을 commit 할 준비)
git commit -m: add로 포함한 파일들을 "설명"과 함께 저장
git push: commit한 파일들을 실제로 github에 업로드