''''
在sklearn中构建机器学习流
'''
from sklearn.preprocessing import StandardScaler 
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
'''
.data
.target
feature_names
target_names
'''
X, y = load_iris(return_X_y=True) 
# 可以通过以下方式将模型类进行机器学习流的集成，只有模型类才能参与构建机器学习流，实用函数不行
pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))

# 进行数据切分
newX, newTX, newY, newTY = train_test_split(X, y, random_state=42)

# fit中有两个步骤，1.进行归一化，2. 再将数据输入到模型中进行建模
pipe.fit(newX, newY)

# 查看精确度
pipe.predict(newTX)

# 查看分数
score = pipe.score(newTX, newTY)
print(score)

# sklearn的模型保存
import joblib

# 保存
joblib.dump(pipe, 'pipe.model')

# 读取
pipe1 = joblib.load('pipe.model')
pipe1.score(newTX, newTY)