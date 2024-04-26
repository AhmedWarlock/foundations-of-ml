from model import LogisticRegression
from utils import accuracy,train_test_split,plot_decision_boundary
from data import generate_data


X, y = generate_data()

X_train, y_train, X_test, y_test = train_test_split(X,y)
print(f" the training shape is: {X_train.shape}")
print(f" the test shape is: {X_test.shape}")

model = LogisticRegression(0.01,n_epochs=10000)
model.fit(X_train,y_train)

ypred_train = model.predict(X_train)
acc = accuracy(y_train,ypred_train)
print(f"The training accuracy is: {acc}")
print(" ")

ypred_test = model.predict(X_test)
acc = accuracy(y_test,ypred_test)
print(f"The test accuracy is: {acc}")

plot_decision_boundary(X_train,model.w,model.w[0],y_train)

plot_decision_boundary(X_test,model.w,model.w[0],y_test)